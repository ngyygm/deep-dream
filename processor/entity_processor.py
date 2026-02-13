"""
实体处理模块：实体搜索、对齐、更新/新建
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import uuid

from .models import Entity, MemoryCache
from .storage import StorageManager
from .llm_client import LLMClient


class EntityProcessor:
    """实体处理器 - 负责实体的搜索、对齐、更新和新建"""
    
    def __init__(self, storage: StorageManager, llm_client: LLMClient,
                 max_similar_entities: int = 10, content_snippet_length: int = 50):
        self.storage = storage
        self.llm_client = llm_client
        self.max_similar_entities = max_similar_entities
        self.content_snippet_length = content_snippet_length
    
    def process_entities(self, extracted_entities: List[Dict[str, str]], 
                        memory_cache_id: str, similarity_threshold: float = 0.7,
                        memory_cache: Optional[MemoryCache] = None, doc_name: str = "",
                        context_text: Optional[str] = None,
                        extracted_relations: Optional[List[Dict[str, str]]] = None,
                        jaccard_search_threshold: Optional[float] = None,
                        embedding_name_search_threshold: Optional[float] = None,
                        embedding_full_search_threshold: Optional[float] = None,
                        on_entity_processed: Optional[callable] = None) -> Tuple[List[Entity], List[Dict], Dict[str, str]]:
        """
        处理抽取的实体：搜索、对齐、更新/新建
        
        Args:
            extracted_entities: 抽取的实体列表（每个包含name和content）
            memory_cache_id: 当前记忆缓存的ID
            similarity_threshold: 相似度阈值（用于搜索，作为默认值）
            memory_cache: 当前记忆缓存对象（可选，用于LLM判断时提供上下文）
            doc_name: 文档名称（只保存文档名，不包含路径）
            context_text: 可选的上下文文本（当前处理的文本片段），用于精细化判断时提供场景信息
            extracted_relations: 步骤3抽取的关系列表（用于判断是否已存在关系）
            jaccard_search_threshold: Jaccard搜索（name_only）的相似度阈值（可选，默认使用similarity_threshold）
            embedding_name_search_threshold: Embedding搜索（name_only）的相似度阈值（可选，默认使用similarity_threshold）
            embedding_full_search_threshold: Embedding搜索（name+content）的相似度阈值（可选，默认使用similarity_threshold）
        
        Returns:
            Tuple[处理后的实体列表, 待处理的关系列表, 实体名称到ID的映射]
            关系信息格式：{"entity1_name": "...", "entity2_name": "...", "content": "...", "relation_type": "alias|normal"}
            注意：关系中的实体使用名称而不是ID，因为新实体在创建前还没有ID
        """
        processed_entities = []
        pending_relations = []  # 待处理的关系（使用实体名称）
        entity_name_to_id = {}  # 实体名称到ID的映射（包括新创建的实体）
        
        # 检测同名实体（在同一批处理中）
        name_to_entities = {}  # 用于检测同名实体
        for extracted_entity in extracted_entities:
            entity_name = extracted_entity['name']
            if entity_name not in name_to_entities:
                name_to_entities[entity_name] = []
            name_to_entities[entity_name].append(extracted_entity)
        
        # 构建当前抽取的实体名称集合（用于过滤候选实体）
        extracted_entity_names = {e['name'] for e in extracted_entities}
        
        # 构建步骤3抽取的关系集合（用于判断是否已存在关系）
        # 使用实体对（名称）和内容哈希作为键
        extracted_relation_pairs = set()
        if extracted_relations:
            for rel in extracted_relations:
                entity1_name = rel.get('entity1_name') or rel.get('from_entity_name', '').strip()
                entity2_name = rel.get('entity2_name') or rel.get('to_entity_name', '').strip()
                content = rel.get('content', '').strip()
                if entity1_name and entity2_name:
                    # 使用排序后的实体对作为键（关系是无向的）
                    pair_key = tuple(sorted([entity1_name, entity2_name]))
                    content_hash = hash(content.lower())
                    extracted_relation_pairs.add((pair_key, content_hash))
        
        # 处理每个实体
        total_entities = len(extracted_entities)
        for idx, extracted_entity in enumerate(extracted_entities, 1):
            entity, relations, name_mapping = self._process_single_entity(
                extracted_entity, 
                memory_cache_id, 
                similarity_threshold,
                memory_cache,
                doc_name,
                context_text,
                entity_index=idx,
                total_entities=total_entities,
                extracted_entity_names=extracted_entity_names,
                extracted_relation_pairs=extracted_relation_pairs,
                jaccard_search_threshold=jaccard_search_threshold,
                embedding_name_search_threshold=embedding_name_search_threshold,
                embedding_full_search_threshold=embedding_full_search_threshold
            )
            if entity:
                processed_entities.append(entity)
                # 更新实体名称到ID的映射（包括原始名称和最终名称）
                # 检查是否已存在同名实体
                if entity.name in entity_name_to_id:
                    existing_id = entity_name_to_id[entity.name]
                    if existing_id != entity.entity_id:
                        print(f"    ⚠️  警告: 发现同名实体 '{entity.name}'，已有ID: {existing_id}，新ID: {entity.entity_id}")
                        # 保留第一个（或使用版本数更多的，但这里简化处理，保留第一个）
                        # 实际应该通过LLM判断是否应该合并
                else:
                    entity_name_to_id[entity.name] = entity.entity_id
                
                # 原始名称映射
                original_name = extracted_entity['name']
                if original_name in entity_name_to_id:
                    existing_id = entity_name_to_id[original_name]
                    if existing_id != entity.entity_id:
                        print(f"    ⚠️  警告: 原始名称 '{original_name}' 已映射到不同ID: {existing_id} vs {entity.entity_id}")
                else:
                    entity_name_to_id[original_name] = entity.entity_id
            
            if relations:
                pending_relations.extend(relations)
            
            # 更新映射（合并已有的映射，但检查冲突）
            if name_mapping:
                for name, eid in name_mapping.items():
                    if name in entity_name_to_id:
                        existing_id = entity_name_to_id[name]
                        if existing_id != eid:
                            print(f"    ⚠️  警告: 映射冲突，名称 '{name}' 已有ID: {existing_id}，新ID: {eid}，保留已有ID")
                        # 保留已有ID，不覆盖
                    else:
                        entity_name_to_id[name] = eid
            
            # 如果提供了回调函数，在处理完每个实体后调用
            if on_entity_processed and entity:
                on_entity_processed(entity, entity_name_to_id, pending_relations)
        
        return processed_entities, pending_relations, entity_name_to_id
    
    def _process_single_entity(self, extracted_entity: Dict[str, str], 
                               memory_cache_id: str, 
                               similarity_threshold: float,
                               memory_cache: Optional[MemoryCache] = None,
                               doc_name: str = "",
                               context_text: Optional[str] = None,
                               entity_index: int = 0,
                               total_entities: int = 0,
                               extracted_entity_names: Optional[set] = None,
                               extracted_relation_pairs: Optional[set] = None,
                               jaccard_search_threshold: Optional[float] = None,
                               embedding_name_search_threshold: Optional[float] = None,
                               embedding_full_search_threshold: Optional[float] = None) -> Tuple[Optional[Entity], List[Dict], Dict[str, str]]:
        """
        处理单个实体
        
        流程：
        6.1 初步筛选：判断当前抽取的实体与检索到的实体列表，是否需要合并或存在关系
        6.2 精细化判断：对需要处理的候选进行详细判断，决定合并/创建关系/新建实体
        6.3 创建新实体并分配ID，更新关系边中的实体名称到ID映射
        
        Returns:
            Tuple[处理后的实体, 待处理的关系列表（使用实体名称）, 实体名称到ID的映射]
        """
        entity_name = extracted_entity['name']
        entity_content = extracted_entity['content']
        
        # 显示进度信息
        if total_entities > 0:
            print(f"  ├─ 处理实体 [{entity_index}/{total_entities}]: {entity_name}")
        else:
            print(f"  ├─ 处理实体: {entity_name}")
        
        # 步骤1：使用混合搜索策略搜索相关实体并合并结果
        # 为三种搜索方法分别设置阈值（如果未指定，使用默认的similarity_threshold）
        jaccard_threshold = jaccard_search_threshold if jaccard_search_threshold is not None else min(similarity_threshold, 0.6)
        embedding_name_threshold = embedding_name_search_threshold if embedding_name_search_threshold is not None else min(similarity_threshold, 0.6)
        embedding_full_threshold = embedding_full_search_threshold if embedding_full_search_threshold is not None else min(similarity_threshold, 0.6)
        
        # 模式0：只用name检索（使用jaccard）
        candidates_jaccard = self.storage.search_entities_by_similarity(
            entity_name,
            query_content=None,
            threshold=jaccard_threshold,
            max_results=self.max_similar_entities,
            content_snippet_length=self.content_snippet_length,
            text_mode="name_only",
            similarity_method="jaccard"
        )
        
        # 模式1：只用name检索（使用embedding）
        candidates_name_embedding = self.storage.search_entities_by_similarity(
            entity_name,
            query_content=None,
            threshold=embedding_name_threshold,
            max_results=self.max_similar_entities,
            content_snippet_length=self.content_snippet_length,
            text_mode="name_only",
            similarity_method="embedding"
        )
        
        # 模式2：使用name+content检索（使用embedding）
        candidates_full_embedding = self.storage.search_entities_by_similarity(
            entity_name,
            query_content=entity_content,
            threshold=embedding_full_threshold,
            max_results=self.max_similar_entities,
            content_snippet_length=self.content_snippet_length,
            text_mode="name_and_content",
            similarity_method="embedding"
        )
        
        # 输出三种搜索方法的结果
        print(f"  │  ├─ Jaccard搜索（name_only）: {len(candidates_jaccard)} 个")
        if candidates_jaccard:
            jaccard_names = [e.name for e in candidates_jaccard[:5]]
            print(f"  │  │   {', '.join(jaccard_names)}{'...' if len(candidates_jaccard) > 5 else ''}")
        
        print(f"  │  ├─ Embedding搜索（name_only）: {len(candidates_name_embedding)} 个")
        if candidates_name_embedding:
            name_embedding_names = [e.name for e in candidates_name_embedding[:5]]
            print(f"  │  │   {', '.join(name_embedding_names)}{'...' if len(candidates_name_embedding) > 5 else ''}")
        
        print(f"  │  ├─ Embedding搜索（name+content）: {len(candidates_full_embedding)} 个")
        if candidates_full_embedding:
            full_embedding_names = [e.name for e in candidates_full_embedding[:5]]
            print(f"  │  │   {', '.join(full_embedding_names)}{'...' if len(candidates_full_embedding) > 5 else ''}")
        
        # 合并结果并去重（按entity_id去重，保留每个entity_id的最新版本）
        entity_dict = {}
        for entity in candidates_jaccard + candidates_name_embedding + candidates_full_embedding:
            if entity.entity_id not in entity_dict:
                entity_dict[entity.entity_id] = entity
            else:
                # 保留物理时间最新的
                if entity.physical_time > entity_dict[entity.entity_id].physical_time:
                    entity_dict[entity.entity_id] = entity
        
        similar_entities = list(entity_dict.values())
        
        # 过滤候选实体：如果候选实体在当前抽取列表中，且与当前实体已经存在关系，则跳过
        # 因为步骤3（关系抽取）应该已经处理过这些实体之间的关系了
        if extracted_entity_names and extracted_relation_pairs:
            filtered_similar_entities = []
            skipped_count = 0
            for candidate in similar_entities:
                # 如果候选实体的name与当前实体name相同，保留（可能是合并的情况）
                if candidate.name == entity_name:
                    filtered_similar_entities.append(candidate)
                # 如果候选实体的name不在当前抽取列表中，保留（需要判断是否与新实体有关系）
                elif candidate.name not in extracted_entity_names:
                    filtered_similar_entities.append(candidate)
                else:
                    # 候选实体的name在当前抽取列表中，且与当前实体name不同
                    # 检查是否在步骤3的关系中已经存在这两个实体之间的关系
                    pair_key = tuple(sorted([entity_name, candidate.name]))
                    # 检查是否有任何关系（不检查内容哈希，因为可能有不同的关系描述）
                    has_relation = any(
                        pair[0] == pair_key for pair in extracted_relation_pairs
                    )
                    if has_relation:
                        # 已经存在关系，跳过
                        skipped_count += 1
                        print(f"  │  │  ├─ {candidate.name}: 跳过已有关系（步骤3已处理）")
                    else:
                        # 虽然都在当前列表中，但还没有关系，需要判断
                        filtered_similar_entities.append(candidate)
            
            similar_entities = filtered_similar_entities
            if skipped_count > 0:
                print(f"  │  跳过 {skipped_count} 个已在当前抽取列表且已存在关系的候选实体（步骤3已处理）")
        
        # # 如果合并后超过最大数量，按物理时间排序，保留最新的
        # if len(similar_entities) > self.max_similar_entities:
        #     similar_entities.sort(key=lambda e: e.physical_time, reverse=True)
        #     similar_entities = similar_entities[:self.max_similar_entities]
        
        if not similar_entities:
            # 没有找到相似实体，直接新建
            new_entity = self._create_new_entity(entity_name, entity_content, memory_cache_id, doc_name)
            print(f"  │  未找到相似实体，创建新实体: {new_entity.entity_id}")
            # 返回实体、空关系列表、实体名称到ID的映射
            entity_name_to_id = {
                entity_name: new_entity.entity_id,
                new_entity.name: new_entity.entity_id
            }
            return new_entity, [], entity_name_to_id
        
        print(f"  │  找到 {len(similar_entities)} 个候选实体")
        
        # 步骤2：找到同ID下最新的实体（去重）
        # 按entity_id分组，每个entity_id只保留最新版本
        entity_dict = {}
        for entity in similar_entities:
            if entity.entity_id not in entity_dict:
                entity_dict[entity.entity_id] = entity
            else:
                # 保留物理时间最新的
                if entity.physical_time > entity_dict[entity.entity_id].physical_time:
                    entity_dict[entity.entity_id] = entity
        
        unique_entities = list(entity_dict.values())
        
        # 步骤3：准备已有实体信息供LLM分析
        # 构建实体组：当前抽取的实体（作为第一个，即"当前分析的实体"）+ 候选实体
        entities_group = [
            {
                'entity_id': 'NEW_ENTITY',  # 标记为新实体
                'name': entity_name,
                'content': entity_content,
                'version_count': 0
            }
        ]
        
        # 添加候选实体信息
        for e in unique_entities:
            version_count = len(self.storage.get_entity_versions(e.entity_id))
            entities_group.append({
                'entity_id': e.entity_id,
                'name': e.name,
                'content': e.content,
                'version_count': version_count
            })
        
        # 步骤4：检查实体对之间是否已有关系（只检查已有实体之间，不包括新实体）
        existing_relations_between = {}
        existing_entity_ids = [e.entity_id for e in unique_entities]
        for i, eid1 in enumerate(existing_entity_ids):
            for eid2 in existing_entity_ids[i+1:]:
                pair_key = "|".join(sorted([eid1, eid2]))
                relations = self.storage.get_relations_by_entities(eid1, eid2)
                if relations:
                    existing_relations_between[pair_key] = [
                        {'relation_id': r.relation_id, 'content': r.content}
                        for r in relations
                    ]
        
        # 步骤5：使用两步流程分析（初步筛选 + 精细化判断）
        print(f"  │  调用LLM分析（候选数: {len(unique_entities)}）")
        
        # 阶段1：初步筛选（使用content snippet快速筛选）
        preliminary_result = self.llm_client.analyze_entity_candidates_preliminary(
            entities_group,
            content_snippet_length=self.content_snippet_length,
            context_text=context_text
        )
        
        possible_merges = preliminary_result.get("possible_merges", [])
        possible_relations = preliminary_result.get("possible_relations", [])
        no_action = preliminary_result.get("no_action", [])
        
        # 阶段2：精细化判断（对筛选出的候选使用完整content进行精确判断）
        # 收集需要精细化判断的候选（只处理merge和relation，no_action不处理）
        candidates_to_analyze = {}
        for item in possible_merges:
            cid = item.get("entity_id") if isinstance(item, dict) else item
            if cid != 'NEW_ENTITY':
                candidates_to_analyze[cid] = {"type": "merge", "reason": item.get("reason", "") if isinstance(item, dict) else ""}
        
        # 对于被判断为关系的候选，先检查是否已有关系，如果有则跳过精细化判断
        skipped_relations_count = 0
        skipped_entity_names = []  # 记录跳过的实体名称
        for item in possible_relations:
            cid = item.get("entity_id") if isinstance(item, dict) else item
            if cid != 'NEW_ENTITY':
                # 注意：由于当前实体是 NEW_ENTITY（尚未保存），无法直接查询关系
                # 但为了保持代码一致性，仍然进行检查（应该总是返回空）
                # 如果未来支持检查候选实体之间的关系，可以在这里扩展
                existing_rels = self.storage.get_relations_by_entities("NEW_ENTITY", cid)
                if existing_rels and len(existing_rels) > 0:
                    # 已有关系，跳过精细化判断
                    skipped_relations_count += 1
                    candidate_entity = next((e for e in unique_entities if e.entity_id == cid), None)
                    if candidate_entity:
                        skipped_entity_names.append(candidate_entity.name)
                else:
                    # 没有关系，需要精细化判断
                    if cid not in candidates_to_analyze:
                        candidates_to_analyze[cid] = {"type": "relation", "reason": item.get("reason", "") if isinstance(item, dict) else ""}
                    else:
                        # 如果同时出现在merge和relation中，优先考虑relation（更保守）
                        candidates_to_analyze[cid]["type"] = "relation"
        
        # 输出初步筛选结果（只统计需要精细化判断的候选）
        relation_count = len([c for c in candidates_to_analyze.values() if c['type'] == 'relation'])
        if skipped_relations_count > 0:
            print(f"  │  ├─ 初步筛选: 合并 {len([c for c in candidates_to_analyze.values() if c['type'] == 'merge'])} 个, 关系 {relation_count} 个 (跳过已有关系: {skipped_relations_count} 个), 跳过 {len(no_action)} 个")
        else:
            print(f"  │  ├─ 初步筛选: 合并 {len([c for c in candidates_to_analyze.values() if c['type'] == 'merge'])} 个, 关系 {relation_count} 个, 跳过 {len(no_action)} 个")
        
        # 准备当前实体信息（新实体）
        current_entity_info = {
            "entity_id": "NEW_ENTITY",
            "name": entity_name,
            "content": entity_content,
            "version_count": 0
        }
        
        # 对每个候选进行精细化判断
        merge_decisions = []  # 精细化判断后确定要合并的，包含候选实体信息
        relation_decisions = []  # 精细化判断后确定要创建关系的
        
        # 如果有需要精细化判断的候选，先打印开始提示
        if candidates_to_analyze:
            skipped_info = ""
            if skipped_entity_names:
                skipped_info = f"，跳过已有关系: {', '.join(skipped_entity_names)}"
            print(f"  │  ├─ 精细化判断开始（共 {len(candidates_to_analyze)} 个候选{skipped_info}）")
        
        for cid, info in candidates_to_analyze.items():
            candidate_entity = next((e for e in unique_entities if e.entity_id == cid), None)
            if not candidate_entity:
                continue
            
            candidate_info = {
                "entity_id": cid,
                "name": candidate_entity.name,
                "content": candidate_entity.content,
                "version_count": len(self.storage.get_entity_versions(cid))
            }
            
            # 获取两个实体之间的已有关系
            existing_rels = self.storage.get_relations_by_entities("NEW_ENTITY", cid)
            # 由于NEW_ENTITY不存在，需要检查已有实体之间的关系
            # 这里简化处理，只检查候选实体与其他已有实体之间的关系
            existing_relations_list = []
            
            # 调用精细化判断（传入上下文文本）
            detailed_result = self.llm_client.analyze_entity_pair_detailed(
                current_entity_info,
                candidate_info,
                existing_relations_list,
                context_text=context_text
            )
            
            action = detailed_result.get("action", "no_action")
            reason = detailed_result.get("reason", "")
            relation_content = detailed_result.get("relation_content", "")
            merge_target = detailed_result.get("merge_target", "")
            
            if action != "no_action":
                print(f"  │  │  ├─ {candidate_entity.name}: {action}", end="")
                if action == "create_relation" and relation_content:
                    print(f" - {relation_content[:50]}{'...' if len(relation_content) > 50 else ''}")
                else:
                    print()
            
            if action == "merge":
                merge_target_id = merge_target if merge_target and merge_target != "NEW_ENTITY" else cid
                merge_decisions.append({
                    "target_entity_id": merge_target_id,
                    "source_entity_id": "NEW_ENTITY",
                    "candidate_entity_id": cid,  # 记录候选实体ID，用于后续收集content
                    "candidate_content": candidate_entity.content,  # 记录候选实体content
                    "candidate_name": candidate_entity.name,  # 记录候选实体名称
                    "reason": reason
                })
            elif action == "create_relation":
                # 确保有关系描述
                if not relation_content:
                    # 如果没有提供关系描述，根据原因生成一个简单描述
                    relation_content = f"{entity_name}与{candidate_entity.name}存在关联关系"
                    if reason:
                        relation_content = f"{reason[:100]}{'...' if len(reason) > 100 else ''}"
                
                relation_decisions.append({
                    "entity1_id": "NEW_ENTITY",
                    "entity2_id": cid,
                    "entity1_name": entity_name,
                    "entity2_name": candidate_entity.name,
                    "content": relation_content,
                    "reason": reason
                })
            elif action == "no_action":
                # 即使返回no_action，也记录一下，方便调试
                print(f"        不处理: {entity_name} 与 {candidate_entity.name} 无明确关联")
        
        # 输出最终分析结果
        if merge_decisions or relation_decisions:
            print(f"  │  └─ 精细化判断: 合并 {len(merge_decisions)} 个, 关系 {len(relation_decisions)} 个")
        
        # 步骤6.1和6.2：处理分析结果（合并决策和关系决策）
        final_entity = None
        pending_relations = []  # 待处理的关系（使用实体名称，因为新实体还没有ID）
        entity_name_to_id = {}  # 实体名称到ID的映射
        other_targets_entities = {}  # 存储其他目标实体的信息（在合并前收集，合并后这些ID就不存在了）
        
        # 6.1-6.2：处理合并决策
        # 如果有多个合并决策，需要选择一个主要目标实体
        # 策略：优先选择版本数最多的实体作为目标
        if merge_decisions:
            # 收集所有目标实体ID
            target_entity_ids = [d.get("target_entity_id") for d in merge_decisions 
                                if d.get("target_entity_id") and d.get("target_entity_id") != 'NEW_ENTITY']
            
            if target_entity_ids:
                # 如果所有合并决策都指向同一个目标，直接使用
                if len(set(target_entity_ids)) == 1:
                    primary_target_id = target_entity_ids[0]
                    other_targets = []  # 没有其他目标
                else:
                    # 如果有多个不同的目标，选择版本数最多的作为主要目标
                    target_version_counts = {}
                    for tid in target_entity_ids:
                        if tid not in target_version_counts:
                            versions = self.storage.get_entity_versions(tid)
                            target_version_counts[tid] = len(versions)
                    
                    primary_target_id = max(target_entity_ids, key=lambda tid: target_version_counts.get(tid, 0))
                    
                    # 输出多个合并目标的信息
                    other_targets = [tid for tid in set(target_entity_ids) if tid != primary_target_id]
                    if other_targets:
                        print(f"  │  ├─ 注意: 多个合并目标，选择版本数最多的实体作为主要目标")
                        print(f"  │  │   主要目标: {primary_target_id} (版本数: {target_version_counts.get(primary_target_id, 0)})")
                        for tid in other_targets:
                            print(f"  │  │   其他目标: {tid} (版本数: {target_version_counts.get(tid, 0)})")
                        
                        # 在合并之前，先收集其他目标实体的信息（合并后这些ID就不存在了）
                        other_targets_entities.clear()  # 清空之前的数据
                        for other_target_id in other_targets:
                            other_entity = self.storage.get_entity_by_id(other_target_id)
                            if other_entity:
                                other_targets_entities[other_target_id] = {
                                    'entity': other_entity,
                                    'name': other_entity.name,
                                    'content': other_entity.content
                                }
                        
                        # 如果有多个不同的目标实体ID，说明这些实体都是同一个实体
                        # 需要将其他目标实体ID合并到主要目标ID
                        print(f"  │  ├─ 执行实体ID合并: 将 {len(other_targets)} 个实体ID合并到主要目标ID")
                        merge_result = self.storage.merge_entity_ids(primary_target_id, other_targets)
                        if merge_result.get("entities_updated", 0) > 0:
                            print(f"  │  │   已合并 {merge_result.get('entities_updated', 0)} 条实体记录")
                        if merge_result.get("relations_updated", 0) > 0:
                            print(f"  │  │   已更新 {merge_result.get('relations_updated', 0)} 条关系记录")
                        
                        # 更新映射：将所有指向旧实体ID的映射更新为新的 primary_target_id
                        # 这确保映射中不会保留指向已合并ID的失效映射
                        updated_mapping_count = 0
                        for name, eid in list(entity_name_to_id.items()):
                            if eid in other_targets:
                                entity_name_to_id[name] = primary_target_id
                                updated_mapping_count += 1
                        if updated_mapping_count > 0:
                            print(f"  │  │   已更新 {updated_mapping_count} 个映射，将旧ID映射更新为 primary_target_id: {primary_target_id}")
                        
                        # 处理合并后产生的自指向关系（暂时跳过，因为entity_processor中没有这个方法）
                        # 自指向关系会在后续的consolidate_knowledge_graph_entity中处理
                
                # 合并新实体到主要目标实体
                latest_entity = self.storage.get_entity_by_id(primary_target_id)
                if latest_entity:
                    target_name = latest_entity.name
                    
                    # 收集所有需要合并到主要目标的实体的content
                    # 包括：主要目标实体 + 新实体 + 所有指向主要目标的候选实体 + 被合并到主要目标的其他目标实体
                    contents_to_merge = [latest_entity.content, entity_content]
                    entities_to_merge_names = [latest_entity.name, entity_name]
                    
                    # 收集被合并到主要目标的其他目标实体的content（如果有多个不同的目标实体ID）
                    # 注意：这些实体ID已经在合并前被收集到 other_targets_entities 中，因为合并后这些ID就不存在了
                    if len(set(target_entity_ids)) > 1 and other_targets_entities:
                        for other_target_id, other_info in other_targets_entities.items():
                            other_content = other_info.get('content')
                            other_name = other_info.get('name')
                            if other_content:
                                # 检查是否已经添加（通过内容比较，避免重复）
                                if not any(other_content == content for content in contents_to_merge):
                                    contents_to_merge.append(other_content)
                                    entities_to_merge_names.append(other_name or f"实体{other_target_id}")
                    
                    # 收集所有指向主要目标的候选实体的content
                    for merge_decision in merge_decisions:
                        candidate_target_id = merge_decision.get("target_entity_id")
                        candidate_entity_id = merge_decision.get("candidate_entity_id")
                        candidate_content = merge_decision.get("candidate_content")
                        candidate_name = merge_decision.get("candidate_name")
                        
                        # 如果这个合并决策指向主要目标，且候选实体不是主要目标本身
                        if candidate_target_id == primary_target_id and candidate_entity_id and candidate_entity_id != primary_target_id:
                            # 添加候选实体的content（如果还没有添加，避免重复）
                            if candidate_content:
                                # 检查是否已经添加（通过内容比较，避免重复）
                                if not any(candidate_content == content for content in contents_to_merge):
                                    contents_to_merge.append(candidate_content)
                                    entities_to_merge_names.append(candidate_name or f"实体{candidate_entity_id}")
                    
                    # 判断是否需要更新
                    need_update = self.llm_client.judge_content_need_update(
                        latest_entity.content,
                        entity_content
                    )
                    
                    if need_update:
                        # 合并名称
                        if entity_name != latest_entity.name:
                            merged_name = self.llm_client.merge_entity_name(
                                latest_entity.name,
                                entity_name
                            )
                        else:
                            merged_name = entity_name
                        
                        # 合并内容：统一使用多实体合并方法（2个实体是多实体的特殊情况）
                        merged_content = self.llm_client.merge_multiple_entity_contents(contents_to_merge)
                        print(f"  │  ├─ 合并 {len(contents_to_merge)} 个实体的content: {', '.join(entities_to_merge_names[:3])}{'...' if len(entities_to_merge_names) > 3 else ''}")
                        
                        # 创建新版本
                        final_entity = self._create_entity_version(
                            primary_target_id,
                            merged_name,
                            merged_content,
                            memory_cache_id,
                            doc_name
                        )
                        print(f"  │  ├─ 合并到: {target_name} (已更新)")
                    else:
                        final_entity = latest_entity
                        print(f"  │  ├─ 合并到: {target_name} (无需更新)")
                    
                    # 更新映射：原始名称和目标实体名称都映射到目标实体ID
                    entity_name_to_id[entity_name] = primary_target_id
                    entity_name_to_id[final_entity.name] = primary_target_id
        
        # 6.2：处理关系决策（记录关系，但使用实体名称，因为新实体可能还没有ID）
        for rel_info in relation_decisions:
            entity1_name = rel_info.get("entity1_name", entity_name)
            entity2_name = rel_info.get("entity2_name", "")
            content = rel_info.get("content", "")
            
            # 判断关系类型
            relation_type = "normal"
            if "别名" in content or "称呼" in content or "简称" in content:
                relation_type = "alias"
            
            print(f"  │  ├─ 关系: {entity1_name} <-> {entity2_name}")
            
            # 关系使用实体名称，ID将在步骤6.3中更新
            pending_relations.append({
                "entity1_name": entity1_name,  # 当前抽取的实体名称
                "entity2_name": entity2_name,  # 候选实体名称
                "content": content,
                "relation_type": relation_type
            })
        
        # 步骤6.3：如果没有匹配或合并，创建新实体并分配ID
        if not final_entity:
            # 检查是否有匹配的实体（通过分析结果判断）
            matched = len(merge_decisions) > 0
            
            if not matched:
                # 没有匹配，创建新实体
                final_entity = self._create_new_entity(entity_name, entity_content, memory_cache_id, doc_name)
                print(f"  │  ├─ 创建新实体: {final_entity.entity_id}")
                # 更新映射：新创建的实体
                entity_name_to_id[entity_name] = final_entity.entity_id
                entity_name_to_id[final_entity.name] = final_entity.entity_id
        
        # 步骤6.3：更新关系边中的实体名称到ID映射
        # 对于pending_relations中的关系，如果涉及当前实体（entity1_name），更新为实际的entity_id
        updated_relations = []
        for rel in pending_relations:
            if rel["entity1_name"] == entity_name:
                # 当前实体已创建，更新为entity_id
                updated_rel = rel.copy()
                updated_rel["entity1_id"] = final_entity.entity_id if final_entity else None
                updated_relations.append(updated_rel)
            else:
                # 保持原样（entity2_name是已有实体，将在步骤7中处理）
                updated_relations.append(rel)
        
        # 输出最终结果
        if final_entity:
            if updated_relations:
                print(f"  └─ 完成: {final_entity.name} ({final_entity.entity_id}), 关系 {len(updated_relations)} 个")
            else:
                print(f"  └─ 完成: {final_entity.name} ({final_entity.entity_id})")
        else:
            if updated_relations:
                print(f"  └─ 完成: 关系 {len(updated_relations)} 个")
        
        return final_entity, updated_relations, entity_name_to_id
    
    def _create_new_entity(self, name: str, content: str, memory_cache_id: str, doc_name: str = "") -> Entity:
        """创建新实体"""
        entity_id = f"ent_{uuid.uuid4().hex[:12]}"
        entity_record_id = f"entity_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 只保存文档名，不包含路径
        doc_name_only = doc_name.split('/')[-1] if doc_name else ""
        
        entity = Entity(
            id=entity_record_id,
            entity_id=entity_id,
            name=name,
            content=content,
            physical_time=datetime.now(),
            memory_cache_id=memory_cache_id,
            doc_name=doc_name_only
        )
        
        self.storage.save_entity(entity)
        return entity
    
    def _create_entity_version(self, entity_id: str, name: str, content: str, 
                              memory_cache_id: str, doc_name: str = "") -> Entity:
        """创建实体的新版本"""
        entity_record_id = f"entity_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 只保存文档名，不包含路径
        doc_name_only = doc_name.split('/')[-1] if doc_name else ""
        
        entity = Entity(
            id=entity_record_id,
            entity_id=entity_id,
            name=name,
            content=content,
            physical_time=datetime.now(),
            memory_cache_id=memory_cache_id,
            doc_name=doc_name_only
        )
        
        self.storage.save_entity(entity)
        return entity
    
    def get_entity_by_name(self, entity_name: str) -> Optional[Entity]:
        """根据名称获取实体（返回最新版本）"""
        # 使用name_only模式，更精确
        similar_entities = self.storage.search_entities_by_similarity(
            entity_name,
            text_mode="name_only",
            similarity_method="embedding"
        )
        if similar_entities:
            # 返回第一个（已经是最新的）
            return similar_entities[0]
        return None
