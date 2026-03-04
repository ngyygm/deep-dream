"""
关系处理模块：关系搜索、对齐、更新/新建
"""
from typing import List, Dict, Optional
from datetime import datetime
import uuid

from .models import Relation
from .storage import StorageManager
from .llm_client import LLMClient


class RelationProcessor:
    """关系处理器 - 负责关系的搜索、对齐、更新和新建"""
    
    def __init__(self, storage: StorageManager, llm_client: LLMClient):
        self.storage = storage
        self.llm_client = llm_client
    
    def process_relations(self, extracted_relations: List[Dict[str, str]], 
                         entity_name_to_id: Dict[str, str],
                         memory_cache_id: str, doc_name: str = "") -> List[Relation]:
        """
        处理抽取的关系：去重合并、搜索、对齐、更新/新建
        
        Args:
            extracted_relations: 抽取的关系列表（每个包含entity1_name, entity2_name, content）
            entity_name_to_id: 实体名称到entity_id的映射
            memory_cache_id: 当前记忆缓存的ID
            doc_name: 文档名称（只保存文档名，不包含路径）
        
        Returns:
            处理后的关系列表（已保存到数据库）
        """
        # 步骤1：对相同实体对的关系进行去重和合并
        merged_relations = self._dedupe_and_merge_relations(
            extracted_relations, entity_name_to_id
        )
        
        # 步骤2：处理合并后的关系（搜索、对齐、更新/新建）
        processed_relations = []
        
        for merged_relation in merged_relations:
            # 检查关系格式是否正确
            if not isinstance(merged_relation, dict):
                print(f"警告：关系格式不正确，跳过: {merged_relation}")
                continue
            
            # 检查必需的字段（支持新旧格式）
            entity1_name = merged_relation.get('entity1_name') or merged_relation.get('from_entity_name', '')
            entity2_name = merged_relation.get('entity2_name') or merged_relation.get('to_entity_name', '')
            
            if not entity1_name or not entity2_name:
                print(f"警告：关系缺少必需字段，跳过: {merged_relation}")
                continue
            
            # 获取实体ID
            entity1_id = entity_name_to_id.get(entity1_name)
            entity2_id = entity_name_to_id.get(entity2_name)
            
            if not entity1_id or not entity2_id:
                print(f"警告：无法找到实体ID - entity1: {entity1_name}, entity2: {entity2_name}")
                continue
            
            # 检查两个实体是否是同一个实体（跳过自关系）
            if entity1_id == entity2_id:
                print(f"警告：跳过自关系（两个实体是同一个） - {entity1_name} ({entity1_id})")
                continue
            
            relation = self._process_single_relation(
                merged_relation,
                entity1_id,
                entity2_id,
                memory_cache_id,
                entity1_name,
                entity2_name,
                doc_name=doc_name
            )
            
            if relation:
                processed_relations.append(relation)
        
        return processed_relations
    
    def _dedupe_and_merge_relations(self, extracted_relations: List[Dict[str, str]],
                                    entity_name_to_id: Dict[str, str]) -> List[Dict[str, str]]:
        """
        对相同实体对的关系进行去重和合并
        
        Args:
            extracted_relations: 抽取的关系列表
            entity_name_to_id: 实体名称到entity_id的映射
        
        Returns:
            去重合并后的关系列表
        """
        # 按实体对分组（使用标准化后的实体对，使关系无向化）
        relations_by_pair = {}
        filtered_count = 0
        filtered_relations = []
        
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
                continue
            
            # 检查实体ID是否存在
            missing_entities = []
            if entity1_name not in entity_name_to_id:
                missing_entities.append(f'entity1: {entity1_name}')
            if entity2_name not in entity_name_to_id:
                missing_entities.append(f'entity2: {entity2_name}')
            
            if missing_entities:
                filtered_count += 1
                filtered_relations.append({
                    'entity1': entity1_name,
                    'entity2': entity2_name,
                    'reason': f'实体不在当前窗口的实体列表中: {", ".join(missing_entities)}'
                })
                continue
            
            # 检查两个实体是否是同一个实体（通过entity_id比较）
            entity1_id = entity_name_to_id.get(entity1_name)
            entity2_id = entity_name_to_id.get(entity2_name)
            
            if entity1_id and entity2_id and entity1_id == entity2_id:
                filtered_count += 1
                filtered_relations.append({
                    'entity1': entity1_name,
                    'entity2': entity2_name,
                    'reason': f'两个实体是同一个实体（entity_id: {entity1_id}）'
                })
                continue
            
            # 标准化实体对（按字母顺序排序，使关系无向化）
            # 使用LLMClient的标准化方法
            normalized_pair = LLMClient._normalize_entity_pair(entity1_name, entity2_name)
            
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
        
        # 输出过滤统计信息
        if filtered_count > 0:
            # 统计不同类型的过滤原因
            missing_entity_count = sum(1 for f in filtered_relations if '实体不在当前窗口' in f['reason'])
            self_relation_count = sum(1 for f in filtered_relations if '两个实体是同一个实体' in f['reason'])
            empty_name_count = sum(1 for f in filtered_relations if '实体名称为空' in f['reason'])
            
            print(f"[关系过滤] ⚠️  共过滤了 {filtered_count} 个关系")
            if missing_entity_count > 0:
                print(f"  - 实体不在当前窗口的实体列表中: {missing_entity_count} 个")
            if self_relation_count > 0:
                print(f"  - 自关系（两个实体是同一个）: {self_relation_count} 个")
            if empty_name_count > 0:
                print(f"  - 实体名称为空: {empty_name_count} 个")
            
            if missing_entity_count > 0:
                print(f"  当前窗口的实体列表包含 {len(entity_name_to_id)} 个实体: {', '.join(list(entity_name_to_id.keys())[:10])}{'...' if len(entity_name_to_id) > 10 else ''}")
            
            print(f"  被过滤的关系示例（前5个）:")
            for i, filtered in enumerate(filtered_relations[:5], 1):
                entity1 = filtered.get('entity1', filtered.get('from', ''))
                entity2 = filtered.get('entity2', filtered.get('to', ''))
                print(f"    {i}. {entity1} <-> {entity2} ({filtered['reason']})")
            if len(filtered_relations) > 5:
                print(f"    ... 还有 {len(filtered_relations) - 5} 个关系被过滤")
        
        if len(merged_relations) > 0:
            print(f"[关系过滤] ✅ 通过过滤的关系: {len(merged_relations)} 个（去重合并后）")
        
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
        
        # 打印合并信息
        print(f"[关系操作] 🔀 合并关系: {pair[0]} <-> {pair[1]} (共{len(relation_contents)}个关系)")
        for i, content in enumerate(relation_contents, 1):
            print(f"  关系{i} content:")
            print(f"    {content[:200]}{'...' if len(content) > 200 else ''}")
        print(f"  合并后content:")
        print(f"    {merged_content[:200]}{'...' if len(merged_content) > 200 else ''}")
        
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
                                 memory_cache_id: str,
                                 entity1_name: str = "",
                                 entity2_name: str = "",
                                 verbose_relation: bool = True,
                                 doc_name: str = "") -> Optional[Relation]:
        """
        处理单个关系
        
        注意：参数 entity1_id 和 entity2_id 是实体的 entity_id（不是绝对ID）
        在创建关系时，会通过 entity_id 获取实体的最新版本，然后使用绝对ID存储
        
        流程：
        1. 根据两个实体ID查找所有已有关系
        2. 用LLM判断是否匹配
        3. 如果匹配且需要更新，更新；如果不匹配，新建
        """
        relation_content = extracted_relation['content']
        entity1_name = entity1_name or extracted_relation.get('entity1_name') or extracted_relation.get('from_entity_name', '')
        entity2_name = entity2_name or extracted_relation.get('entity2_name') or extracted_relation.get('to_entity_name', '')
        
        # 步骤1：根据两个实体的 entity_id 查找所有已有关系
        # 注意：这里传入的是 entity_id，方法内部会转换为所有版本的绝对ID来查询
        existing_relations = self.storage.get_relations_by_entities(
            entity1_id,
            entity2_id
        )
        
        if not existing_relations:
            # 没有找到已有关系，直接新建
            return self._create_new_relation(
                entity1_id,
                entity2_id,
                relation_content,
                memory_cache_id,
                entity1_name,
                entity2_name,
                verbose_relation,
                doc_name
            )
        
        # 步骤2：准备已有关系信息供LLM判断
        # 按relation_id分组，每个relation_id只保留最新版本
        relation_dict = {}
        for relation in existing_relations:
            if relation.relation_id not in relation_dict:
                relation_dict[relation.relation_id] = relation
            else:
                # 保留物理时间最新的
                if relation.physical_time > relation_dict[relation.relation_id].physical_time:
                    relation_dict[relation.relation_id] = relation
        
        unique_relations = list(relation_dict.values())
        
        existing_relations_info = [
            {
                'relation_id': r.relation_id,
                'content': r.content
            }
            for r in unique_relations
        ]
        
        # 步骤3：用LLM判断是否匹配
        match_result = self.llm_client.judge_relation_match(
            extracted_relation,
            existing_relations_info
        )
        # LLM 有时返回 list 而非 dict，统一取第一个元素
        if isinstance(match_result, list) and len(match_result) > 0:
            match_result = match_result[0] if isinstance(match_result[0], dict) else None
        elif not isinstance(match_result, dict):
            match_result = None

        if match_result and match_result.get('relation_id'):
            # 匹配到已有关系
            relation_id = match_result['relation_id']
            
            # 获取最新版本的content
            latest_relation = unique_relations[0]  # 已经是最新的
            if not latest_relation:
                # 如果找不到最新版本，直接新建
                return self._create_new_relation(
                    entity1_id,
                    entity2_id,
                    relation_content,
                    memory_cache_id,
                    entity1_name,
                    entity2_name,
                    verbose_relation,
                    doc_name
                )
            
            # 判断是否需要更新：比较最新版本的content和当前抽取的content
            need_update = self.llm_client.judge_content_need_update(
                latest_relation.content,
                relation_content
            )
            
            if need_update:
                # 需要更新：合并内容
                # 获取数据库中该relation_id的记录数
                current_versions = self.storage.get_relation_versions(relation_id)
                record_count = len(current_versions)

                # 合并内容
                merged_content = self.llm_client.merge_relation_content(
                    latest_relation.content,
                    relation_content
                )
                
                # 创建新版本
                if verbose_relation:
                    print(f"[关系操作] 🔄 更新关系: {entity1_name} <-> {entity2_name} (relation_id: {relation_id}) - 数据库中该relation_id有 {record_count} 个版本")
                    print(f"  更新前content:")
                    print(f"    {latest_relation.content[:200]}{'...' if len(latest_relation.content) > 200 else ''}")
                    print(f"  新抽取content:")
                    print(f"    {relation_content[:200]}{'...' if len(relation_content) > 200 else ''}")
                    print(f"  合并后content:")
                    print(f"    {merged_content[:200]}{'...' if len(merged_content) > 200 else ''}")
                
                new_relation = self._create_relation_version(
                    relation_id,
                    entity1_id,
                    entity2_id,
                    merged_content,
                    memory_cache_id,
                    verbose_relation,
                    doc_name,
                    entity1_name,
                    entity2_name
                )
                
                if verbose_relation:
                    # 查询更新后的版本数量
                    updated_versions = self.storage.get_relation_versions(relation_id)
                    updated_count = len(updated_versions)
                    print(f"  更新后，数据库中该relation_id有 {updated_count} 个版本")
                
                return new_relation
            else:
                # 不需要更新，返回最新版本
                if verbose_relation:
                    # 获取数据库中该relation_id的版本数量
                    current_versions = self.storage.get_relation_versions(relation_id)
                    version_count = len(current_versions)
                    print(f"[关系操作] ⏭️  匹配但无需更新: {entity1_name} <-> {entity2_name} (relation_id: {relation_id}, 数据库中有 {version_count} 个版本)")
                return latest_relation
        else:
            # 没有匹配到，新建关系
            return self._create_new_relation(
                entity1_id,
                entity2_id,
                relation_content,
                memory_cache_id,
                entity1_name,
                entity2_name,
                verbose_relation
            )
    
    def _create_new_relation(self, entity1_id: str, entity2_id: str,
                            content: str, memory_cache_id: str,
                            entity1_name: str = "", entity2_name: str = "",
                            verbose_relation: bool = True, doc_name: str = "") -> Optional[Relation]:
        """
        创建新关系
        
        注意：
        - 参数 entity1_id 和 entity2_id 是实体的 entity_id（不是绝对ID）
        - 通过 entity_id 获取实体的最新版本，然后使用绝对ID（entity.id）存储到关系中
        - 这确保了关系始终指向实体的最新版本
        
        Returns:
            Relation对象，如果实体不存在则返回None
        """
        # 通过 entity_id 获取实体的最新版本
        entity1 = self.storage.get_entity_by_id(entity1_id)
        entity2 = self.storage.get_entity_by_id(entity2_id)
        
        if not entity1 or not entity2:
            missing_info = []
            if not entity1:
                missing_info.append(f"entity1: {entity1_name or '(未提供名称)'} (entity_id: {entity1_id})")
            if not entity2:
                missing_info.append(f"entity2: {entity2_name or '(未提供名称)'} (entity_id: {entity2_id})")
            
            if verbose_relation:
                print(f"[关系操作] ⚠️  警告: 无法找到实体: {', '.join(missing_info)}，跳过关系创建")
            return None
        
        relation_id = f"rel_{uuid.uuid4().hex[:12]}"
        relation_record_id = f"relation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 标准化实体对（按字母顺序排序，使关系无向化）
        # 通过实体名称来排序，确保 entity1 < entity2
        if entity1.name <= entity2.name:
            entity1_absolute_id = entity1.id
            entity2_absolute_id = entity2.id
        else:
            entity1_absolute_id = entity2.id
            entity2_absolute_id = entity1.id
        
        # 只保存文档名，不包含路径
        doc_name_only = doc_name.split('/')[-1] if doc_name else ""
        
        relation = Relation(
            id=relation_record_id,
            relation_id=relation_id,
            entity1_absolute_id=entity1_absolute_id,  # 使用标准化后的实体对（按名称排序）
            entity2_absolute_id=entity2_absolute_id,  # 使用标准化后的实体对（按名称排序）
            content=content,
            physical_time=datetime.now(),
            memory_cache_id=memory_cache_id,
            doc_name=doc_name_only
        )
        
        self.storage.save_relation(relation)
        
        if verbose_relation:
            # 查询数据库中该relation_id的版本数量（创建后应该有1个版本）
            relation_versions = self.storage.get_relation_versions(relation_id)
            version_count = len(relation_versions)
            print(f"[关系操作] ✅ 创建新关系: {entity1_name} <-> {entity2_name} (relation_id: {relation_id}, 数据库中有 {version_count} 个版本)")
        
        return relation
    
    def _create_relation_version(self, relation_id: str, entity1_id: str,
                                 entity2_id: str, content: str,
                                 memory_cache_id: str,
                                 verbose_relation: bool = True,
                                 doc_name: str = "",
                                 entity1_name: str = "",
                                 entity2_name: str = "") -> Optional[Relation]:
        """
        创建关系的新版本
        
        注意：
        - 参数 entity1_id 和 entity2_id 是实体的 entity_id（不是绝对ID）
        - 通过 entity_id 获取实体的最新版本，然后使用绝对ID（entity.id）存储到关系中
        - 这确保了关系的新版本始终指向实体的最新版本
        
        Returns:
            Relation对象，如果实体不存在则返回None
        """
        # 通过 entity_id 获取实体的最新版本
        entity1 = self.storage.get_entity_by_id(entity1_id)
        entity2 = self.storage.get_entity_by_id(entity2_id)
        
        if not entity1 or not entity2:
            missing_info = []
            if not entity1:
                missing_info.append(f"entity1: {entity1_name or '(未提供名称)'} (entity_id: {entity1_id})")
            if not entity2:
                missing_info.append(f"entity2: {entity2_name or '(未提供名称)'} (entity_id: {entity2_id})")
            
            if verbose_relation:
                print(f"[关系操作] ⚠️  警告: 无法找到实体: {', '.join(missing_info)}，跳过关系版本创建")
            return None
        
        relation_record_id = f"relation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 标准化实体对（按字母顺序排序，使关系无向化）
        # 通过实体名称来排序，确保 entity1 < entity2
        if entity1.name <= entity2.name:
            entity1_absolute_id = entity1.id
            entity2_absolute_id = entity2.id
        else:
            entity1_absolute_id = entity2.id
            entity2_absolute_id = entity1.id
        
        # 只保存文档名，不包含路径
        doc_name_only = doc_name.split('/')[-1] if doc_name else ""
        
        relation = Relation(
            id=relation_record_id,
            relation_id=relation_id,
            entity1_absolute_id=entity1_absolute_id,  # 使用标准化后的实体对（按名称排序）
            entity2_absolute_id=entity2_absolute_id,  # 使用标准化后的实体对（按名称排序）
            content=content,
            physical_time=datetime.now(),
            memory_cache_id=memory_cache_id,
            doc_name=doc_name_only
        )
        
        self.storage.save_relation(relation)
        return relation
