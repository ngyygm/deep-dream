"""
实体处理模块：实体搜索、对齐、更新/新建
"""
from typing import List, Dict, Optional
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
                        memory_cache: Optional[MemoryCache] = None, doc_name: str = "") -> List[Entity]:
        """
        处理抽取的实体：搜索、对齐、更新/新建
        
        Args:
            extracted_entities: 抽取的实体列表（每个包含name和content）
            memory_cache_id: 当前记忆缓存的ID
            similarity_threshold: 相似度阈值（用于搜索）
            memory_cache: 当前记忆缓存对象（可选，用于LLM判断时提供上下文）
            doc_name: 文档名称（只保存文档名，不包含路径）
        
        Returns:
            处理后的实体列表（已保存到数据库）
        """
        processed_entities = []
        
        for extracted_entity in extracted_entities:
            entity = self._process_single_entity(
                extracted_entity, 
                memory_cache_id, 
                similarity_threshold,
                memory_cache,
                doc_name
            )
            if entity:
                processed_entities.append(entity)
        
        return processed_entities
    
    def _process_single_entity(self, extracted_entity: Dict[str, str], 
                               memory_cache_id: str, 
                               similarity_threshold: float,
                               memory_cache: Optional[MemoryCache] = None,
                               doc_name: str = "") -> Optional[Entity]:
        """
        处理单个实体
        
        流程：
        1. 根据词相似度搜索相关实体（使用 name + content[:50] 放宽要求）
        2. 找到同ID下最新的实体（去重）
        3. 用LLM判断是否匹配（结合记忆缓存和实体名称+内容）
        4. 如果匹配，更新；如果不匹配，新建
        """
        entity_name = extracted_entity['name']
        entity_content = extracted_entity['content']
        
        # 步骤1：使用两种模式搜索相关实体并合并结果
        # 模式1：只用name检索（更精确，避免content干扰）
        # 模式2：使用name+content检索（更全面，捕获语义相似）
        half_results = max(1, self.max_similar_entities // 2)  # 对半分，至少1个
        
        # 模式1：只用name检索（使用embedding或文本相似度）
        similar_entities_name = self.storage.search_entities_by_similarity(
            entity_name,
            query_content=None,
            threshold=similarity_threshold,
            max_results=half_results,
            content_snippet_length=self.content_snippet_length,
            text_mode="name_only",
            similarity_method="embedding"  # 优先使用embedding，如果不可用会自动回退
        )
        
        # 模式2：使用name+content检索
        similar_entities_full = self.storage.search_entities_by_similarity(
            entity_name,
            query_content=entity_content,
            threshold=similarity_threshold,
            max_results=half_results,
            content_snippet_length=self.content_snippet_length,
            text_mode="name_and_content",
            similarity_method="embedding"  # 优先使用embedding，如果不可用会自动回退
        )
        
        # 合并结果并去重（按entity_id去重，保留每个entity_id的最新版本）
        entity_dict = {}
        for entity in similar_entities_name + similar_entities_full:
            if entity.entity_id not in entity_dict:
                entity_dict[entity.entity_id] = entity
            else:
                # 保留物理时间最新的
                if entity.physical_time > entity_dict[entity.entity_id].physical_time:
                    entity_dict[entity.entity_id] = entity
        
        similar_entities = list(entity_dict.values())
        
        # 如果合并后超过最大数量，按物理时间排序，保留最新的
        if len(similar_entities) > self.max_similar_entities:
            similar_entities.sort(key=lambda e: e.physical_time, reverse=True)
            similar_entities = similar_entities[:self.max_similar_entities]
        
        if not similar_entities:
            # 没有找到相似实体，直接新建
            return self._create_new_entity(entity_name, entity_content, memory_cache_id, doc_name)
        
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
        
        # 步骤3：准备已有实体信息供LLM判断
        existing_entities_info = [
            {
                'entity_id': e.entity_id,
                'name': e.name,
                'content': e.content
            }
            for e in unique_entities
        ]
        
        # 步骤4：用LLM判断是否匹配（传入记忆缓存以提供上下文）
        match_result = self.llm_client.judge_entity_match(
            extracted_entity, 
            existing_entities_info,
            memory_cache=memory_cache
        )
        
        # 确保 match_result 是字典格式
        if match_result and isinstance(match_result, dict) and match_result.get('entity_id'):
            # 匹配到已有实体
            entity_id = match_result['entity_id']
            
            # 获取最新版本的content
            latest_entity = self.storage.get_entity_by_id(entity_id)
            if not latest_entity:
                # 如果找不到最新版本，直接新建
                return self._create_new_entity(entity_name, entity_content, memory_cache_id, doc_name)
            
            # 判断是否需要更新：比较最新版本的content和当前抽取的content
            need_update = self.llm_client.judge_content_need_update(
                latest_entity.content,
                entity_content
            )
            
            if need_update:
                # 需要更新：合并名称和内容
                # 获取数据库中该entity_id的记录数
                current_versions = self.storage.get_entity_versions(entity_id)
                record_count = len(current_versions)

                # 合并名称（如果名称不同）
                if entity_name != latest_entity.name:
                    merged_name = self.llm_client.merge_entity_name(
                        latest_entity.name,
                        entity_name
                    )
                else:
                    merged_name = entity_name
                
                # 合并内容
                merged_content = self.llm_client.merge_entity_content(
                    latest_entity.content,
                    entity_content
                )

                # 创建新版本
                print(f"[实体操作] 🔄 更新实体: {entity_name} (entity_id: {entity_id}) - 数据库中该entity_id有 {record_count} 个版本")
                if entity_name != latest_entity.name:
                    print(f"  名称合并: {latest_entity.name} + {entity_name} -> {merged_name}")
                print(f"  更新前content:")
                print(f"    {latest_entity.content[:200]}{'...' if len(latest_entity.content) > 200 else ''}")
                print(f"  新抽取content:")
                print(f"    {entity_content[:200]}{'...' if len(entity_content) > 200 else ''}")
                print(f"  合并后content:")
                print(f"    {merged_content[:200]}{'...' if len(merged_content) > 200 else ''}")
                
                new_entity = self._create_entity_version(
                    entity_id,
                    merged_name,  # 使用合并后的名称
                    merged_content,
                    memory_cache_id,
                    doc_name
                )
                
                # 查询更新后的版本数量
                updated_versions = self.storage.get_entity_versions(entity_id)
                updated_count = len(updated_versions)
                print(f"  更新后，数据库中该entity_id有 {updated_count} 个版本")
                
                return new_entity
            else:
                # 不需要更新，返回最新版本
                # 获取数据库中该entity_id的版本数量
                current_versions = self.storage.get_entity_versions(entity_id)
                version_count = len(current_versions)
                print(f"[实体操作] ⏭️  匹配但无需更新: {entity_name} (entity_id: {entity_id}, 数据库中有 {version_count} 个版本, 匹配实体名称: {latest_entity.name})")
                return latest_entity
        else:
            # 没有匹配到，新建实体
            return self._create_new_entity(entity_name, entity_content, memory_cache_id, doc_name)
    
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
        
        # 查询数据库中该entity_id的版本数量（创建后应该有1个版本）
        entity_versions = self.storage.get_entity_versions(entity_id)
        version_count = len(entity_versions)
        
        print(f"[实体操作] ✅ 创建新实体: {name} (entity_id: {entity_id}, 数据库中有 {version_count} 个版本)")
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
