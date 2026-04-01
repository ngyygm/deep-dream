"""
实体处理模块：实体搜索、对齐、更新/新建
"""
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import uuid
import numpy as np

from ..models import Entity, MemoryCache, ContentPatch
from ..storage.manager import StorageManager
from ..llm.client import LLMClient
from ..utils import clean_markdown_code_blocks, clean_separator_tags, wprint
from ..content_schema import (
    ENTITY_SECTIONS,
    content_to_sections,
    compute_section_diff,
    has_any_change,
    collect_changed_sections,
    render_markdown_sections,
    section_hash,
)


class EntityProcessor:
    """实体处理器 - 负责实体的搜索、对齐、更新和新建"""
    
    def __init__(self, storage: StorageManager, llm_client: LLMClient,
                 max_similar_entities: int = 10, content_snippet_length: int = 50,
                 max_alignment_candidates: Optional[int] = None,
                 verbose: bool = True,
                 entity_progress_verbose: bool = False):
        self.storage = storage
        self.llm_client = llm_client
        self.max_similar_entities = max_similar_entities
        self.content_snippet_length = content_snippet_length
        self.max_alignment_candidates = max_alignment_candidates  # None = 不限制
        self.batch_resolution_enabled = True
        self.batch_resolution_confidence_threshold = 0.55
        self.verbose = verbose
        # 逐实体树状进度（处理实体 x/y、批量候选等）；默认关闭以免服务/API 控制台刷屏
        self.entity_progress_verbose = entity_progress_verbose

    def _entity_tree_log(self) -> bool:
        return self.verbose and self.entity_progress_verbose

    def encode_entities_for_candidate_table(
        self, extracted_entities: List[Dict[str, str]]
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """为本窗实体批量编码 name / name+snippet，供 _build_entity_candidate_table 使用（可异步预取）。"""
        if not extracted_entities:
            return None, None
        if not self.storage.embedding_client or not self.storage.embedding_client.is_available():
            return None, None
        snip = self.llm_client.effective_entity_snippet_length()
        name_embeddings = self.storage.embedding_client.encode([e["name"] for e in extracted_entities])
        full_embeddings = self.storage.embedding_client.encode([
            f"{e['name']} {e['content'][:snip]}" for e in extracted_entities
        ])
        return name_embeddings, full_embeddings

    def process_entities(self, extracted_entities: List[Dict[str, str]],
                        memory_cache_id: str, similarity_threshold: float = 0.7,
                        memory_cache: Optional[MemoryCache] = None, source_document: str = "",
                        context_text: Optional[str] = None,
                        extracted_relations: Optional[List[Dict[str, str]]] = None,
                        jaccard_search_threshold: Optional[float] = None,
                        embedding_name_search_threshold: Optional[float] = None,
                        embedding_full_search_threshold: Optional[float] = None,
                        on_entity_processed: Optional[callable] = None,
                        base_time: Optional[datetime] = None,
                        max_workers: Optional[int] = None,
                        verbose: Optional[bool] = None,
                        entity_embedding_prefetch: Optional[Future] = None) -> Tuple[List[Entity], List[Dict], Dict[str, str]]:
        """
        处理抽取的实体：搜索、对齐、更新/新建。
        当 max_workers > 1 且实体数 > 1 时使用多线程并行；合并冲突时以数据库中已存在的 entity_id 为准。
        
        Args:
            extracted_entities: 抽取的实体列表（每个包含name和content）
            memory_cache_id: 当前记忆缓存的ID
            similarity_threshold: 相似度阈值（用于搜索，作为默认值）
            memory_cache: 当前记忆缓存对象（可选，用于LLM判断时提供上下文）
            source_document: 文档名称（只保存文档名，不包含路径）
            context_text: 可选的上下文文本（当前处理的文本片段），用于精细化判断时提供场景信息
            extracted_relations: 步骤3抽取的关系列表（用于判断是否已存在关系）
            jaccard_search_threshold: Jaccard搜索（name_only）的相似度阈值（可选，默认使用similarity_threshold）
            embedding_name_search_threshold: Embedding搜索（name_only）的相似度阈值（可选，默认使用similarity_threshold）
            embedding_full_search_threshold: Embedding搜索（name+content）的相似度阈值（可选，默认使用similarity_threshold）
            on_entity_processed: 每个实体处理完的回调（可选）
            base_time: 基准时间（可选）
            max_workers: 并行线程数；>1 且实体数>1 时启用多线程，合并冲突时以数据库已有 id 为准
            entity_embedding_prefetch: 可选 Future，结果为 encode_entities_for_candidate_table 的返回值；失败时回退为现场 encode
        
        Returns:
            Tuple[处理后的实体列表, 待处理的关系列表, 实体名称到ID的映射]
            关系信息格式：{"entity1_name": "...", "entity2_name": "...", "content": "...", "relation_type": "alias|normal"}
            注意：关系中的实体使用名称而不是ID，因为新实体在创建前还没有ID
        """
        # 临时覆盖 verbose
        _orig_verbose = self.verbose
        if verbose is not None:
            self.verbose = verbose

        try:
            prefetched_embeddings: Optional[Tuple[Optional[Any], Optional[Any]]] = None
            if entity_embedding_prefetch is not None:
                try:
                    prefetched_embeddings = entity_embedding_prefetch.result()
                except Exception:
                    prefetched_embeddings = None
            use_parallel = (max_workers is not None and max_workers > 1 and len(extracted_entities) > 1)
            if use_parallel:
                result = self._process_entities_parallel(
                    extracted_entities=extracted_entities,
                    memory_cache_id=memory_cache_id,
                    similarity_threshold=similarity_threshold,
                    memory_cache=memory_cache,
                    source_document=source_document,
                    context_text=context_text,
                    extracted_relations=extracted_relations,
                    jaccard_search_threshold=jaccard_search_threshold,
                    embedding_name_search_threshold=embedding_name_search_threshold,
                    embedding_full_search_threshold=embedding_full_search_threshold,
                    on_entity_processed=on_entity_processed,
                    base_time=base_time,
                    max_workers=max_workers,
                    prefetched_embeddings=prefetched_embeddings,
                )
            else:
                result = self._process_entities_sequential(
                    extracted_entities=extracted_entities,
                    memory_cache_id=memory_cache_id,
                    similarity_threshold=similarity_threshold,
                    memory_cache=memory_cache,
                    source_document=source_document,
                    context_text=context_text,
                    extracted_relations=extracted_relations,
                    jaccard_search_threshold=jaccard_search_threshold,
                    embedding_name_search_threshold=embedding_name_search_threshold,
                    embedding_full_search_threshold=embedding_full_search_threshold,
                    on_entity_processed=on_entity_processed,
                    base_time=base_time,
                    prefetched_embeddings=prefetched_embeddings,
                )
            return result
        finally:
            self.verbose = _orig_verbose

    def _process_entities_sequential(self, extracted_entities: List[Dict[str, str]],
                        memory_cache_id: str, similarity_threshold: float = 0.7,
                        memory_cache: Optional[MemoryCache] = None, source_document: str = "",
                        context_text: Optional[str] = None,
                        extracted_relations: Optional[List[Dict[str, str]]] = None,
                        jaccard_search_threshold: Optional[float] = None,
                        embedding_name_search_threshold: Optional[float] = None,
                        embedding_full_search_threshold: Optional[float] = None,
                        on_entity_processed: Optional[callable] = None,
                        base_time: Optional[datetime] = None,
                        prefetched_embeddings: Optional[Tuple[Optional[Any], Optional[Any]]] = None) -> Tuple[List[Entity], List[Dict], Dict[str, str]]:
        """串行处理实体（原逻辑）。"""
        processed_entities: List[Entity] = []
        pending_relations: List[Dict] = []
        entity_name_to_id: Dict[str, str] = {}
        entities_to_persist: List[Entity] = []

        extracted_entity_names = {e['name'] for e in extracted_entities}
        extracted_relation_pairs = set()
        # 收集所有有关系关联的实体名称（用于过滤孤立实体）
        related_entity_names = set()
        if extracted_relations:
            for rel in extracted_relations:
                entity1_name = rel.get('entity1_name') or rel.get('from_entity_name', '').strip()
                entity2_name = rel.get('entity2_name') or rel.get('to_entity_name', '').strip()
                content = rel.get('content', '').strip()
                if entity1_name and entity2_name:
                    pair_key = tuple(sorted([entity1_name, entity2_name]))
                    extracted_relation_pairs.add((pair_key, hash(content.lower())))
                    related_entity_names.add(entity1_name)
                    related_entity_names.add(entity2_name)

        candidate_table = self._build_entity_candidate_table(
            extracted_entities,
            similarity_threshold=similarity_threshold,
            jaccard_search_threshold=jaccard_search_threshold,
            embedding_name_search_threshold=embedding_name_search_threshold,
            embedding_full_search_threshold=embedding_full_search_threshold,
            prefetched_embeddings=prefetched_embeddings,
        )

        total_entities = len(extracted_entities)
        _skipped_orphans = 0
        for idx, extracted_entity in enumerate(extracted_entities, 1):
            # 跳过没有关系关联的孤立实体
            if related_entity_names and extracted_entity['name'] not in related_entity_names:
                _skipped_orphans += 1
                continue
            candidates = candidate_table.get(idx - 1, [])
            entity, relations, name_mapping, to_persist = self._process_single_entity_batch(
                extracted_entity=extracted_entity,
                candidates=candidates,
                memory_cache_id=memory_cache_id,
                similarity_threshold=similarity_threshold,
                memory_cache=memory_cache,
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
            )

            if entity:
                processed_entities.append(entity)
                entity_name_to_id[entity.name] = entity.entity_id
                entity_name_to_id[extracted_entity['name']] = entity.entity_id
            if relations:
                pending_relations.extend(relations)
            if name_mapping:
                entity_name_to_id.update(name_mapping)
            if to_persist:
                entities_to_persist.append(to_persist)
            if on_entity_processed and entity:
                on_entity_processed(entity, entity_name_to_id, relations or [])

        if entities_to_persist:
            self.storage.bulk_save_entities(entities_to_persist)

        if _skipped_orphans > 0:
            wprint(f"  ⚠️ 跳过 {_skipped_orphans} 个孤立实体（无关系边关联）")

        return processed_entities, pending_relations, entity_name_to_id

    def _process_entities_parallel(self, extracted_entities: List[Dict[str, str]],
                        memory_cache_id: str, similarity_threshold: float = 0.7,
                        memory_cache: Optional[MemoryCache] = None, source_document: str = "",
                        context_text: Optional[str] = None,
                        extracted_relations: Optional[List[Dict[str, str]]] = None,
                        jaccard_search_threshold: Optional[float] = None,
                        embedding_name_search_threshold: Optional[float] = None,
                        embedding_full_search_threshold: Optional[float] = None,
                        on_entity_processed: Optional[callable] = None,
                        base_time: Optional[datetime] = None,
                        max_workers: int = 2,
                        prefetched_embeddings: Optional[Tuple[Optional[Any], Optional[Any]]] = None) -> Tuple[List[Entity], List[Dict], Dict[str, str]]:
        """多线程处理实体；合并冲突时以数据库中已存在的 entity_id 为准。"""
        extracted_entity_names = {e['name'] for e in extracted_entities}
        extracted_relation_pairs = set()
        # 收集所有有关系关联的实体名称（用于过滤孤立实体）
        related_entity_names = set()
        if extracted_relations:
            for rel in extracted_relations:
                entity1_name = rel.get('entity1_name') or rel.get('from_entity_name', '').strip()
                entity2_name = rel.get('entity2_name') or rel.get('to_entity_name', '').strip()
                content = rel.get('content', '').strip()
                if entity1_name and entity2_name:
                    pair_key = tuple(sorted([entity1_name, entity2_name]))
                    extracted_relation_pairs.add((pair_key, hash(content.lower())))
                    related_entity_names.add(entity1_name)
                    related_entity_names.add(entity2_name)

        # 过滤掉没有关系关联的孤立实体
        if related_entity_names:
            _filtered = [
                (idx, e) for idx, e in enumerate(extracted_entities)
                if e['name'] in related_entity_names
            ]
            _skipped_orphans = len(extracted_entities) - len(_filtered)
            if _skipped_orphans > 0:
                wprint(f"  ⚠️ 跳过 {_skipped_orphans} 个孤立实体（无关系边关联）")
            # 重建索引映射：原索引 → 过滤后索引
            _orig_indices = [orig_idx for orig_idx, _ in _filtered]
            filtered_entities = [e for _, e in _filtered]
        else:
            _skipped_orphans = 0
            _orig_indices = list(range(len(extracted_entities)))
            filtered_entities = extracted_entities

        candidate_table = self._build_entity_candidate_table(
            extracted_entities,
            similarity_threshold=similarity_threshold,
            jaccard_search_threshold=jaccard_search_threshold,
            embedding_name_search_threshold=embedding_name_search_threshold,
            embedding_full_search_threshold=embedding_full_search_threshold,
            prefetched_embeddings=prefetched_embeddings,
        )
        total_entities = len(extracted_entities)
        _distill_step = self.llm_client._current_distill_step
        _priority = getattr(self.llm_client._priority_local, 'priority', 5)

        def task(idx: int, extracted_entity: Dict[str, str], orig_idx: int):
            # 将主线程的 distill step 和优先级传播到工作线程（threading.local）
            self.llm_client._current_distill_step = _distill_step
            self.llm_client._priority_local.priority = _priority
            candidates = candidate_table.get(orig_idx, [])
            entity, relations, name_mapping, to_persist = self._process_single_entity_batch(
                extracted_entity=extracted_entity,
                candidates=candidates,
                memory_cache_id=memory_cache_id,
                similarity_threshold=similarity_threshold,
                memory_cache=memory_cache,
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
            )
            return (idx, entity, relations, name_mapping, to_persist)

        results: List[Tuple[int, Optional[Entity], List[Dict], Dict[str, str], Optional[Entity]]] = []
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tmg-llm") as executor:
            futures = {
                executor.submit(task, idx, extracted_entity, orig_idx): idx
                for idx, (extracted_entity, orig_idx) in enumerate(
                    zip(filtered_entities, _orig_indices), 1
                )
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda r: r[0])

        name_to_ids: Dict[str, set] = {}
        for idx, entity, relations, name_mapping, to_persist in results:
            if name_mapping:
                for name, eid in name_mapping.items():
                    if name and eid:
                        name_to_ids.setdefault(name, set()).add(eid)

        entity_name_to_id: Dict[str, str] = {}
        for name, ids in name_to_ids.items():
            # 优先使用数据库中已存在的 entity_id（同名实体被多个线程分别匹配到不同候选）
            in_storage = [eid for eid in ids if self.storage.get_entity_by_entity_id(eid) is not None]
            if in_storage:
                entity_name_to_id[name] = in_storage[0]
            else:
                entity_name_to_id[name] = sorted(ids)[0]

        redirect_pairs = []
        for name, ids in name_to_ids.items():
            canonical_id = entity_name_to_id.get(name)
            if not canonical_id:
                continue
            for eid in ids:
                if eid and eid != canonical_id:
                    redirect_pairs.append((eid, canonical_id))
        for source_id, canonical_id in redirect_pairs:
            self.storage.register_entity_redirect(source_id, canonical_id)

        # 对于被合并到 canonical ID 的非 canonical 实体，需要从 results 中修正
        for i, (idx, entity, relations, name_mapping, to_persist) in enumerate(results):
            if entity and entity.entity_id != entity_name_to_id.get(entity.name):
                canonical_id = entity_name_to_id.get(entity.name)
                if canonical_id:
                    canonical_entity = self.storage.get_entity_by_entity_id(canonical_id)
                    if canonical_entity:
                        results[i] = (idx, canonical_entity, relations, name_mapping, to_persist)

        canonical_ids = set(entity_name_to_id.values())
        all_to_persist: List[Entity] = [r[4] for r in results if r[4] is not None]
        entities_to_persist_final = [e for e in all_to_persist if e.entity_id in canonical_ids]
        if entities_to_persist_final:
            self.storage.bulk_save_entities(entities_to_persist_final)

        processed_entities = [r[1] for r in results if r[1] is not None]
        pending_relations: List[Dict] = []
        for r in results:
            if r[2]:
                pending_relations.extend(r[2])
        if on_entity_processed:
            for r in results:
                if r[1]:
                    on_entity_processed(r[1], entity_name_to_id, r[2] or [])

        return processed_entities, pending_relations, entity_name_to_id
    
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        set1 = set((text1 or "").lower())
        set2 = set((text2 or "").lower())
        union = len(set1 | set2)
        if union == 0:
            return 0.0
        return len(set1 & set2) / union

    def _cosine_similarity(self, embedding1: Optional[np.ndarray], embedding2: Optional[np.ndarray]) -> float:
        if embedding1 is None or embedding2 is None:
            return 0.0
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2 + 1e-9))

    def _build_entity_candidate_table(self,
                                      extracted_entities: List[Dict[str, str]],
                                      similarity_threshold: float,
                                      jaccard_search_threshold: Optional[float] = None,
                                      embedding_name_search_threshold: Optional[float] = None,
                                      embedding_full_search_threshold: Optional[float] = None,
                                      prefetched_embeddings: Optional[Tuple[Optional[Any], Optional[Any]]] = None,
                                      ) -> Dict[int, List[Dict[str, Any]]]:
        """窗口级批量候选生成，替代逐实体三次全库搜索。

        prefetched_embeddings: 非 None 时表示已预取 (name_emb, full_emb)，不再现场 encode；
        为 None 时在本函数内按需 encode（与无预取行为一致）。
        """
        projections = self.storage.get_latest_entities_projection(self.llm_client.effective_entity_snippet_length())
        if not projections:
            return {}

        jaccard_threshold = jaccard_search_threshold if jaccard_search_threshold is not None else min(similarity_threshold, 0.6)
        embedding_name_threshold = embedding_name_search_threshold if embedding_name_search_threshold is not None else min(similarity_threshold, 0.6)
        embedding_full_threshold = embedding_full_search_threshold if embedding_full_search_threshold is not None else min(similarity_threshold, 0.6)

        name_embeddings: Optional[Any] = None
        full_embeddings: Optional[Any] = None
        if prefetched_embeddings is not None:
            name_embeddings, full_embeddings = prefetched_embeddings
        elif self.storage.embedding_client and self.storage.embedding_client.is_available():
            name_embeddings = self.storage.embedding_client.encode([entity["name"] for entity in extracted_entities])
            full_embeddings = self.storage.embedding_client.encode([
                f"{entity['name']} {entity['content'][:self.llm_client.effective_entity_snippet_length()]}"
                for entity in extracted_entities
            ])

        candidate_table: Dict[int, List[Dict[str, Any]]] = {}
        for idx, extracted_entity in enumerate(extracted_entities):
            candidate_rows: List[Dict[str, Any]] = []
            for projection in projections:
                lexical_score = self._calculate_jaccard_similarity(extracted_entity["name"], projection["name"])
                dense_name_score = 0.0
                dense_full_score = 0.0
                if name_embeddings is not None:
                    dense_name_score = self._cosine_similarity(
                        np.array(name_embeddings[idx], dtype=np.float32),
                        projection.get("embedding_array"),
                    )
                if full_embeddings is not None:
                    dense_full_score = self._cosine_similarity(
                        np.array(full_embeddings[idx], dtype=np.float32),
                        projection.get("embedding_array"),
                    )

                if (
                    lexical_score >= jaccard_threshold
                    or dense_name_score >= embedding_name_threshold
                    or dense_full_score >= embedding_full_threshold
                ):
                    candidate_rows.append({
                        "entity_id": projection["entity_id"],
                        "name": projection["name"],
                        "content": projection["content"],
                        "source_document": projection["entity"].source_document if projection.get("entity") else "",
                        "version_count": projection["version_count"],
                        "lexical_score": lexical_score,
                        "dense_score": max(dense_name_score, dense_full_score),
                        "combined_score": max(lexical_score, dense_name_score, dense_full_score),
                    })

            candidate_rows.sort(key=lambda row: row["combined_score"], reverse=True)
            limit = self.max_alignment_candidates or self.max_similar_entities
            candidate_table[idx] = candidate_rows[:limit]
        return candidate_table

    def _process_single_entity_batch(self,
                                     extracted_entity: Dict[str, str],
                                     candidates: List[Dict[str, Any]],
                                     memory_cache_id: str,
                                     similarity_threshold: float,
                                     memory_cache: Optional[MemoryCache] = None,
                                     source_document: str = "",
                                     context_text: Optional[str] = None,
                                     entity_index: int = 0,
                                     total_entities: int = 0,
                                     extracted_entity_names: Optional[set] = None,
                                     extracted_relation_pairs: Optional[set] = None,
                                     jaccard_search_threshold: Optional[float] = None,
                                     embedding_name_search_threshold: Optional[float] = None,
                                     embedding_full_search_threshold: Optional[float] = None,
                                     base_time: Optional[datetime] = None) -> Tuple[Optional[Entity], List[Dict], Dict[str, str], Optional[Entity]]:
        """批量候选 + 批量裁决主路径，低置信度时回退旧逻辑。"""
        entity_name = extracted_entity["name"]
        entity_content = extracted_entity["content"]
        if self._entity_tree_log() and total_entities > 0:
            wprint(f"  ├─ 处理实体 [{entity_index}/{total_entities}]: {entity_name}")

        if not candidates:
            new_entity = self._build_new_entity(entity_name, entity_content, memory_cache_id, source_document, base_time=base_time)
            if self._entity_tree_log():
                wprint(f"  │  未找到候选实体，批量路径创建新实体: {new_entity.entity_id}")
            return new_entity, [], {entity_name: new_entity.entity_id, new_entity.name: new_entity.entity_id}, new_entity

        if self._entity_tree_log():
            wprint(f"  │  批量候选生成: {len(candidates)} 个")
        batch_result = self.llm_client.resolve_entity_candidates_batch(
            {
                "entity_id": "NEW_ENTITY",
                "name": entity_name,
                "content": entity_content,
                "source_document": source_document.split('/')[-1] if source_document else "",
                "version_count": 0,
            },
            candidates,
            context_text=context_text,
        )
        confidence = float(batch_result.get("confidence", 0.0) or 0.0)
        if (not self.batch_resolution_enabled) or batch_result.get("update_mode") == "fallback" or confidence < self.batch_resolution_confidence_threshold:
            if self._entity_tree_log():
                wprint(f"  │  批量裁决置信度不足，回退到旧逻辑 (confidence={confidence:.2f})")
            entity, relations, name_mapping = self._process_single_entity(
                extracted_entity,
                memory_cache_id,
                similarity_threshold,
                memory_cache,
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
            )
            return entity, relations, name_mapping, None

        relations_to_create: List[Dict] = []
        for relation in batch_result.get("relations_to_create", []) or []:
            candidate = next((item for item in candidates if item.get("entity_id") == relation.get("entity_id")), None)
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
        if match_existing_id:
            latest_entity = self.storage.get_entity_by_entity_id(match_existing_id)
            if not latest_entity:
                if self._entity_tree_log():
                    wprint(f"  │  批量裁决命中的实体不存在，回退旧逻辑: {match_existing_id}")
                entity, relations, name_mapping = self._process_single_entity(
                    extracted_entity,
                    memory_cache_id,
                    similarity_threshold,
                    memory_cache,
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
                )
                return entity, relations, name_mapping, None

            update_mode = batch_result.get("update_mode") or "reuse_existing"
            if update_mode == "merge_into_latest":
                merged_name = (batch_result.get("merged_name") or latest_entity.name).strip()
                merged_content = (batch_result.get("merged_content") or "").strip()
                if not merged_content:
                    merged_content = self.llm_client.merge_multiple_entity_contents([latest_entity.content, entity_content])
                entity_version = self._build_entity_version(
                    latest_entity.entity_id,
                    merged_name,
                    merged_content,
                    memory_cache_id,
                    source_document,
                    base_time=base_time,
                )
                if self._entity_tree_log():
                    wprint(f"  │  批量裁决: 合并到已有实体 {latest_entity.entity_id} 并生成新版本")
                return entity_version, relations_to_create, {
                    entity_name: latest_entity.entity_id,
                    entity_version.name: latest_entity.entity_id,
                }, entity_version

            if self._entity_tree_log():
                wprint(f"  │  批量裁决: 复用已有实体 {latest_entity.entity_id}")
            return latest_entity, relations_to_create, {
                entity_name: latest_entity.entity_id,
                latest_entity.name: latest_entity.entity_id,
            }, None

        merged_name = (batch_result.get("merged_name") or entity_name).strip() or entity_name
        merged_content = (batch_result.get("merged_content") or entity_content).strip() or entity_content
        new_entity = self._build_new_entity(merged_name, merged_content, memory_cache_id, source_document, base_time=base_time)
        if self._entity_tree_log():
            wprint(f"  │  批量裁决: 创建新实体 {new_entity.entity_id}")
        return new_entity, relations_to_create, {
            entity_name: new_entity.entity_id,
            new_entity.name: new_entity.entity_id,
        }, new_entity

    def _process_single_entity(self, extracted_entity: Dict[str, str], 
                               memory_cache_id: str, 
                               similarity_threshold: float,
                               memory_cache: Optional[MemoryCache] = None,
                               source_document: str = "",
                               context_text: Optional[str] = None,
                               entity_index: int = 0,
                               total_entities: int = 0,
                               extracted_entity_names: Optional[set] = None,
                               extracted_relation_pairs: Optional[set] = None,
                               jaccard_search_threshold: Optional[float] = None,
                               embedding_name_search_threshold: Optional[float] = None,
                               embedding_full_search_threshold: Optional[float] = None,
                               base_time: Optional[datetime] = None) -> Tuple[Optional[Entity], List[Dict], Dict[str, str]]:
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
        if self._entity_tree_log():
            if total_entities > 0:
                wprint(f"  ├─ 处理实体 [{entity_index}/{total_entities}]: {entity_name}")
            else:
                wprint(f"  ├─ 处理实体: {entity_name}")
        
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
            content_snippet_length=self.llm_client.effective_entity_snippet_length(),
            text_mode="name_only",
            similarity_method="jaccard"
        )
        
        # 模式1：只用name检索（使用embedding）
        candidates_name_embedding = self.storage.search_entities_by_similarity(
            entity_name,
            query_content=None,
            threshold=embedding_name_threshold,
            max_results=self.max_similar_entities,
            content_snippet_length=self.llm_client.effective_entity_snippet_length(),
            text_mode="name_only",
            similarity_method="embedding"
        )
        
        # 模式2：使用name+content检索（使用embedding）
        candidates_full_embedding = self.storage.search_entities_by_similarity(
            entity_name,
            query_content=entity_content,
            threshold=embedding_full_threshold,
            max_results=self.max_similar_entities,
            content_snippet_length=self.llm_client.effective_entity_snippet_length(),
            text_mode="name_and_content",
            similarity_method="embedding"
        )
        
        # 输出三种搜索方法的结果
        if self._entity_tree_log():
            wprint(f"  │  ├─ Jaccard搜索（name_only）: {len(candidates_jaccard)} 个")
            wprint(f"  │  ├─ Embedding搜索（name_only）: {len(candidates_name_embedding)} 个")
            wprint(f"  │  ├─ Embedding搜索（name+content）: {len(candidates_full_embedding)} 个")
        
        # 合并结果并去重（按entity_id去重，保留每个entity_id的最新版本）
        entity_dict = {}
        for entity in candidates_jaccard + candidates_name_embedding + candidates_full_embedding:
            if entity.entity_id not in entity_dict:
                entity_dict[entity.entity_id] = entity
            else:
                # 保留物理时间最新的
                if entity.processed_time > entity_dict[entity.entity_id].processed_time:
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
                        if self._entity_tree_log():
                            wprint(f"  │  │  ├─ {candidate.name}: 跳过已有关系（步骤3已处理）")
                    else:
                        # 虽然都在当前列表中，但还没有关系，需要判断
                        filtered_similar_entities.append(candidate)
            
            similar_entities = filtered_similar_entities
            if self._entity_tree_log() and skipped_count > 0:
                wprint(f"  │  跳过 {skipped_count} 个已在当前抽取列表且已存在关系的候选实体（步骤3已处理）")
        
        # # 如果合并后超过最大数量，按物理时间排序，保留最新的
        # if len(similar_entities) > self.max_similar_entities:
        #     similar_entities.sort(key=lambda e: e.processed_time, reverse=True)
        #     similar_entities = similar_entities[:self.max_similar_entities]
        
        if not similar_entities:
            # 没有找到相似实体，直接新建
            new_entity = self._create_new_entity(entity_name, entity_content, memory_cache_id, source_document, base_time=base_time)
            if self._entity_tree_log():
                wprint(f"  │  未找到相似实体，创建新实体: {new_entity.entity_id}")
            # 返回实体、空关系列表、实体名称到ID的映射
            entity_name_to_id = {
                entity_name: new_entity.entity_id,
                new_entity.name: new_entity.entity_id
            }
            return new_entity, [], entity_name_to_id
        
        if self._entity_tree_log():
            wprint(f"  │  找到 {len(similar_entities)} 个候选实体")
        
        # 步骤2：找到同ID下最新的实体（去重）
        # 按entity_id分组，每个entity_id只保留最新版本
        entity_dict = {}
        for entity in similar_entities:
            if entity.entity_id not in entity_dict:
                entity_dict[entity.entity_id] = entity
            else:
                # 保留物理时间最新的
                if entity.processed_time > entity_dict[entity.entity_id].processed_time:
                    entity_dict[entity.entity_id] = entity
        
        unique_entities = list(entity_dict.values())
        
        # 步骤3：准备已有实体信息供LLM分析
        # 构建实体组：当前抽取的实体（作为第一个，即"当前分析的实体"）+ 候选实体
        entities_group = [
            {
                'entity_id': 'NEW_ENTITY',  # 标记为新实体
                'name': entity_name,
                'content': entity_content,
                'source_document': source_document.split('/')[-1] if source_document else "",
                'version_count': 0
            }
        ]
        
        # 添加候选实体信息（批量获取版本数，避免 N+1 查询）
        entity_ids = [e.entity_id for e in unique_entities]
        version_counts = self.storage.get_entity_version_counts(entity_ids)
        for e in unique_entities:
            entities_group.append({
                'entity_id': e.entity_id,
                'name': e.name,
                'content': e.content,
                'source_document': e.source_document,
                'version_count': version_counts.get(e.entity_id, 1)
            })

        # 步骤4：检查实体对之间是否已有关系（批量查询，避免 N+1）
        existing_relations_between = {}
        existing_entity_ids = [e.entity_id for e in unique_entities]
        entity_pairs = [
            (existing_entity_ids[i], existing_entity_ids[j])
            for i in range(len(existing_entity_ids))
            for j in range(i + 1, len(existing_entity_ids))
        ]
        if entity_pairs:
            batch_relations = self.storage.get_relations_by_entity_pairs(entity_pairs)
            for pair_key_tuple, relations in batch_relations.items():
                if relations:
                    pair_key = "|".join(pair_key_tuple)
                    existing_relations_between[pair_key] = [
                        {'relation_id': r.relation_id, 'content': r.content}
                        for r in relations
                    ]
        
        # 步骤5：使用两步流程分析（初步筛选 + 精细化判断）
        if self._entity_tree_log():
            wprint(f"  │  调用LLM分析（候选数: {len(unique_entities)}）")
        
        # 阶段1：初步筛选（使用content snippet快速筛选）
        preliminary_result = self.llm_client.analyze_entity_candidates_preliminary(
            entities_group,
            content_snippet_length=self.llm_client.effective_entity_snippet_length(),
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
        if self._entity_tree_log():
            relation_count = len([c for c in candidates_to_analyze.values() if c['type'] == 'relation'])
            if skipped_relations_count > 0:
                wprint(f"  │  ├─ 初步筛选: 合并 {len([c for c in candidates_to_analyze.values() if c['type'] == 'merge'])} 个, 关系 {relation_count} 个 (跳过已有关系: {skipped_relations_count} 个), 跳过 {len(no_action)} 个")
            else:
                wprint(f"  │  ├─ 初步筛选: 合并 {len([c for c in candidates_to_analyze.values() if c['type'] == 'merge'])} 个, 关系 {relation_count} 个, 跳过 {len(no_action)} 个")
        
        # 准备当前实体信息（新实体）
        current_entity_info = {
            "entity_id": "NEW_ENTITY",
            "name": entity_name,
            "content": entity_content,
            "source_document": source_document.split('/')[-1] if source_document else "",
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
            if self._entity_tree_log():
                wprint(f"  │  ├─ 精细化判断开始（共 {len(candidates_to_analyze)} 个候选{skipped_info}）")
        
        for cid, info in candidates_to_analyze.items():
            candidate_entity = next((e for e in unique_entities if e.entity_id == cid), None)
            if not candidate_entity:
                continue
            
            candidate_info = {
                "entity_id": cid,
                "name": candidate_entity.name,
                "content": candidate_entity.content,
                "source_document": candidate_entity.source_document,
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
                pass

        # 输出最终分析结果
        if merge_decisions or relation_decisions:
            if self._entity_tree_log():
                wprint(f"  │  └─ 精细化判断: 合并 {len(merge_decisions)} 个, 关系 {len(relation_decisions)} 个")
        
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
                        if self._entity_tree_log():
                            wprint(f"  │  ├─ 多合并目标: 选择 {primary_target_id} 为主要目标（版本数最多）")
                        
                        # 在合并之前，先收集其他目标实体的信息（合并后这些ID就不存在了）
                        other_targets_entities.clear()  # 清空之前的数据
                        for other_target_id in other_targets:
                            other_entity = self.storage.get_entity_by_entity_id(other_target_id)
                            if other_entity:
                                other_targets_entities[other_target_id] = {
                                    'entity': other_entity,
                                    'name': other_entity.name,
                                    'content': other_entity.content
                                }
                        
                        # 如果有多个不同的目标实体ID，说明这些实体都是同一个实体
                        # 需要将其他目标实体ID合并到主要目标ID
                        merge_result = self.storage.merge_entity_ids(primary_target_id, other_targets)
                        
                        # 更新映射：将所有指向旧实体ID的映射更新为新的 primary_target_id
                        # 这确保映射中不会保留指向已合并ID的失效映射
                        updated_mapping_count = 0
                        for name, eid in list(entity_name_to_id.items()):
                            if eid in other_targets:
                                entity_name_to_id[name] = primary_target_id
                                updated_mapping_count += 1
                        # 处理合并后产生的自指向关系（暂时跳过，因为entity_processor中没有这个方法）
                        # 自指向关系会在后续的consolidate_knowledge_graph_entity中处理
                
                # 合并新实体到主要目标实体
                latest_entity = self.storage.get_entity_by_entity_id(primary_target_id)
                if latest_entity:
                    target_name = latest_entity.name
                    
                    # 收集所有需要合并到主要目标的实体的content
                    # 包括：主要目标实体 + 新实体 + 所有指向主要目标的候选实体 + 被合并到主要目标的其他目标实体
                    contents_to_merge = [latest_entity.content, entity_content]
                    entities_to_merge_names = [latest_entity.name, entity_name]
                    entity_sources_to_merge = [latest_entity.source_document, source_document]
                    
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
                                    other_entity = other_info.get('entity')
                                    entity_sources_to_merge.append(other_entity.source_document if other_entity else "")
                    
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
                                    entity_sources_to_merge.append(candidate_entity.source_document)
                    
                    # 判断是否需要更新
                    need_update = self.llm_client.judge_content_need_update(
                        latest_entity.content,
                        entity_content,
                        old_source_document=latest_entity.source_document,
                        new_source_document=source_document,
                        old_name=latest_entity.name,
                        new_name=entity_name,
                        object_type="实体",
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
                        merged_content = self.llm_client.merge_multiple_entity_contents(
                            contents_to_merge,
                            entity_sources=entity_sources_to_merge,
                            entity_names=entities_to_merge_names,
                        )
                        if self._entity_tree_log():
                            wprint(f"  │  ├─ 合并 {len(contents_to_merge)} 个实体的content: {', '.join(entities_to_merge_names[:3])}{'...' if len(entities_to_merge_names) > 3 else ''}")
                        
                        # 创建新版本
                        final_entity = self._create_entity_version(
                            primary_target_id,
                            merged_name,
                            merged_content,
                            memory_cache_id,
                            source_document,
                            base_time=base_time,
                        )
                    else:
                        final_entity = latest_entity
                    
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
            
            if self._entity_tree_log():
                wprint(f"  │  ├─ 关系: {entity1_name} <-> {entity2_name}")
            
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

            if matched:
                # 有合并决策但未成功生成 final_entity，尝试取第一个候选作为兜底
                if self._entity_tree_log():
                    wprint(f"  │  ⚠️ 合并决策存在但未生成最终实体，使用兜底逻辑")
                first_target_id = merge_decisions[0].get("target_entity_id", "")
                if first_target_id:
                    fallback_entity = self.storage.get_entity_by_entity_id(first_target_id)
                    if fallback_entity:
                        final_entity = fallback_entity
                        entity_name_to_id[entity_name] = final_entity.entity_id
                        entity_name_to_id[final_entity.name] = final_entity.entity_id

            if not final_entity:
                # 没有匹配或兜底失败，创建新实体
                final_entity = self._create_new_entity(entity_name, entity_content, memory_cache_id, source_document, base_time=base_time)
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
        if self._entity_tree_log():
            if final_entity:
                if updated_relations:
                    wprint(f"  └─ 完成: {final_entity.name} ({final_entity.entity_id}), 关系 {len(updated_relations)} 个")
                else:
                    wprint(f"  └─ 完成: {final_entity.name} ({final_entity.entity_id})")
            else:
                if updated_relations:
                    wprint(f"  └─ 完成: 关系 {len(updated_relations)} 个")
        
        return final_entity, updated_relations, entity_name_to_id
    
    def _build_new_entity(self, name: str, content: str, memory_cache_id: str,
                          source_document: str = "", base_time: Optional[datetime] = None) -> Entity:
        """构建新实体对象，但不立即写库。"""
        event_time = base_time if base_time is not None else datetime.now()
        processed_time = datetime.now()
        entity_id = f"ent_{uuid.uuid4().hex[:12]}"
        entity_record_id = f"entity_{processed_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        source_document_only = source_document.split('/')[-1] if source_document else ""

        entity = Entity(
            absolute_id=entity_record_id,
            entity_id=entity_id,
            name=name,
            content=content,
            event_time=event_time,
            processed_time=processed_time,
            memory_cache_id=memory_cache_id,
            source_document=source_document_only,
            content_format="markdown",
        )
        return entity

    def _create_new_entity(self, name: str, content: str, memory_cache_id: str,
                           source_document: str = "", base_time: Optional[datetime] = None) -> Entity:
        """创建新实体"""
        entity = self._build_new_entity(name, content, memory_cache_id, source_document, base_time=base_time)
        self.storage.save_entity(entity)
        return entity

    def _build_entity_version(self, entity_id: str, name: str, content: str,
                              memory_cache_id: str, source_document: str = "",
                              base_time: Optional[datetime] = None) -> Entity:
        """构建实体新版本对象，但不立即写库。"""
        event_time = base_time if base_time is not None else datetime.now()
        processed_time = datetime.now()
        entity_record_id = f"entity_{processed_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        source_document_only = source_document.split('/')[-1] if source_document else ""

        entity = Entity(
            absolute_id=entity_record_id,
            entity_id=entity_id,
            name=name,
            content=content,
            event_time=event_time,
            processed_time=processed_time,
            memory_cache_id=memory_cache_id,
            source_document=source_document_only,
            content_format="markdown",
        )
        return entity

    def _create_entity_version(self, entity_id: str, name: str, content: str,
                              memory_cache_id: str, source_document: str = "",
                              base_time: Optional[datetime] = None,
                              old_content: str = "",
                              old_content_format: str = "plain") -> Entity:
        """创建实体的新版本，并记录 section 级 patches。"""
        entity = self._build_entity_version(entity_id, name, content, memory_cache_id, source_document, base_time=base_time)
        self.storage.save_entity(entity)

        # 计算 section patches
        if old_content:
            patches = self._compute_entity_patches(
                entity_id=entity_id,
                old_content=old_content,
                old_content_format=old_content_format,
                new_content=content,
                new_absolute_id=entity.absolute_id,
                source_document=source_document_only if source_document else "",
                event_time=entity.event_time,
            )
            if patches:
                self.storage.save_content_patches(patches)

        return entity
    
    def _compute_entity_patches(
        self,
        entity_id: str,
        old_content: str,
        old_content_format: str,
        new_content: str,
        new_absolute_id: str,
        source_document: str = "",
        event_time: Optional[datetime] = None,
    ) -> list:
        """计算新旧内容之间的 section 级变更 patches。"""
        old_sections = content_to_sections(old_content, old_content_format, ENTITY_SECTIONS)
        new_sections = content_to_sections(new_content, "markdown", ENTITY_SECTIONS)
        diff = compute_section_diff(old_sections, new_sections)
        if not has_any_change(diff):
            return []
        patches = []
        for key, info in diff.items():
            if not info.get("changed", False):
                continue
            patches.append(ContentPatch(
                uuid=str(uuid.uuid4()),
                target_type="Entity",
                target_absolute_id=new_absolute_id,
                target_entity_id=entity_id,
                section_key=key,
                change_type=info.get("change_type", "modified"),
                old_hash=section_hash(info.get("old", "") or ""),
                new_hash=section_hash(info.get("new", "") or ""),
                diff_summary=f"Section '{key}' {info.get('change_type', 'modified')}",
                source_document=source_document,
                event_time=event_time or datetime.now(),
            ))
        return patches

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
