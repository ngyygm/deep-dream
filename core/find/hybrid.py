"""
混合搜索：BM25 全文搜索 + 向量语义搜索 + 图上下文扩展 + RRF 融合排序。

HybridSearcher 封装了三路搜索（BM25 + embedding + graph-context），
使用 Reciprocal Rank Fusion (RRF) 将多路结果合并为统一排序列表。
可选 confidence 加权重排序，确保低置信度实体排名靠后。
可选时间衰减，让长期未被提及的概念自然降低排名（概念淡出）。
"""

import heapq
import logging
import math
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple

from core.models import Entity, Relation

logger = logging.getLogger(__name__)


class HybridSearcher:
    """混合搜索引擎：BM25 + 向量搜索 + 图上下文扩展 + RRF 融合。"""

    def __init__(self, storage: Any):
        """
        Args:
            storage: StorageManager 或 Neo4jStorageManager 实例
        """
        self.storage = storage
        self._traverser = None  # Lazy-initialized GraphTraversalSearcher
        self._search_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid")
        self._emb_cache: OrderedDict[str, Tuple[float, List[float]]] = OrderedDict()  # text -> (timestamp, embedding), LRU order
        self._emb_cache_ttl = 60.0  # seconds
        self._emb_cache_max = 256
        # Cache embedding_client reference to avoid repeated getattr
        self._emb_client = None
        self._emb_client_checked = False

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Compute embedding via storage's embedding_client if available. Cached for TTL seconds.

        Uses OrderedDict for O(1) LRU eviction: move_to_end on access, popitem for eviction.
        """
        now = time.time()
        cached = self._emb_cache.get(text)
        if cached is not None and (now - cached[0]) < self._emb_cache_ttl:
            self._emb_cache.move_to_end(text)  # LRU refresh
            return cached[1]
        # Cached embedding_client lookup (avoid getattr per call)
        if not self._emb_client_checked:
            ec = getattr(self.storage, 'embedding_client', None)
            self._emb_client = ec if (ec and getattr(ec, 'is_available', lambda: False)()) else None
            self._emb_client_checked = True
        emb_client = self._emb_client
        if emb_client:
            try:
                emb = emb_client.encode(text)
                if emb is not None:
                    self._emb_cache[text] = (now, emb)
                    self._emb_cache.move_to_end(text)  # New entry at end
                    # O(1) eviction: remove oldest entries
                    while len(self._emb_cache) > self._emb_cache_max:
                        self._emb_cache.popitem(last=False)
                return emb
            except Exception as e:
                logger.debug("Embedding computation failed: %s", e)
        return None

    def _graph_context_expand(
        self,
        seed_family_ids: List[str],
        max_depth: int = 1,
        max_nodes: int = 30,
    ) -> List[Entity]:
        """图上下文扩展：从种子实体出发 BFS 1-2 跳，发现结构关联实体。

        Args:
            seed_family_ids: 种子实体的 family_id 列表
            max_depth: BFS 扩展深度（默认1跳）
            max_nodes: 最多返回的节点数

        Returns:
            通过图结构发现的关联实体列表
        """
        if not seed_family_ids:
            return []

        try:
            if self._traverser is None:
                from .graph_traversal import GraphTraversalSearcher
                self._traverser = GraphTraversalSearcher(self.storage)
            return self._traverser.bfs_expand(seed_family_ids, max_depth=max_depth, max_nodes=max_nodes)
        except Exception as e:
            logger.debug("Graph context expansion failed: %s", e)
            return []

    def search_entities(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 20,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        graph_weight: float = 0.15,
        semantic_threshold: float = 0.5,
        semantic_max_results: int = 50,
        enable_graph_expansion: bool = True,
        graph_depth: int = 1,
    ) -> List[Tuple[Entity, float]]:
        """混合搜索实体。

        三路搜索: BM25 + 向量语义 + 图上下文扩展
        当两路无重叠时，使用 name-only 补充搜索提升短查询召回。

        Args:
            query_text: 搜索文本（用于 BM25）
            query_embedding: 查询向量（用于语义搜索，为 None 时自动计算）
            top_k: 最终返回数量
            vector_weight: 向量搜索权重
            bm25_weight: BM25 搜索权重
            graph_weight: 图上下文扩展权重
            semantic_threshold: 语义搜索相似度阈值
            semantic_max_results: 语义搜索最大候选数
            enable_graph_expansion: 是否启用图上下文扩展
            graph_depth: 图扩展深度（1或2）

        Returns:
            [(Entity, fusion_score), ...] 按 fusion_score 降序排列
        """
        result_lists = []
        weights = []

        # Pre-compute embedding (needed for vector search regardless of parallelism)
        if query_embedding is None:
            query_embedding = self._get_embedding(query_text)

        # Run BM25 and vector search in parallel — they are independent
        _bm25_results = []
        _vector_results = []

        def _run_bm25():
            try:
                return self.storage.search_entities_by_bm25(query_text, limit=semantic_max_results)
            except Exception as e:
                logger.debug("BM25 search failed: %s", e)
                return []

        def _run_vector():
            if query_embedding is None:
                return []
            try:
                return self.storage.search_entities_by_similarity(
                    query_name=query_text,
                    query_content=query_text,
                    threshold=semantic_threshold,
                    max_results=semantic_max_results,
                    query_embedding=query_embedding,
                )
            except Exception as e:
                logger.debug("Vector search failed: %s", e)
                return []

        _bm25_fut = self._search_pool.submit(_run_bm25)
        _vec_fut = self._search_pool.submit(_run_vector)
        _bm25_results = _bm25_fut.result() or []
        _vector_results = _vec_fut.result() or []

        # Incrementally track family_ids to avoid O(R) rebuild for overlap checks
        _accumulated_fids: set = set()
        if _bm25_results:
            result_lists.append(_bm25_results)
            weights.append(bm25_weight)
            for item in _bm25_results:
                fid = item.family_id
                if fid:
                    _accumulated_fids.add(fid)
        if _vector_results:
            result_lists.append(_vector_results)
            weights.append(vector_weight)
            for item in _vector_results:
                fid = item.family_id
                if fid:
                    _accumulated_fids.add(fid)

        if query_embedding is not None:

            # 路径 2b + 3: name-only 语义搜索和图上下文扩展互相独立，并行执行
            _total_primary = sum(len(rl) for rl in result_lists)
            _need_name_only = _total_primary < max(3, semantic_max_results // 2)
            _need_graph = enable_graph_expansion and result_lists

            if _need_name_only or _need_graph:
                # Collect seed fids upfront for graph expansion (needed by both paths)
                _seed_fids_for_graph = []
                if _need_graph:
                    _seen = set()
                    for rl in result_lists:
                        for item in rl:
                            fid = item.family_id
                            if fid and fid not in _seen:
                                _seed_fids_for_graph.append(fid)
                                _seen.add(fid)
                                if len(_seed_fids_for_graph) >= 5:
                                    break
                        if len(_seed_fids_for_graph) >= 5:
                            break
                    _need_graph = bool(_seed_fids_for_graph)

                def _run_name_only():
                    if not _need_name_only:
                        return None
                    try:
                        return self.storage.search_entities_by_similarity(
                            query_name=query_text,
                            threshold=semantic_threshold,
                            max_results=semantic_max_results,
                            text_mode="name_only",
                            query_embedding=query_embedding,
                        )
                    except Exception as e:
                        logger.debug("Name-only vector search failed: %s", e)
                        return None

                def _run_graph():
                    if not _need_graph:
                        return None
                    try:
                        return self._graph_context_expand(
                            _seed_fids_for_graph, max_depth=graph_depth, max_nodes=semantic_max_results
                        )
                    except Exception as e:
                        logger.debug("Graph expansion failed: %s", e)
                        return None

                _name_fut = self._search_pool.submit(_run_name_only)
                _graph_fut = self._search_pool.submit(_run_graph)
                name_only_results = _name_fut.result()
                graph_entities = _graph_fut.result()

                # Merge name-only results
                if name_only_results:
                    new_fids = {e.family_id for e in name_only_results if e.family_id}
                    if new_fids - _accumulated_fids:
                        result_lists.append(name_only_results)
                        weights.append(vector_weight * 0.5)
                        _accumulated_fids.update(new_fids)

                # Merge graph results
                if graph_entities:
                    result_lists.append(graph_entities)
                    weights.append(graph_weight)

        if not result_lists:
            return []

        # RRF 融合
        return self.reciprocal_rank_fusion(result_lists, weights, top_k=top_k)

    def search_relations(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 20,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        semantic_threshold: float = 0.3,
        semantic_max_results: int = 50,
    ) -> List[Tuple[Relation, float]]:
        """混合搜索关系。参数同 search_entities。"""
        result_lists = []
        weights = []

        # Pre-compute embedding
        if query_embedding is None:
            query_embedding = self._get_embedding(query_text)

        # Run BM25 and vector search in parallel
        def _run_bm25():
            try:
                return self.storage.search_relations_by_bm25(query_text, limit=semantic_max_results)
            except Exception as e:
                logger.debug("BM25 search failed: %s", e)
                return []

        def _run_vector():
            if query_embedding is None:
                return []
            try:
                return self.storage.search_relations_by_similarity(
                    query_text=query_text,
                    threshold=semantic_threshold,
                    max_results=semantic_max_results,
                    query_embedding=query_embedding,
                )
            except Exception as e:
                logger.debug("Relation vector search failed: %s", e)
                return []

        _bm25_fut = self._search_pool.submit(_run_bm25)
        _vec_fut = self._search_pool.submit(_run_vector)
        _bm25_results = _bm25_fut.result() or []
        _vector_results = _vec_fut.result() or []

        if _bm25_results:
            result_lists.append(_bm25_results)
            weights.append(bm25_weight)
        if _vector_results:
            result_lists.append(_vector_results)
            weights.append(vector_weight)

        if not result_lists:
            return []

        return self.reciprocal_rank_fusion(result_lists, weights, top_k=top_k)

    @staticmethod
    def reciprocal_rank_fusion(
        result_lists: List[List[Any]],
        weights: List[float],
        k: int = 60,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Any, float]]:
        """Reciprocal Rank Fusion (RRF) 融合多路搜索结果。

        Args:
            result_lists: 多路搜索结果列表
            weights: 每路搜索的权重
            k: RRF 常数（默认 60），越大则排名差异的影响越小
            top_k: 只返回前 top_k 个结果（使用 heapq.nlargest，O(N log K)）

        Returns:
            [(item, fusion_score), ...] 按 fusion_score 降序排列
        """
        # Single dict: key → [total_score, best_contribution, item]
        entries: Dict[str, list] = {}

        for results, weight in zip(result_lists, weights):
            for rank, item in enumerate(results):
                # 使用 family_id 去重（同一实体不同版本只保留最高分）
                fid = item.family_id
                key = fid if fid else item.absolute_id
                rrf_score = weight / (k + rank + 1)
                entry = entries.get(key)
                if entry is None:
                    entries[key] = [rrf_score, rrf_score, item]
                else:
                    entry[0] += rrf_score
                    # 保留本轮贡献最高的版本
                    if rrf_score > entry[1]:
                        entry[1] = rrf_score
                        entry[2] = item

        if not entries:
            return []
        # top-K selection: heapq.nlargest is O(N log K) vs full sort O(N log N)
        vals_iter = entries.values()
        if top_k is not None and top_k < len(entries):
            vals = heapq.nlargest(top_k, vals_iter, key=itemgetter(0))
        else:
            vals = list(vals_iter)
            vals.sort(key=itemgetter(0), reverse=True)
        max_score = vals[0][0]
        if max_score > 0:
            inv_max = 1.0 / max_score
            return [(e[2], round(e[0] * inv_max, 4)) for e in vals]
        return [(e[2], e[0]) for e in vals]

    # ------------------------------------------------------------------
    # Phase B: MMR 多样性重排序 + Node Degree 重排序
    # ------------------------------------------------------------------

    def node_degree_rerank(
        self,
        items: List[Tuple[Entity, float]],
        degree_map: Dict[str, int],
        alpha: float = 0.3,
    ) -> List[Tuple[Entity, float]]:
        """Node Degree 重排序：优先返回连接数高的实体（更重要的实体）。

        Args:
            items: [(Entity, score), ...] 原始排序
            degree_map: {family_id: degree} 实体度数字典
            alpha: 度数影响因子（0-1）

        Returns:
            重排序后的 [(Entity, adjusted_score), ...]
        """
        if not items:
            return items
        max_degree = max(degree_map.values()) if degree_map else 1
        if max_degree == 0:
            max_degree = 1

        results = []
        inv_alpha_factor = 1 - alpha  # precompute outside loop
        for entity, score in items:
            degree = degree_map.get(entity.family_id, 0)
            degree_factor = degree / max_degree
            adjusted = score * inv_alpha_factor + degree_factor * alpha
            results.append((entity, round(adjusted, 6)))

        results.sort(key=itemgetter(1), reverse=True)
        return results

    # ------------------------------------------------------------------
    # Phase C: Confidence-weighted reranking
    # ------------------------------------------------------------------

    def confidence_rerank(
        self,
        items: List[Tuple[Any, float]],
        alpha: float = 0.2,
        time_decay_half_life_days: float = 0.0,
    ) -> List[Tuple[Any, float]]:
        """置信度 + 时间衰减重排序：低置信度、长期未更新的概念排名靠后。

        final_score = rrf_score * (1 - alpha + alpha * confidence) * time_decay

        time_decay = exp(-ln(2) * days_since_processed / half_life)
        - half_life=0: 禁用时间衰减
        - half_life=30: 30天未更新的概念衰减50%
        - half_life=90: 90天未更新的概念衰减50%

        Args:
            items: [(Entity/Relation, score), ...] 原始排序
            alpha: 置信度影响因子（0-1）
            time_decay_half_life_days: 时间衰减半衰期（天），0表示禁用

        Returns:
            重排序后的 [(item, adjusted_score), ...]
        """
        if not items:
            return items

        now = datetime.now(timezone.utc) if time_decay_half_life_days > 0 else None
        ln2 = 0.6931471805599453  # math.log(2) precomputed

        # Pre-parse all processed_time values once (avoid try/except per-item)
        _pt_cache = {}
        _conf_cache = {}
        if now is not None:
            for item, _ in items:
                pt = getattr(item, 'processed_time', None)
                if pt is not None:
                    try:
                        if isinstance(pt, str):
                            pt = datetime.fromisoformat(pt)
                        if pt.tzinfo is None:
                            pt = pt.replace(tzinfo=timezone.utc)
                        _pt_cache[id(item)] = max(0.0, (now - pt).total_seconds() / 86400.0)
                    except (ValueError, TypeError, OverflowError):
                        pass
        # Pre-cache confidence values (avoid getattr per-item)
        for item, _ in items:
            _conf_cache[id(item)] = getattr(item, 'confidence', None) or 0.5

        results = []
        inv_alpha_factor = 1 - alpha  # precompute outside loop
        for item, score in items:
            confidence = _conf_cache[id(item)]
            adjusted = score * (inv_alpha_factor + alpha * confidence)

            # Time decay: use pre-parsed days_old
            if time_decay_half_life_days > 0:
                days_old = _pt_cache.get(id(item))
                if days_old is not None:
                    decay = math.exp(-ln2 * days_old / time_decay_half_life_days)
                    adjusted *= decay

            results.append((item, round(adjusted, 6)))

        results.sort(key=itemgetter(1), reverse=True)
        return results
