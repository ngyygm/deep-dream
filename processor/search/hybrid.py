"""
混合搜索：BM25 全文搜索 + 向量语义搜索 + 图上下文扩展 + RRF 融合排序。

HybridSearcher 封装了三路搜索（BM25 + embedding + graph-context），
使用 Reciprocal Rank Fusion (RRF) 将多路结果合并为统一排序列表。
可选 confidence 加权重排序，确保低置信度实体排名靠后。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..models import Entity, Relation

logger = logging.getLogger(__name__)


class HybridSearcher:
    """混合搜索引擎：BM25 + 向量搜索 + 图上下文扩展 + RRF 融合。"""

    def __init__(self, storage: Any):
        """
        Args:
            storage: StorageManager 或 Neo4jStorageManager 实例
        """
        self.storage = storage

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Compute embedding via storage's embedding_client if available."""
        emb_client = getattr(self.storage, 'embedding_client', None)
        if emb_client and getattr(emb_client, 'is_available', lambda: False)():
            try:
                return emb_client.encode(text)
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
            from .graph_traversal import GraphTraversalSearcher
            traverser = GraphTraversalSearcher(self.storage)
            return traverser.bfs_expand(seed_family_ids, max_depth=max_depth, max_nodes=max_nodes)
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

        # 路径 1: BM25 全文搜索
        try:
            bm25_results = self.storage.search_entities_by_bm25(query_text, limit=semantic_max_results)
            if bm25_results:
                result_lists.append(bm25_results)
                weights.append(bm25_weight)
        except Exception as e:
            logger.debug("BM25 search failed: %s", e)

        # 路径 2: 向量语义搜索（name + content 模式）
        if query_embedding is None:
            query_embedding = self._get_embedding(query_text)
        if query_embedding is not None:
            try:
                vector_results = self.storage.search_entities_by_similarity(
                    query_name=query_text,
                    query_content=query_text,
                    threshold=semantic_threshold,
                    max_results=semantic_max_results,
                )
                if vector_results:
                    result_lists.append(vector_results)
                    weights.append(vector_weight)
            except Exception as e:
                logger.debug("Vector search failed: %s", e)

            # 路径 2b: name-only 语义搜索（短查询对短名称匹配更好）
            # 仅当向量 name+content 搜索无结果或结果太少时补充
            _total_primary = sum(len(rl) for rl in result_lists)
            if _total_primary < max(3, semantic_max_results // 2):
                try:
                    name_only_results = self.storage.search_entities_by_similarity(
                        query_name=query_text,
                        threshold=semantic_threshold,
                        max_results=semantic_max_results,
                        text_mode="name_only",
                    )
                    if name_only_results:
                        # 检查与已有结果的重叠度
                        existing_fids = set()
                        for rl in result_lists:
                            for item in rl:
                                fid = getattr(item, 'family_id', None)
                                if fid:
                                    existing_fids.add(fid)
                        new_fids = {getattr(e, 'family_id', '') for e in name_only_results}
                        overlap = existing_fids & new_fids
                        # 有新增实体才加入（避免纯重复）
                        if new_fids - existing_fids:
                            result_lists.append(name_only_results)
                            weights.append(vector_weight * 0.5)  # name-only 权重较低
                except Exception as e:
                    logger.debug("Name-only vector search failed: %s", e)

        # 路径 3: 图上下文扩展 — 从前两路 top 结果出发，BFS 发现结构关联实体
        if enable_graph_expansion and (result_lists):
            try:
                # 从前两路融合结果的 top 种子出发
                seed_fids = []
                if result_lists:
                    pre_fused = self.reciprocal_rank_fusion(result_lists, weights[:len(result_lists)])
                    seed_fids = [e.family_id for e, _ in pre_fused[:5] if e.family_id]
                if seed_fids:
                    graph_entities = self._graph_context_expand(
                        seed_fids, max_depth=graph_depth, max_nodes=semantic_max_results
                    )
                    if graph_entities:
                        result_lists.append(graph_entities)
                        weights.append(graph_weight)
            except Exception as e:
                logger.debug("Graph expansion failed: %s", e)

        if not result_lists:
            return []

        # RRF 融合
        fused = self.reciprocal_rank_fusion(result_lists, weights)
        return fused[:top_k]

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

        # 路径 1: BM25 全文搜索
        try:
            bm25_results = self.storage.search_relations_by_bm25(query_text, limit=semantic_max_results)
            if bm25_results:
                result_lists.append(bm25_results)
                weights.append(bm25_weight)
        except Exception as e:
            logger.debug("BM25 search failed: %s", e)

        # 路径 2: 向量语义搜索（自动计算 embedding）
        if query_embedding is None:
            query_embedding = self._get_embedding(query_text)
        if query_embedding is not None:
            try:
                vector_results = self.storage.search_relations_by_similarity(
                    query_text=query_text,
                    threshold=semantic_threshold,
                    max_results=semantic_max_results,
                )
                if vector_results:
                    result_lists.append(vector_results)
                    weights.append(vector_weight)
            except Exception as e:
                logger.debug("Relation vector search failed: %s", e)

        if not result_lists:
            return []

        fused = self.reciprocal_rank_fusion(result_lists, weights)
        return fused[:top_k]

    @staticmethod
    def reciprocal_rank_fusion(
        result_lists: List[List[Any]],
        weights: List[float],
        k: int = 60,
    ) -> List[Tuple[Any, float]]:
        """Reciprocal Rank Fusion (RRF) 融合多路搜索结果。

        Args:
            result_lists: 多路搜索结果列表
            weights: 每路搜索的权重
            k: RRF 常数（默认 60），越大则排名差异的影响越小

        Returns:
            [(item, fusion_score), ...] 按 fusion_score 降序排列
        """
        scores: Dict[str, float] = {}
        items: Dict[str, Any] = {}
        item_scores: Dict[str, float] = {}  # track per-item best score

        for results, weight in zip(result_lists, weights):
            for rank, item in enumerate(results):
                # 使用 family_id 去重（同一实体不同版本只保留最高分）
                fid = getattr(item, 'family_id', None)
                key = fid if fid else item.absolute_id
                rrf_score = weight / (k + rank + 1)
                # RRF 核心逻辑：多路分数累加，同一实体取最高排名的贡献
                # 即使已在其他路径出现过，当前路径的分数仍需累加
                if key not in scores:
                    scores[key] = 0.0
                scores[key] += rrf_score
                # 保留本轮贡献最高的版本
                if key not in items or rrf_score > item_scores.get(key, 0):
                    items[key] = item
                    item_scores[key] = rrf_score

        # 按融合分数降序排列
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(items[key], score) for key, score in sorted_items]

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
        for entity, score in items:
            degree = degree_map.get(entity.family_id, 0)
            degree_factor = degree / max_degree
            adjusted = score * (1 - alpha) + degree_factor * alpha
            results.append((entity, round(adjusted, 6)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Phase C: Confidence-weighted reranking
    # ------------------------------------------------------------------

    def confidence_rerank(
        self,
        items: List[Tuple[Any, float]],
        alpha: float = 0.2,
    ) -> List[Tuple[Any, float]]:
        """置信度加权重排序：低置信度实体排名靠后。

        final_score = rrf_score * (1 - alpha + alpha * confidence)

        Args:
            items: [(Entity/Relation, score), ...] 原始排序
            alpha: 置信度影响因子（0-1）

        Returns:
            重排序后的 [(item, adjusted_score), ...]
        """
        if not items:
            return items

        results = []
        for item, score in items:
            confidence = getattr(item, 'confidence', None) or 0.5  # default 0.5 if unset
            adjusted = score * (1 - alpha + alpha * confidence)
            results.append((item, round(adjusted, 6)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
