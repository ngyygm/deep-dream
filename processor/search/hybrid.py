"""
混合搜索：BM25 全文搜索 + 向量语义搜索 + RRF 融合排序。

HybridSearcher 封装了双路搜索（BM25 + embedding 余弦相似度），
使用 Reciprocal Rank Fusion (RRF) 将两路结果合并为统一排序列表。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from ..models import Entity, Relation

logger = logging.getLogger(__name__)


class HybridSearcher:
    """混合搜索引擎：BM25 + 向量搜索 + RRF 融合。"""

    def __init__(self, storage: Any):
        """
        Args:
            storage: StorageManager 或 Neo4jStorageManager 实例
        """
        self.storage = storage

    def search_entities(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 20,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        semantic_threshold: float = 0.5,
        semantic_max_results: int = 50,
    ) -> List[Tuple[Entity, float]]:
        """混合搜索实体。

        Args:
            query_text: 搜索文本（用于 BM25）
            query_embedding: 查询向量（用于语义搜索，可选）
            top_k: 最终返回数量
            vector_weight: 向量搜索权重
            bm25_weight: BM25 搜索权重
            semantic_threshold: 语义搜索相似度阈值
            semantic_max_results: 语义搜索最大候选数

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

        # 路径 2: 向量语义搜索
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

        # 路径 2: 向量语义搜索
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
                logger.debug("Vector search failed: %s", e)

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

        for results, weight in zip(result_lists, weights):
            for rank, item in enumerate(results):
                # 使用 family_id 去重（同一实体不同版本只保留最高分）
                fid = getattr(item, 'family_id', None)
                key = fid if fid else item.absolute_id
                rrf_score = weight / (k + rank + 1)
                existing = scores.get(key)
                if existing is not None and existing >= rrf_score:
                    continue  # 已有更高分版本，跳过
                scores[key] = scores.get(key, 0) + rrf_score
                items[key] = item

        # 按融合分数降序排列
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(items[key], score) for key, score in sorted_items]

    # ------------------------------------------------------------------
    # Phase B: MMR 多样性重排序 + Node Degree 重排序
    # ------------------------------------------------------------------

    @staticmethod
    def maximal_marginal_relevance(
        items: List[Tuple[Any, float]],
        embeddings: Dict[str, List[float]],
        lambda_param: float = 0.5,
        top_k: int = 20,
    ) -> List[Tuple[Any, float]]:
        """Maximal Marginal Relevance (MMR) 多样性重排序。

        在保持相关性的同时，引入多样性，避免返回过于相似的候选。

        Args:
            items: [(item, score), ...] 按 score 降序排列的候选
            embeddings: {item_key: embedding_vector} 嵌入向量字典
            lambda_param: 相关性 vs 多样性的权衡（0=纯多样性，1=纯相关性）
            top_k: 返回数量

        Returns:
            重排序后的 [(item, mmr_score), ...]
        """
        if not items or not embeddings:
            return items[:top_k]

        selected: List[str] = []
        remaining = list(items)
        result: List[Tuple[Any, float]] = []

        for _ in range(min(top_k, len(items))):
            best_idx = -1
            best_mmr = -math.inf

            for idx, (item, score) in enumerate(remaining):
                key = item.absolute_id
                relevance = score

                if selected:
                    # 计算与已选项目的最大相似度
                    item_emb = embeddings.get(key)
                    if item_emb is None:
                        max_sim = 0.0
                    else:
                        max_sim = 0.0
                        for sel_key in selected:
                            sel_emb = embeddings.get(sel_key)
                            if sel_emb:
                                sim = HybridSearcher._cosine_sim(item_emb, sel_emb)
                                max_sim = max(max_sim, sim)
                else:
                    max_sim = 0.0

                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                best_item, best_score = remaining.pop(best_idx)
                selected.append(best_item.absolute_id)
                result.append((best_item, best_mmr))

        return result

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        """计算两个向量的余弦相似度。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

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
