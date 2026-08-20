"""
搜索结果聚类：HybridSearcher.cluster_results 按语义相似度对搜索结果聚类。

历史上此类还封装了三路混合搜索（BM25 + embedding + 图上下文扩展 + RRF 融合），
该死代码面已删除；仅保留 cluster_results（core/server/routes/concepts.py 在用）。
"""

from typing import Any, List, Optional


class HybridSearcher:
    """搜索结果聚类器（cluster_results 的宿主，供 server routes 复用）。"""

    def __init__(self, storage: Any):
        """
        Args:
            storage: SQLiteGraphStorageManager 实例
        """
        self.storage = storage

    # ------------------------------------------------------------------
    # Phase B2: Result clustering
    # ------------------------------------------------------------------

    def cluster_results(
        self,
        items: List[dict],
        num_clusters: int = 5,
        sim_threshold: float = 0.5,
    ) -> List[dict]:
        """Cluster search results by semantic similarity.

        Returns a list of cluster dicts: {"label", "count", "items"}.
        Each item retains its original fields plus a "cluster_label" field.

        Uses greedy agglomerative clustering on cosine similarity of
        item embeddings. Falls back to bigram Jaccard if embeddings
        are unavailable.
        """
        if not items or len(items) < 3:
            return []

        n = min(len(items), 100)  # cap for efficiency
        items = items[:n]
        num_clusters = max(2, min(num_clusters, n // 2))

        # Build similarity matrix
        try:
            import numpy as _np

            # Try to get embeddings from items
            emb_list: List[Optional[_np.ndarray]] = []
            has_embeddings = False
            for item in items:
                emb = item.get("_embedding")
                if emb is not None and isinstance(emb, (list, _np.ndarray)):
                    arr = _np.array(emb, dtype=_np.float32).reshape(-1)
                    norm = _np.linalg.norm(arr)
                    if norm > 0:
                        arr = arr / norm
                    emb_list.append(arr)
                    has_embeddings = True
                else:
                    emb_list.append(None)

            if has_embeddings and sum(1 for e in emb_list if e is not None) >= n * 0.5:
                # Build matrix from available embeddings
                mat = _np.zeros((n, emb_list[0].size), dtype=_np.float32)
                for i, emb in enumerate(emb_list):
                    if emb is not None:
                        mat[i] = emb
                    else:
                        # Use zero vector for missing embeddings
                        pass
                sim_matrix = mat @ mat.T
                # Clamp negative similarities to 0
                sim_matrix = _np.maximum(sim_matrix, 0.0)
            else:
                sim_matrix = None
        except Exception:
            sim_matrix = None

        # Fallback: Jaccard on bigrams
        if sim_matrix is None:
            def _bigrams(s: str):
                if len(s) < 2:
                    return frozenset(s) if s else frozenset()
                return frozenset(s[i:i + 2] for i in range(len(s) - 1))

            item_sets = []
            for item in items:
                text = (item.get("name") or "") + " " + (item.get("content") or "")
                item_sets.append(_bigrams(text))

            sim_matrix = _np.zeros((n, n), dtype=_np.float64)
            for i in range(n):
                for j in range(i + 1, n):
                    u = len(item_sets[i] | item_sets[j])
                    sim = len(item_sets[i] & item_sets[j]) / u if u else 0.0
                    sim_matrix[i][j] = sim
                    sim_matrix[j][i] = sim

        # Greedy agglomerative clustering
        # Each cluster is a set of indices; start with each item as its own cluster
        clusters: List[set] = [{i} for i in range(n)]

        while len(clusters) > num_clusters:
            best_sim = -1.0
            best_pair = (0, 1)
            for ci in range(len(clusters)):
                for cj in range(ci + 1, len(clusters)):
                    # Average pairwise similarity between clusters
                    total_sim = 0.0
                    count = 0
                    for a in clusters[ci]:
                        for b in clusters[cj]:
                            total_sim += sim_matrix[a][b]
                            count += 1
                    avg_sim = total_sim / count if count > 0 else 0.0
                    if avg_sim > best_sim:
                        best_sim = avg_sim
                        best_pair = (ci, cj)

            if best_sim < sim_threshold:
                break  # no more similar pairs to merge

            ci, cj = best_pair
            clusters[ci] = clusters[ci] | clusters[cj]
            clusters.pop(cj)

        # Build result: label = shortest name among top-scored items (concept names
        # are short; dialogue fragments are long). Fall back to highest-scored if all
        # names are long (>20 chars).
        result = []
        for cluster in clusters:
            if not cluster:
                continue
            # Find shortest name among items — concept names are concise
            min_name_idx = min(cluster, key=lambda i: len(items[i].get("name", "zzz")))
            min_name_len = len(items[min_name_idx].get("name", ""))
            # If shortest name is still long (>20 chars), use highest-scored instead
            if min_name_len <= 20:
                label = items[min_name_idx].get("name", "Other")
            else:
                best_idx = max(cluster, key=lambda i: items[i].get("_score", 0.0) or items[i].get("relevance", 0.0))
                label = items[best_idx].get("name", "Other")
            cluster_items = []
            for idx in cluster:
                item = dict(items[idx])  # shallow copy
                item["cluster_label"] = label
                cluster_items.append(item)
            result.append({
                "label": label,
                "count": len(cluster_items),
                "items": cluster_items,
            })

        # Sort clusters by total score (sum of _score in cluster) descending
        result.sort(key=lambda c: sum(
            (it.get("_score", 0.0) or it.get("relevance", 0.0) or 0.0) for it in c["items"]
        ), reverse=True)
        return result
