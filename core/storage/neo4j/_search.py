"""Neo4j SearchMixin — extracted from neo4j_store."""
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from ...models import Entity, Relation
from ...perf import _perf_timer
from ._helpers import _ENTITY_RETURN_FIELDS, _neo4j_record_to_entity, _neo4j_record_to_relation

logger = logging.getLogger(__name__)


class SearchMixin:
    """Search operations for Neo4j backend.
    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              → Neo4j session factory
        self._run(session, cypher, **kw) → execute Cypher with graph_id injection
        self._graph_id: str          → active graph ID
        self.embedding_client        → EmbeddingClient (optional)
    """


    def _search_with_embedding(self, query_text: str, entities_with_embeddings: List[tuple],
                                threshold: float, use_content: bool = False,
                                max_results: int = 10, content_snippet_length: int = 50,
                                text_mode: str = "name_and_content",
                                query_embedding=None) -> List[Entity]:
        """使用 Neo4j 向量索引进行实体相似度搜索。"""
        # 1. Encode + 归一化 query (skip if caller provided embedding)
        if query_embedding is None:
            query_embedding = self.embedding_client.encode(query_text)
        if query_embedding is None:
            return self.search_entities_by_bm25(query_text, limit=max_results * 3)[:max_results]

        query_emb = np.asarray(query_embedding, dtype=np.float32)
        if query_emb.ndim > 1:
            query_emb = query_emb[0]
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm

        # 2. Neo4j vector index KNN
        knn_limit = max_results * 5
        query_vector = query_emb.tolist()
        with self._session() as session:
            try:
                result = session.run(
                    """
                    CALL db.index.vector.queryNodes('entity_embedding', $k, $queryVector)
                    YIELD node, score
                    WHERE node.graph_id = $graph_id AND node.invalid_at IS NULL
                    RETURN node, score
                    ORDER BY score DESC
                    """,
                    k=knn_limit,
                    queryVector=query_vector,
                    graph_id=self._graph_id,
                )
                records = list(result)
            except Exception as e:
                logger.warning("Neo4j vector search failed, falling back to BM25: %s", e)
                return self.search_entities_by_bm25(query_text, limit=max_results * 3)[:max_results]

        if not records:
            return []

        # 3. 去重（同 family_id 取最高分）+ 过滤 threshold
        seen = set()
        results = []
        for record in records:
            node = record["node"]
            score = record["score"]
            if score < threshold:
                break
            family_id = node.get("family_id")
            if family_id in seen:
                continue
            seen.add(family_id)
            # Convert node properties to entity via the standard helper
            entity_dict = {
                "uuid": node.get("uuid"),
                "family_id": family_id,
                "name": node.get("name", ""),
                "content": node.get("content", ""),
                "summary": node.get("summary"),
                "attributes": node.get("attributes"),
                "confidence": node.get("confidence"),
                "content_format": node.get("content_format", "plain"),
                "community_id": node.get("community_id"),
                "valid_at": node.get("valid_at"),
                "invalid_at": node.get("invalid_at"),
                "event_time": node.get("event_time"),
                "processed_time": node.get("processed_time"),
                "episode_id": node.get("episode_id", ""),
                "source_document": node.get("source_document", ""),
                "embedding": node.get("embedding"),
            }
            entity = _neo4j_record_to_entity(entity_dict)
            results.append(entity)
            if len(results) >= max_results:
                break
        return results

    # ------------------------------------------------------------------
    # Relation 操作
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------

    def search_entities_by_bm25(self, query: str, limit: int = 20) -> List[Entity]:
        """BM25 全文搜索实体（Neo4j 5.x 全文索引），去重 family_id 只保留最高分版本。

        增强逻辑：当 BM25 结果中不包含核心名称匹配的实体时，用前缀匹配补充
        （解决"曹操"搜不到"曹操（155年－220年）"的问题）。
        """
        if not query:
            return []
        # Short TTL cache — same query within a pipeline run benefits from dedup
        cache_key = f"bm25_entity:{hash(query)}:{limit}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            with self._session() as session:
                # 多取一些再在 Python 层去重，确保 limit 个 unique family_id
                raw_limit = min(limit * 5, 500)
                result = self._run(session, 
                    """CALL db.index.fulltext.queryNodes('entityFulltext', $search_query)
                       YIELD node, score
                       WHERE node.invalid_at IS NULL
                       RETURN node.uuid AS uuid, node.family_id AS family_id,
                              node.name AS name, node.content AS content,
                              node.summary AS summary,
                              node.attributes AS attributes, node.confidence AS confidence,
                              node.content_format AS content_format, node.community_id AS community_id,
                              node.valid_at AS valid_at, node.invalid_at AS invalid_at,
                              node.event_time AS event_time,
                              node.processed_time AS processed_time,
                              node.episode_id AS episode_id,
                              node.source_document AS source_document,
                              score AS bm25_score
                       ORDER BY score DESC
                       LIMIT $raw_limit""",
                    search_query=query, raw_limit=raw_limit,
                )
                seen_fids = set()
                raw_entities = []
                for record in result:
                    fid = record.get("family_id")
                    if fid and fid in seen_fids:
                        continue
                    if fid:
                        seen_fids.add(fid)
                    raw_entities.append((fid, _neo4j_record_to_entity(record)))
                    if len(raw_entities) >= limit:
                        break

                # ---- 核心名称前缀匹配补充 ----
                # 检查 BM25 结果中是否已有核心名称匹配
                _has_core_match = False
                _prefix_cjk = query + '\uff08'  # '（'
                _prefix_ascii = query + '('
                for _, ent in raw_entities:
                    name = getattr(ent, 'name', '')
                    # 精确匹配 或 名称以 query 开头（消歧括号场景）
                    if name == query or name.startswith(_prefix_cjk) or name.startswith(_prefix_ascii):
                        _has_core_match = True
                        break
                if not _has_core_match and len(query) >= 2:
                    # 用前缀匹配补充：query="曹操" → 匹配 "曹操（155年－220年）"
                    # Inline Cypher within same session to avoid opening a 2nd session
                    try:
                        prefix_result = self._run(session,
                            """
                            MATCH (e:Entity)
                            WHERE (e.name STARTS WITH $prefix OR e.name = $prefix)
                              AND e.invalid_at IS NULL
                            RETURN e.uuid AS uuid, e.family_id AS family_id,
                                   e.name AS name, e.content AS content,
                                   e.summary AS summary,
                                   e.attributes AS attributes, e.confidence AS confidence,
                                   e.content_format AS content_format, e.community_id AS community_id,
                                   e.source_document AS source_document, e.episode_id AS episode_id,
                                   e.processed_time AS processed_time, e.event_time AS event_time,
                                   e.valid_at AS valid_at, e.invalid_at AS invalid_at
                            ORDER BY e.processed_time DESC
                            LIMIT $prefix_limit
                            """,
                            prefix=query, prefix_limit=5,
                        )
                        for record in prefix_result:
                            fid = record.get("family_id")
                            if fid and fid in seen_fids:
                                continue
                            if fid:
                                seen_fids.add(fid)
                            raw_entities.append((fid, _neo4j_record_to_entity(record)))
                    except Exception:
                        pass

                # Resolve family_id redirects (merged entities point to canonical id)
                raw_fids = [fid for fid, _ in raw_entities if fid]
                resolved_map = self.resolve_family_ids(raw_fids) if raw_fids else {}

                entities = []
                for fid, ent in raw_entities:
                    resolved_fid = resolved_map.get(fid, fid) if fid else fid
                    if resolved_fid != fid:
                        ent.family_id = resolved_fid  # Mutate — dataclasses are mutable
                    entities.append(ent)
                result = entities[:limit]
                self._cache.set(cache_key, result, ttl=30)
                return result
        except Exception as e:
            logger.warning("BM25 search failed, falling back to empty: %s", e)
            self._cache.set(cache_key, [], ttl=10)
            return []


    # ------------------------------------------------------------------

    def search_entities_by_similarity(self, query_name: str, query_content: Optional[str] = None,
                                       threshold: float = 0.7, max_results: int = 10,
                                       content_snippet_length: int = 50,
                                       text_mode: Literal["name_only", "content_only", "name_and_content"] = "name_and_content",
                                       similarity_method: Literal["embedding", "text", "jaccard", "bleu"] = "embedding",
                                       query_embedding=None) -> List[Entity]:
        """根据相似度搜索实体。"""
        # 结果缓存
        cache_key = f"sim_search:{hash(query_name)}:{hash(query_content or '')}:{threshold}:{max_results}:{text_mode}:{similarity_method}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        with _perf_timer("search_entities_by_similarity"):
            if text_mode == "name_only":
                query_text = query_name
                use_content = False
            elif text_mode == "content_only":
                if not query_content:
                    self._cache.set(cache_key, [], ttl=30)
                    return []
                query_text = query_content
                use_content = True
            else:
                if query_content:
                    query_text = f"{query_name} {query_content}"
                else:
                    query_text = query_name
                use_content = query_content is not None

            if similarity_method == "embedding" and self.embedding_client and self.embedding_client.is_available():
                # KNN 路径：不需要加载全量数据
                result = self._search_with_embedding(
                    query_text, [], threshold,
                    use_content, max_results, content_snippet_length, text_mode,
                    query_embedding=query_embedding,
                )
            else:
                # BM25 替代 Jaccard 全量扫描，走 Neo4j 全文索引
                result = self.search_entities_by_bm25(query_text, limit=max_results * 3)
                result = result[:max_results]
            self._cache.set(cache_key, result, ttl=30)
            return result



    def search_relations_by_bm25(self, query: str, limit: int = 20,
                                  include_candidates: bool = False) -> List[Relation]:
        """BM25 全文搜索关系（Neo4j 5.x 全文索引），去重 family_id 只保留最高分版本。"""
        if not query:
            return []
        # Short TTL cache — same query within a pipeline run benefits from dedup
        cache_key = f"bm25_relation:{hash(query)}:{limit}:{include_candidates}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            with self._session() as session:
                raw_limit = min(limit * 5, 500)
                result = self._run(session, 
                    """CALL db.index.fulltext.queryNodes('relationFulltext', $search_query)
                       YIELD node, score
                       WHERE node.invalid_at IS NULL
                       RETURN node.uuid AS uuid, node.family_id AS family_id,
                              node.entity1_absolute_id AS entity1_absolute_id,
                              node.entity2_absolute_id AS entity2_absolute_id,
                              node.content AS content,
                              node.event_time AS event_time,
                              node.processed_time AS processed_time,
                              node.episode_id AS episode_id,
                              node.source_document AS source_document,
                              node.valid_at AS valid_at,
                              node.invalid_at AS invalid_at,
                              node.summary AS summary,
                              node.attributes AS attributes,
                              node.confidence AS confidence,
                              node.provenance AS provenance,
                              score AS bm25_score
                       ORDER BY score DESC
                       LIMIT $raw_limit""",
                    search_query=query, raw_limit=raw_limit,
                )
                seen_fids = set()
                relations = []
                for record in result:
                    fid = record.get("family_id")
                    if fid and fid in seen_fids:
                        continue
                    if fid:
                        seen_fids.add(fid)
                    relations.append(_neo4j_record_to_relation(record))
                    if len(relations) >= limit:
                        break
                result = self._filter_dream_candidates(relations, include_candidates)
                self._cache.set(cache_key, result)
                return result
        except Exception as e:
            logger.warning("BM25 search failed, falling back to empty: %s", e)
            return []

    # ------------------------------------------------------------------
    # Entity Search
    # ------------------------------------------------------------------



    def search_relations_by_similarity(self, query_text: str,
                                       threshold: float = 0.3,
                                       max_results: int = 10,
                                       include_candidates: bool = False,
                                       query_embedding=None) -> List[Relation]:
        """根据相似度搜索关系。"""
        if self.embedding_client and self.embedding_client.is_available():
            # KNN 路径：不需要加载全量数据
            results = self._search_relations_with_embedding(
                query_text, [], threshold, max_results, query_embedding=query_embedding
            )
            return self._filter_dream_candidates(results, include_candidates)
        else:
            # BM25 替代文本全量扫描
            return self.search_relations_by_bm25(query_text, limit=max_results,
                                                  include_candidates=include_candidates)

