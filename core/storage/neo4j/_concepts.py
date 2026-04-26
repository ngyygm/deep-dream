"""Neo4j ConceptMixin — extracted from neo4j_store."""
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...models import Entity, Relation
import numpy as np
from ._helpers import _neo4j_types_to_native

try:
    import jieba as _jieba
except ImportError:
    _jieba = None

logger = logging.getLogger(__name__)

# Pre-compiled regex for token splitting fallback
_TOKEN_SPLIT_RE = re.compile(r'[\s,;，；、]+')


class ConceptMixin:
    """Concept operations for Neo4j backend.
    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              → Neo4j session factory
        self._run(session, cypher, **kw) → execute Cypher with graph_id injection
        self._graph_id: str          → active graph ID
    """


    def count_concepts(self, role: str = None, time_point: str = None) -> int:
        """统计概念数量。可选 time_point 过滤。"""
        tp = self._tp_to_datetime(time_point)
        with self._session() as session:
            if role:
                result = self._run(session, """
                    MATCH (c:Concept {role: $role})
                    WHERE ($tp IS NULL OR c.valid_at IS NULL OR c.valid_at <= $tp)
                      AND ($tp IS NULL OR c.invalid_at IS NULL OR c.invalid_at > $tp)
                    RETURN count(c) AS cnt
                """, role=role, tp=tp)
            else:
                result = self._run(session, """
                    MATCH (c:Concept)
                    WHERE ($tp IS NULL OR c.valid_at IS NULL OR c.valid_at <= $tp)
                      AND ($tp IS NULL OR c.invalid_at IS NULL OR c.invalid_at > $tp)
                    RETURN count(c) AS cnt
                """, tp=tp)
            return result.single()["cnt"]



    def get_concept_by_family_id(self, family_id: str, time_point: str = None) -> Optional[dict]:
        """获取任意 role 的概念最新版本。可选 time_point 过滤仅返回当时有效的版本。"""
        tp = self._tp_to_datetime(time_point)
        with self._session() as session:
            result = self._run(session, """
                MATCH (c:Concept {family_id: $fid})
                WHERE ($tp IS NULL OR c.valid_at IS NULL OR c.valid_at <= $tp)
                  AND ($tp IS NULL OR c.invalid_at IS NULL OR c.invalid_at > $tp)
                RETURN c.uuid AS id, c.family_id AS family_id, c.role AS role,
                       c.name AS name, c.content AS content,
                       c.event_time AS event_time, c.processed_time AS processed_time,
                       c.source_document AS source_document, c.summary AS summary,
                       c.confidence AS confidence
                ORDER BY c.processed_time DESC LIMIT 1
            """, fid=family_id, tp=tp)
            r = result.single()
            if not r:
                return None
            return _neo4j_types_to_native(dict(r))



    def get_concept_mentions(self, family_id: str, time_point: str = None) -> List[dict]:
        """获取提及此概念的所有 Episode。可选 time_point 过滤。"""
        return self.get_concept_provenance(family_id, time_point=time_point)



    def get_concept_neighbors(self, family_id: str, max_depth: int = 1,
                               time_point: str = None) -> List[dict]:
        """获取概念的邻居（无论 role）。可选 time_point 过滤。

        统一概念模型下的邻居语义：
        - entity → Relation 概念（引用此实体的关系）+ RELATES_TO 连接的 Entity
        - relation → 端点 Entity 概念 + MENTIONS 此关系的 Episode
        - observation → MENTIONS 连接的所有 Concept（entity/relation）
        """
        # First get the concept with time_point filtering
        concept = self.get_concept_by_family_id(family_id, time_point=time_point)
        if not concept:
            return []
        abs_id = concept.get("id")
        role = concept.get("role")
        if not abs_id or not role:
            return []

        tp = self._tp_to_datetime(time_point)
        neighbors = []
        with self._session() as session:
            if role == 'entity':
                # 1a. RELATES_TO 连接的 Entity 邻居
                result = self._run(session, """
                    MATCH (e:Entity {uuid: $abs_id})-[r:RELATES_TO]-(other:Entity)
                    WHERE ($tp IS NULL OR other.valid_at IS NULL OR other.valid_at <= $tp)
                      AND ($tp IS NULL OR other.invalid_at IS NULL OR other.invalid_at > $tp)
                    RETURN DISTINCT other.family_id AS family_id, other.uuid AS id,
                           other.name AS name, 'entity' AS role, other.content AS content
                """, abs_id=abs_id, tp=tp)
                neighbors.extend(_neo4j_types_to_native(dict(r)) for r in result)
                # 1b. 引用此实体的 Relation 概念邻居
                result = self._run(session, """
                    MATCH (rel:Relation)
                    WHERE (rel.entity1_absolute_id = $abs_id OR rel.entity2_absolute_id = $abs_id)
                      AND ($tp IS NULL OR rel.valid_at IS NULL OR rel.valid_at <= $tp)
                      AND ($tp IS NULL OR rel.invalid_at IS NULL OR rel.invalid_at > $tp)
                    RETURN DISTINCT rel.family_id AS family_id, rel.uuid AS id,
                           rel.name AS name, 'relation' AS role, rel.content AS content
                """, abs_id=abs_id, tp=tp)
                neighbors.extend(_neo4j_types_to_native(dict(r)) for r in result)
            elif role == 'relation':
                # 2a. 端点 Entity 邻居
                result = self._run(session, """
                    MATCH (r:Relation {uuid: $abs_id})
                    MATCH (e:Entity)
                    WHERE (e.uuid = r.entity1_absolute_id OR e.uuid = r.entity2_absolute_id)
                      AND ($tp IS NULL OR e.valid_at IS NULL OR e.valid_at <= $tp)
                      AND ($tp IS NULL OR e.invalid_at IS NULL OR e.invalid_at > $tp)
                    RETURN DISTINCT e.family_id AS family_id, e.uuid AS id,
                           e.name AS name, 'entity' AS role, e.content AS content
                """, abs_id=abs_id, tp=tp)
                neighbors.extend(_neo4j_types_to_native(dict(r)) for r in result)
                # 2b. MENTIONS 此关系的 Episode 邻居
                result = self._run(session, """
                    MATCH (ep:Episode)-[:MENTIONS]->(r:Relation {uuid: $abs_id})
                    WHERE ($tp IS NULL OR ep.valid_at IS NULL OR ep.valid_at <= $tp)
                      AND ($tp IS NULL OR ep.invalid_at IS NULL OR ep.invalid_at > $tp)
                    RETURN DISTINCT ep.family_id AS family_id, ep.uuid AS id,
                           ep.content AS name, 'observation' AS role, ep.content AS content
                """, abs_id=abs_id, tp=tp)
                neighbors.extend(_neo4j_types_to_native(dict(r)) for r in result)
            elif role == 'observation':
                result = self._run(session, """
                    MATCH (ep:Episode {uuid: $abs_id})-[:MENTIONS]->(c:Concept)
                    WHERE ($tp IS NULL OR c.valid_at IS NULL OR c.valid_at <= $tp)
                      AND ($tp IS NULL OR c.invalid_at IS NULL OR c.invalid_at > $tp)
                    RETURN DISTINCT c.family_id AS family_id, c.uuid AS id,
                           c.name AS name, c.role AS role, c.content AS content
                """, abs_id=abs_id, tp=tp)
                neighbors.extend(_neo4j_types_to_native(dict(r)) for r in result)
            else:
                return []
            # Dedup by family_id
            seen = set()
            deduped = []
            for n in neighbors:
                fid = n.get('family_id', '')
                if fid and fid not in seen:
                    seen.add(fid)
                    deduped.append(n)
            return deduped



    def get_concept_provenance(self, family_id: str, time_point: str = None) -> List[dict]:
        """溯源：返回所有提及此概念的 observation。可选 time_point 过滤。

        根据 concept role 分派到正确的 Cypher 查询：
        - entity: Episode->Entity MENTIONS（含 indirect via Relation fallback）
        - relation: Episode->Relation MENTIONS
        - observation: 查找此 episode 本身（返回自身）
        """
        concept = self.get_concept_by_family_id(family_id, time_point=time_point)
        if not concept:
            return []
        role = concept.get("role")
        abs_id = concept.get("id")
        if not abs_id:
            return []

        tp = self._tp_to_datetime(time_point)

        if role == 'entity':
            # Delegate to existing entity provenance (handles direct + indirect)
            return self.get_entity_provenance(family_id)

        elif role == 'relation':
            # Episode->Relation MENTIONS
            with self._session() as session:
                result = self._run(session, """
                    MATCH (ep:Episode)-[m:MENTIONS]->(r:Relation)
                    WHERE r.family_id = $fid AND r.graph_id = $graph_id
                      AND ($tp IS NULL OR ep.event_time IS NULL OR ep.event_time <= $tp)
                    RETURN DISTINCT ep.uuid AS episode_id, m.context AS context,
                           ep.content AS content, ep.source_document AS source_document
                """, fid=family_id, graph_id=self._graph_id, tp=tp, graph_id_safe=False)
                return [{"episode_id": r["episode_id"], "context": r.get("context", ""),
                         "content": r.get("content", ""), "source_document": r.get("source_document", "")}
                        for r in result]

        elif role == 'observation':
            # Observation IS the episode itself — return self-reference
            return [{"episode_id": abs_id, "role": "observation",
                     "content": concept.get("content", "")}]

        return []



    def get_episode_concepts(self, episode_id: str) -> List[dict]:
        """获取 Episode 提及的所有概念（entity + relation，返回统一 concept 格式）。"""
        with self._session() as session:
            # 1. MENTIONS -> Entity concepts
            ent_result = self._run(session, """
                MATCH (ep:Episode {uuid: $eid})-[:MENTIONS]->(e:Entity)
                WHERE e.graph_id = $graph_id
                RETURN DISTINCT e.family_id AS family_id, e.uuid AS id,
                       'entity' AS role, e.name AS name, e.content AS content
            """, eid=episode_id, graph_id=self._graph_id, graph_id_safe=False)
            # 2. MENTIONS -> Relation concepts
            rel_result = self._run(session, """
                MATCH (ep:Episode {uuid: $eid})-[:MENTIONS]->(r:Relation)
                WHERE r.graph_id = $graph_id
                RETURN DISTINCT r.family_id AS family_id, r.uuid AS id,
                       'relation' AS role, '' AS name, r.content AS content
            """, eid=episode_id, graph_id=self._graph_id, graph_id_safe=False)

            results = [_neo4j_types_to_native(dict(r)) for r in ent_result]
            results.extend(_neo4j_types_to_native(dict(r)) for r in rel_result)
            return results



    def list_concepts(self, role: str = None, limit: int = 50, offset: int = 0,
                       time_point: str = None) -> List[dict]:
        """列出概念（分页 + 可选 role 过滤 + 可选 time_point 过滤）。"""
        tp = self._tp_to_datetime(time_point)
        with self._session() as session:
            if role:
                result = self._run(session, """
                    MATCH (c:Concept {role: $role})
                    WHERE ($tp IS NULL OR c.valid_at IS NULL OR c.valid_at <= $tp)
                      AND ($tp IS NULL OR c.invalid_at IS NULL OR c.invalid_at > $tp)
                    RETURN c.uuid AS id, c.family_id AS family_id, c.role AS role,
                           c.name AS name, c.content AS content,
                           c.event_time AS event_time, c.processed_time AS processed_time
                    ORDER BY c.processed_time DESC SKIP $offset LIMIT $limit
                """, role=role, offset=offset, limit=limit, tp=tp)
            else:
                result = self._run(session, """
                    MATCH (c:Concept)
                    WHERE ($tp IS NULL OR c.valid_at IS NULL OR c.valid_at <= $tp)
                      AND ($tp IS NULL OR c.invalid_at IS NULL OR c.invalid_at > $tp)
                    RETURN c.uuid AS id, c.family_id AS family_id, c.role AS role,
                           c.name AS name, c.content AS content,
                           c.event_time AS event_time, c.processed_time AS processed_time
                    ORDER BY c.processed_time DESC SKIP $offset LIMIT $limit
                """, offset=offset, limit=limit, tp=tp)
            return [_neo4j_types_to_native(dict(r)) for r in result]



    def traverse_concepts(self, start_family_ids: List[str], max_depth: int = 2,
                           time_point: str = None) -> dict:
        """BFS 遍历概念图。可选 time_point 过滤。

        统一概念模型：遍历 entity/relation/observation 所有 role 的概念节点，
        返回 concepts（所有概念）和 edges（概念间连接关系）。

        Optimized: batch-fetches frontier concepts + neighbors per BFS level
        using 3 Cypher queries (not 2*N sessions).
        """
        if not start_family_ids:
            return {"concepts": {}, "edges": [], "relations": [], "visited_count": 0}

        tp = self._tp_to_datetime(time_point)
        visited = set()
        queue = list(start_family_ids)
        all_concepts = {}
        all_edges = []

        for _ in range(max_depth):
            frontier = [fid for fid in queue if fid not in visited]
            if not frontier:
                break
            visited.update(frontier)

            # --- Batch fetch all frontier concepts (1 query) ---
            with self._session() as session:
                concept_result = self._run(session, """
                    UNWIND $fids AS fid
                    MATCH (c:Concept {family_id: fid})
                    WHERE ($tp IS NULL OR c.valid_at IS NULL OR c.valid_at <= $tp)
                      AND ($tp IS NULL OR c.invalid_at IS NULL OR c.invalid_at > $tp)
                    WITH fid, c
                    ORDER BY c.processed_time DESC
                    WITH fid, collect(c)[0] AS c
                    RETURN c.uuid AS id, c.family_id AS family_id, c.role AS role,
                           c.name AS name, c.content AS content,
                           c.event_time AS event_time, c.processed_time AS processed_time,
                           c.source_document AS source_document, c.summary AS summary,
                           c.confidence AS confidence
                """, fids=frontier, tp=tp)

                frontier_abs_ids = {}  # family_id → abs_id
                for r in concept_result:
                    concept = _neo4j_types_to_native(dict(r))
                    fid = concept.get('family_id', '')
                    if fid:
                        all_concepts[fid] = concept
                        frontier_abs_ids[fid] = concept.get('id', '')

                if not frontier_abs_ids:
                    break

                abs_id_list = list(frontier_abs_ids.values())  # Neo4j driver needs list for UNWIND
                abs_to_fid = {v: k for k, v in frontier_abs_ids.items()}

                # --- Batch RELATES_TO neighbors (1 query) ---
                relates_result = self._run(session, """
                    UNWIND $abs_ids AS aid
                    MATCH (c:Entity {uuid: aid})-[:RELATES_TO]-(other:Entity)
                    WHERE ($tp IS NULL OR other.valid_at IS NULL OR other.valid_at <= $tp)
                      AND ($tp IS NULL OR other.invalid_at IS NULL OR other.invalid_at > $tp)
                    RETURN DISTINCT aid AS source_abs_id,
                           other.family_id AS family_id, other.uuid AS id,
                           other.name AS name, 'entity' AS role
                """, abs_ids=abs_id_list, tp=tp)

                # --- Batch Relation neighbors (1 query) ---
                rel_result = self._run(session, """
                    UNWIND $abs_ids AS aid
                    MATCH (rel:Relation)
                    WHERE (rel.entity1_absolute_id = aid OR rel.entity2_absolute_id = aid)
                      AND ($tp IS NULL OR rel.valid_at IS NULL OR rel.valid_at <= $tp)
                      AND ($tp IS NULL OR rel.invalid_at IS NULL OR rel.invalid_at > $tp)
                    RETURN DISTINCT aid AS source_abs_id,
                           rel.family_id AS family_id, rel.uuid AS id,
                           rel.name AS name, 'relation' AS role
                """, abs_ids=abs_id_list, tp=tp)

            # Collect edges and next frontier
            next_queue = []
            for r in relates_result:
                n = _neo4j_types_to_native(dict(r))
                nfid = n.get('family_id', '')
                source_fid = abs_to_fid.get(n.get('source_abs_id', ''))
                if nfid and source_fid:
                    all_edges.append({"from": source_fid, "to": nfid,
                                      "to_role": n.get('role', ''), "to_name": n.get('name', '')})
                    if nfid not in visited:
                        next_queue.append(nfid)

            for r in rel_result:
                n = _neo4j_types_to_native(dict(r))
                nfid = n.get('family_id', '')
                source_fid = abs_to_fid.get(n.get('source_abs_id', ''))
                if nfid and source_fid:
                    all_edges.append({"from": source_fid, "to": nfid,
                                      "to_role": n.get('role', ''), "to_name": n.get('name', '')})
                    if nfid not in visited:
                        next_queue.append(nfid)

            queue = next_queue

        # Backward compat: also provide "relations" key for old consumers
        relation_concepts = [c for c in all_concepts.values() if c.get('role') == 'relation']

        return {
            "concepts": all_concepts,
            "edges": all_edges,
            "relations": relation_concepts,
            "visited_count": len(visited),
        }

    def search_concepts_by_bm25(self, query: str, role: str = None, limit: int = 20,
                             time_point: str = None) -> List[dict]:
        """搜索概念（Neo4j 全文索引优先，回退到分词 CONTAINS）。可选 time_point 过滤。"""
        if not query:
            return []
        tp = self._tp_to_datetime(time_point)
        # Pre-compute jieba segmentation once (used in both fulltext and fallback paths)
        _jieba_tokens = None
        if _jieba:
            try:
                _jieba_tokens = [t.strip() for t in _jieba.cut(query) if t.strip()]
            except Exception:
                pass
        with self._session() as session:
            role_filter = " AND c.role = $role" if role else ""
            # 尝试全文索引搜索（真正的 BM25 排序）
            # For Chinese text, segment with jieba and construct AND query for better precision
            try:
                # Build Lucene query: segment Chinese text and join with AND
                try:
                    if _jieba_tokens is not None:
                        if len(_jieba_tokens) > 1:
                            search_query = " AND ".join(_jieba_tokens)
                        else:
                            search_query = query
                    else:
                        search_query = query
                except Exception:
                    search_query = query
                result = self._run(session,
                    f"""CALL db.index.fulltext.queryNodes('conceptFulltext', $search_query)
                       YIELD node, score
                       WHERE node.invalid_at IS NULL
                         AND ($tp IS NULL OR node.valid_at IS NULL OR node.valid_at <= $tp)
                         AND ($tp IS NULL OR node.invalid_at IS NULL OR node.invalid_at > $tp)
                         {role_filter.replace('c.', 'node.')}
                       RETURN node.uuid AS id, node.family_id AS family_id, node.role AS role,
                              node.name AS name, node.content AS content,
                              node.event_time AS event_time, node.processed_time AS processed_time,
                              score AS bm25_score
                       ORDER BY score DESC LIMIT $limit""",
                    search_query=search_query, role=role, limit=limit, tp=tp,
                )
                rows = [_neo4j_types_to_native(dict(r)) for r in result]
                if rows:
                    return rows
            except Exception as e:
                logger.debug("Concept fulltext search failed, falling back to CONTAINS: %s", e)

            # 回退: jieba 中文分词 + CONTAINS（每个词独立匹配，OR 组合）
            # Use jieba for proper Chinese word segmentation (reuse cached tokens)
            if _jieba_tokens is not None:
                tokens = [t for t in _jieba_tokens if len(t) >= 2]
            else:
                tokens = [t for t in _TOKEN_SPLIT_RE.split(query) if len(t) >= 2]
            if not tokens:
                tokens = [query]
            # 构建分词 CONTAINS 条件 — single pass
            conditions = []
            params = {}
            for i, token in enumerate(tokens):
                conditions.append(f"(c.content CONTAINS $t_{i} OR c.name CONTAINS $t_{i})")
                params[f"t_{i}"] = token
            token_clauses = " OR ".join(conditions)
            params["role"] = role
            params["limit"] = limit
            params["tp"] = tp

            cypher = f"""
                MATCH (c:Concept)
                WHERE ({token_clauses})
                  AND ($tp IS NULL OR c.valid_at IS NULL OR c.valid_at <= $tp)
                  AND ($tp IS NULL OR c.invalid_at IS NULL OR c.invalid_at > $tp)
                  {role_filter}
                RETURN c.uuid AS id, c.family_id AS family_id, c.role AS role,
                       c.name AS name, c.content AS content,
                       c.event_time AS event_time, c.processed_time AS processed_time
                ORDER BY c.processed_time DESC LIMIT $limit
            """
            result = self._run(session, cypher, **params)
            return [_neo4j_types_to_native(dict(r)) for r in result]

    def search_concepts_by_similarity(self, query_text: str, role: str = None,
                                   threshold: float = 0.5, max_results: int = 20,
                                   time_point: str = None) -> List[dict]:
        """概念语义搜索：搜索 entity_vectors + relation_vectors sqlite-vec 表，
        然后从 Neo4j 批量获取匹配的 Concept 节点。

        Args:
            query_text: 查询文本
            role: 可选过滤 'entity' | 'relation' | 'observation'
            threshold: 余弦相似度阈值
            max_results: 最大返回数
            time_point: 可选时间点过滤

        Returns:
            List[dict] with concept fields (id, family_id, role, name, content, ...)
        """
        if not query_text or not self.embedding_client or not self.embedding_client.is_available():
            return []

        # 1. Encode + normalize query
        query_embedding = self.embedding_client.encode(query_text)
        if query_embedding is None:
            return []

        query_emb = np.asarray(
            query_embedding[0] if isinstance(query_embedding, (list, np.ndarray)) else query_embedding,
            dtype=np.float32
        )
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm

        # 2. Determine which vector tables to search based on role
        tables = []
        fallback_to_bm25 = False
        if role is None:
            tables = ["entity_vectors", "relation_vectors"]
            # observation has no vector table — always supplement with BM25
            fallback_to_bm25 = True
        elif role == "entity":
            tables = ["entity_vectors"]
        elif role == "relation":
            tables = ["relation_vectors"]
        else:
            # observation/episode — no vector table; fall back to BM25
            fallback_to_bm25 = True

        # 3. KNN search across selected tables
        knn_limit = max_results * 5
        uuid_dist: Dict[str, float] = {}

        for table in tables:
            try:
                knn_results = self._vector_store.search(
                    table, [], limit=knn_limit,
                    _precomputed_bytes=query_emb.astype(np.float32).tobytes()
                )
                for uuid, dist in knn_results:
                    # Keep lowest distance per uuid
                    if uuid not in uuid_dist or dist < uuid_dist[uuid]:
                        uuid_dist[uuid] = dist
            except Exception as e:
                logger.debug("Concept similarity search on %s failed: %s", table, e)

        # 4. Batch fetch concepts from Neo4j by uuid (skip if no vector hits)
        seen: Dict[str, float] = {}  # family_id -> best score
        results = []

        if uuid_dist:
            tp = self._tp_to_datetime(time_point)
            uuids = list(uuid_dist)

            with self._session() as session:
                role_filter = " AND c.role = $role" if role else ""
                result = self._run(session,
                    f"""
                    MATCH (c:Concept)
                    WHERE c.uuid IN $uuids
                      AND ($tp IS NULL OR c.valid_at IS NULL OR c.valid_at <= $tp)
                      AND ($tp IS NULL OR c.invalid_at IS NULL OR c.invalid_at > $tp)
                      {role_filter}
                    RETURN c.uuid AS id, c.family_id AS family_id, c.role AS role,
                           c.name AS name, c.content AS content,
                           c.event_time AS event_time, c.processed_time AS processed_time,
                           c.source_document AS source_document, c.summary AS summary,
                           c.confidence AS confidence
                    """,
                    uuids=uuids,
                    role=role,
                    tp=tp,
                )
                concepts = [_neo4j_types_to_native(dict(r)) for r in result]

            # 5. Filter by threshold + dedup by family_id (keep highest similarity)
            scored = []
            for c in concepts:
                uuid = c.get("id")
                l2_dist_sq = uuid_dist.get(uuid)
                if l2_dist_sq is None:
                    continue
                cos_sim = 1.0 - l2_dist_sq / 2.0
                scored.append((c, cos_sim))

            scored.sort(key=lambda x: x[1], reverse=True)

            for c, cos_sim in scored:
                if cos_sim < threshold:
                    continue
                fid = c.get("family_id")
                if fid in seen:
                    continue
                seen[fid] = cos_sim
                c["score"] = round(cos_sim, 4)
                results.append(c)
                if len(results) >= max_results:
                    break

        # BM25 fallback for observations (no vector table)
        if fallback_to_bm25:
            bm25_role = role if role in ("observation",) else None
            try:
                bm25_results = self.search_concepts_by_bm25(
                    query_text, role=bm25_role, limit=max_results,
                    time_point=time_point,
                )
                for c in bm25_results:
                    fid = c.get("family_id") or c.get("id")
                    if fid and fid not in seen:
                        score = c.get("bm25_score", c.get("score", 0.0))
                        seen[fid] = score
                        c["score"] = round(float(score), 4)
                        results.append(c)
                        if len(results) >= max_results:
                            break
            except Exception as e:
                logger.debug("Concept similarity BM25 fallback failed: %s", e)

        return results

