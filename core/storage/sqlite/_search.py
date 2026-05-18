"""Search mixin — BM25 and vector similarity for entities & relations."""

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from ...models import Entity, Relation
from ...perf import _perf_timer
from .helpers import ENTITY_COLUMNS, RELATION_COLUMNS, _row_to_entity, _row_to_relation

logger = logging.getLogger(__name__)


class _SearchMixin:

    def search_entities_by_bm25(self, query: str, limit: int = 20) -> List[Entity]:
        if not query:
            return []
        cache_key = f"bm25_entity:{hash(query)}:{limit}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            conn = self._connect()
            try:
                raw_limit = min(limit * 5, 500)
                # FTS5 MATCH query - split into tokens and join with OR for broad matching
                tokens = query.replace('"', '').split()
                if tokens:
                    fts_query = ' OR '.join(f'"{t}"' for t in tokens[:10])
                else:
                    fts_query = '""'
                rows = conn.execute(
                    f"SELECT rowid FROM entity_fts WHERE entity_fts MATCH ? AND graph_id = ? ORDER BY rank LIMIT ?",
                    (fts_query, self._graph_id, raw_limit),
                ).fetchall()
                if not rows:
                    # Fallback to LIKE
                    rows = conn.execute(
                        f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity "
                        f"WHERE (name LIKE ? OR content LIKE ?) AND graph_id = ? "
                        f"ORDER BY processed_time DESC LIMIT ?",
                        (f"%{query}%", f"%{query}%", self._graph_id, raw_limit),
                    ).fetchall()
                else:
                    # Get actual entity data from the matched FTS rows
                    rowids = [r["rowid"] for r in rows]
                    # Get the corresponding entity uuids by joining
                    rid_ph = ",".join("?" * len(rowids))
                    rows = conn.execute(
                        f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity "
                        f"WHERE rowid IN ({rid_ph}) AND graph_id = ? "
                        f"ORDER BY processed_time DESC",
                        rowids + [self._graph_id],
                    ).fetchall()
            finally:
                conn.rollback()
            seen_fids = set()
            entities = []
            for row in rows:
                entity = _row_to_entity(dict(row))
                if entity.family_id and entity.family_id in seen_fids:
                    continue
                if entity.family_id:
                    seen_fids.add(entity.family_id)
                entities.append(entity)
                if len(entities) >= limit:
                    break
            # Prefix match supplement
            _has_core_match = False
            for ent in entities:
                name = ent.name
                if name == query or name.startswith(query + "(") or name.startswith(query + "("):
                    _has_core_match = True
                    break
            if not _has_core_match and len(query) >= 2:
                conn = self._connect()
                try:
                    prefix_rows = conn.execute(
                        f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity "
                        f"WHERE (name LIKE ? OR name = ?) AND graph_id = ? "
                        f"ORDER BY processed_time DESC LIMIT 5",
                        (query + "%", query, self._graph_id),
                    ).fetchall()
                finally:
                    conn.rollback()
                for r in prefix_rows:
                    entity = _row_to_entity(dict(r))
                    if entity.family_id and entity.family_id not in seen_fids:
                        seen_fids.add(entity.family_id)
                        entities.append(entity)
            # Resolve redirects
            raw_fids = [e.family_id for e in entities if e.family_id]
            if raw_fids:
                resolved_map = self.resolve_family_ids(raw_fids)
                for ent in entities:
                    resolved_fid = resolved_map.get(ent.family_id, ent.family_id) if ent.family_id else ent.family_id
                    if resolved_fid != ent.family_id:
                        ent.family_id = resolved_fid
            result = entities[:limit]
            self._cache.set(cache_key, result, ttl=30)
            return result
        except Exception as e:
            logger.warning("BM25 search failed: %s", e)
            self._cache.set(cache_key, [], ttl=10)
            return []

    def _search_with_embedding(self, query_text: str, entities_with_embeddings: List[tuple],
                                threshold: float, use_content: bool = False,
                                max_results: int = 10, content_snippet_length: int = 50,
                                text_mode: str = "name_and_content", query_embedding=None) -> List[Entity]:
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
        entities_with_emb = self._get_entities_with_embeddings()
        if not entities_with_emb:
            return self.search_entities_by_bm25(query_text, limit=max_results * 3)[:max_results]

        # Fast path: HNSW approximate nearest neighbor search
        if self._entity_hnsw is not None and self._entity_hnsw_items is not None:
            k = min(max_results * 3, len(self._entity_hnsw_items))
            try:
                labels, distances = self._entity_hnsw.knn_query(query_emb.reshape(1, -1), k=k)
                seen = set()
                results = []
                for idx, dist in zip(labels[0], distances[0]):
                    score = 1.0 - dist  # cosine distance -> similarity
                    if score < threshold:
                        break
                    entity = self._entity_hnsw_items[idx]
                    if entity.family_id in seen:
                        continue
                    seen.add(entity.family_id)
                    results.append(entity)
                    if len(results) >= max_results:
                        break
                if results:
                    return results
            except Exception:
                pass  # fall through to brute-force

        # Brute-force cosine similarity against all entity embeddings
        scored = []
        for entity, emb_array in entities_with_emb:
            if emb_array is None:
                continue
            score = float(np.dot(query_emb, emb_array))
            if score >= threshold:
                scored.append((score, entity))
        scored.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        results = []
        for score, entity in scored:
            if entity.family_id in seen:
                continue
            seen.add(entity.family_id)
            results.append(entity)
            if len(results) >= max_results:
                break
        return results

    def search_entities_by_similarity(self, query_name: str, query_content: Optional[str] = None,
                                       threshold: float = 0.7, max_results: int = 10,
                                       content_snippet_length: int = 50,
                                       text_mode: Literal["name_only", "content_only", "name_and_content"] = "name_and_content",
                                       similarity_method: Literal["embedding", "text", "jaccard", "bleu"] = "embedding",
                                       query_embedding=None) -> List[Entity]:
        cache_key = f"sim_search:{hash(query_name)}:{hash(query_content or '')}:{threshold}:{max_results}:{text_mode}:{similarity_method}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        with _perf_timer("search_entities_by_similarity"):
            if text_mode == "name_only":
                query_text = query_name
            elif text_mode == "content_only":
                if not query_content:
                    self._cache.set(cache_key, [], ttl=30)
                    return []
                query_text = query_content
            else:
                query_text = f"{query_name} {query_content}" if query_content else query_name
            if similarity_method == "embedding" and self.embedding_client and self.embedding_client.is_available():
                result = self._search_with_embedding(query_text, [], threshold, False, max_results, content_snippet_length, text_mode, query_embedding=query_embedding)
            else:
                result = self.search_entities_by_bm25(query_text, limit=max_results * 3)[:max_results]
            self._cache.set(cache_key, result, ttl=30)
            return result

    def search_relations_by_bm25(self, query: str, limit: int = 20, include_candidates: bool = False) -> List[Relation]:
        if not query:
            return []
        cache_key = f"bm25_relation:{hash(query)}:{limit}:{include_candidates}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            conn = self._connect()
            try:
                raw_limit = min(limit * 5, 500)
                # FTS5 MATCH query - split into tokens and join with OR for broad matching
                tokens = query.replace('"', '').split()
                if tokens:
                    fts_query = ' OR '.join(f'"{t}"' for t in tokens[:10])
                else:
                    fts_query = '""'
                fts_rows = conn.execute(
                    "SELECT rowid FROM relation_fts WHERE relation_fts MATCH ? AND graph_id = ? ORDER BY rank LIMIT ?",
                    (fts_query, self._graph_id, raw_limit),
                ).fetchall()
                if not fts_rows:
                    rows = conn.execute(
                        f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation "
                        f"WHERE content LIKE ? AND graph_id = ? "
                        f"ORDER BY processed_time DESC LIMIT ?",
                        (f"%{query}%", self._graph_id, raw_limit),
                    ).fetchall()
                else:
                    rowids = [r["rowid"] for r in fts_rows]
                    rid_ph = ",".join("?" * len(rowids))
                    rows = conn.execute(
                        f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation "
                        f"WHERE rowid IN ({rid_ph}) AND graph_id = ? "
                        f"ORDER BY processed_time DESC",
                        rowids + [self._graph_id],
                    ).fetchall()
            finally:
                conn.rollback()
            seen_fids = set()
            relations = []
            for r in rows:
                rel = _row_to_relation(dict(r))
                if rel.family_id and rel.family_id in seen_fids:
                    continue
                if rel.family_id:
                    seen_fids.add(rel.family_id)
                relations.append(rel)
                if len(relations) >= limit:
                    break
            result = self._filter_dream_candidates(relations, include_candidates)
            self._cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.warning("Relation BM25 search failed: %s", e)
            return []

    def _search_relations_with_embedding(self, query_text: str, relations_with_embeddings: List[tuple],
                                          threshold: float, max_results: int, query_embedding=None) -> List[Relation]:
        if query_embedding is None:
            query_embedding = self.embedding_client.encode(query_text)
        if query_embedding is None:
            return []
        query_emb = np.asarray(query_embedding, dtype=np.float32)
        if query_emb.ndim > 1:
            query_emb = query_emb[0]
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm
        rels_with_emb = self._get_relations_with_embeddings()
        if not rels_with_emb:
            return []

        # Fast path: HNSW approximate nearest neighbor search
        if self._relation_hnsw is not None and self._relation_hnsw_items is not None:
            k = min(max_results * 3, len(self._relation_hnsw_items))
            try:
                labels, distances = self._relation_hnsw.knn_query(query_emb.reshape(1, -1), k=k)
                seen = set()
                results = []
                for idx, dist in zip(labels[0], distances[0]):
                    score = 1.0 - dist
                    if score < threshold:
                        break
                    rel = self._relation_hnsw_items[idx]
                    if rel.family_id in seen:
                        continue
                    seen.add(rel.family_id)
                    results.append(rel)
                    if len(results) >= max_results:
                        break
                if results:
                    return results
            except Exception:
                pass

        # Brute-force cosine similarity
        scored = []
        for rel, emb_array in rels_with_emb:
            if emb_array is None:
                continue
            score = float(np.dot(query_emb, emb_array))
            if score >= threshold:
                scored.append((score, rel))
        scored.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        results = []
        for score, rel in scored:
            if rel.family_id in seen:
                continue
            seen.add(rel.family_id)
            results.append(rel)
            if len(results) >= max_results:
                break
        return results

    def search_relations_by_similarity(self, query_text: str, threshold: float = 0.3,
                                       max_results: int = 10, include_candidates: bool = False,
                                       query_embedding=None) -> List[Relation]:
        if self.embedding_client and self.embedding_client.is_available():
            results = self._search_relations_with_embedding(query_text, [], threshold, max_results, query_embedding=query_embedding)
            return self._filter_dream_candidates(results, include_candidates)
        else:
            return self.search_relations_by_bm25(query_text, limit=max_results, include_candidates=include_candidates)
