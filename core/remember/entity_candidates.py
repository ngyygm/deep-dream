"""
Entity candidate generation — simplified embedding-first approach.

Retrieval strategy:
1. Neo4j vector index top-K search (primary)
2. Exact name dict lookup from projections (supplement)

That's it. No Jaccard matrix, BM25, content-mention, neighbor expansion, etc.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.debug_log import log_struct as _dbg_struct
from core.utils import wprint_info
from .helpers import _PAREN_ANNOTATION_RE
from functools import lru_cache
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Name normalization (used by entity.py and cross_window.py)
# ---------------------------------------------------------------------------

_TITLE_SUFFIXES_RE = re.compile(
    r'(?:教授|博士|先生|女士|同学|老师|工程师|经理|总监|院长|所长|主任|校长|站长|馆长|主编|首席|总裁'
    r'|部长|省长|市长|县长|区长|镇长|村长|将军|上校|中校|少校|大校|司令|参谋|政委|舰长|机长)$'
)


@lru_cache(maxsize=4096)
def normalize_entity_name_for_matching(name: str) -> str:
    core = _PAREN_ANNOTATION_RE.sub('', name).strip()
    core = _TITLE_SUFFIXES_RE.sub('', core).strip()
    return core


# ---------------------------------------------------------------------------
# Candidate table builder
# ---------------------------------------------------------------------------

class EntityCandidateBuilder:
    """Embedding-first candidate builder for entity alignment.

    Two retrieval channels:
    1. Neo4j vector index top-K — semantic similarity via embedding cosine
    2. Exact name lookup — O(1) dict match on name / core-name
    """

    def __init__(self, storage, llm_client, *,
                 max_alignment_candidates: Optional[int] = None,
                 max_similar_entities: int = 10,
                 merge_safe_embedding_threshold: float = 0.55,
                 merge_safe_jaccard_threshold: float = 0.4,
                 verbose: bool = True,
                 entity_progress_verbose: bool = False):
        self.storage = storage
        self.llm_client = llm_client
        self.max_alignment_candidates = max_alignment_candidates
        self.max_similar_entities = max_similar_entities
        self.merge_safe_embedding_threshold = merge_safe_embedding_threshold
        self.verbose = verbose
        self.entity_progress_verbose = entity_progress_verbose

    def _entity_tree_log(self) -> bool:
        return self.verbose and self.entity_progress_verbose

    def build_candidate_table(
        self,
        extracted_entities: List[Dict[str, str]],
        similarity_threshold: float,
        jaccard_search_threshold: Optional[float] = None,
        embedding_name_search_threshold: Optional[float] = None,
        embedding_full_search_threshold: Optional[float] = None,
        prefetched_embeddings: Optional[Tuple[Optional[Any], Optional[Any]]] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Build candidate table: vector top-K + exact name lookup."""
        _t0 = time.monotonic()

        # ── Fetch projections & build lookup dicts ──
        projections = self.storage.get_latest_entities_projection(
            self.llm_client.effective_entity_snippet_length()
        )
        if not projections:
            wprint_info("[candidate_table] ⚠️ No existing entities for alignment")
            return {}

        name_to_proj: Dict[str, Dict] = {}
        core_to_proj: Dict[str, Dict] = {}
        fid_to_proj: Dict[str, Dict] = {}
        for p in projections:
            fid_to_proj[p["family_id"]] = p
            name_to_proj[p["name"]] = p
            core = normalize_entity_name_for_matching(p["name"])
            p["_core_name"] = core
            if core not in core_to_proj:
                core_to_proj[core] = p

        wprint_info(f"[candidate_table] {len(projections)} existing entities")

        # ── Encode extracted entities ──
        name_embeddings: Optional[Any] = None
        full_embeddings: Optional[Any] = None
        if prefetched_embeddings is not None:
            name_embeddings, full_embeddings = prefetched_embeddings
        elif self.storage.embedding_client and self.storage.embedding_client.is_available():
            _N = len(extracted_entities)
            _snippet_len = self.llm_client.effective_entity_snippet_length()
            _name_texts = [e["name"] for e in extracted_entities]
            _full_texts = [
                f"# {e['name']}\n{e['content'][:_snippet_len]}"
                for e in extracted_entities
            ]
            _all_embs = self.storage.embedding_client.encode(_name_texts + _full_texts)
            name_embeddings = _all_embs[:_N]
            full_embeddings = _all_embs[_N:]

        _t_encode = time.monotonic()
        wprint_info(f"[candidate_timing] projections + encode: {_t_encode - _t0:.3f}s")

        # ── Neo4j vector top-K search ──
        top_k = max(self.max_alignment_candidates or 50, 50)
        name_emb_scores, full_emb_scores = self._search_embedding_top_k(
            extracted_entities, name_embeddings, full_embeddings, top_k,
        )

        _t_vec = time.monotonic()
        wprint_info(f"[candidate_timing] vector top-K: {_t_vec - _t_encode:.3f}s")

        # ── Build per-entity candidates ──
        candidate_table: Dict[int, List[Dict[str, Any]]] = {}
        limit = self.max_alignment_candidates or self.max_similar_entities
        for idx, ee in enumerate(extracted_entities):
            candidates = self._build_candidates_for_entity(
                idx, ee,
                name_to_proj, core_to_proj, fid_to_proj,
                name_emb_scores.get(idx, {}),
                full_emb_scores.get(idx, {}),
            )
            candidates.sort(key=lambda c: c["combined_score"], reverse=True)
            candidate_table[idx] = candidates[:limit]

        _t_build = time.monotonic()
        wprint_info(f"[candidate_timing] build + rank: {_t_build - _t_vec:.3f}s")
        wprint_info(f"[candidate_timing] TOTAL: {_t_build - _t0:.3f}s")

        # Debug trace
        for idx, ee in enumerate(extracted_entities):
            rows = candidate_table.get(idx, [])
            top3 = "; ".join(
                f"{r.get('name','?')}(score={r.get('combined_score',0):.3f},type={r.get('name_match_type','?')})"
                for r in rows[:3]
            )
            _dbg_struct("candidate_table_built",
                        entity_name=ee["name"],
                        n_candidates=len(rows),
                        top3=top3)

        return candidate_table

    def _build_candidates_for_entity(
        self,
        idx: int,
        ee: Dict[str, str],
        name_to_proj: Dict[str, Dict],
        core_to_proj: Dict[str, Dict],
        fid_to_proj: Dict[str, Dict],
        name_emb_scores: Dict[str, float],
        full_emb_scores: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Build candidates from vector search results + exact name match."""
        ee_name = ee["name"]
        ee_core = normalize_entity_name_for_matching(ee_name)
        seen_fids: set = set()
        candidates: List[Dict[str, Any]] = []

        # ── 1. Vector search results ──
        all_emb: Dict[str, float] = {}
        for fid, score in name_emb_scores.items():
            all_emb[fid] = max(all_emb.get(fid, 0.0), score)
        for fid, score in full_emb_scores.items():
            all_emb[fid] = max(all_emb.get(fid, 0.0), score)

        for fid, dense_score in all_emb.items():
            proj = fid_to_proj.get(fid)
            if not proj:
                continue
            seen_fids.add(fid)
            name_match = bool(ee_core and proj.get("_core_name") == ee_core)
            candidates.append({
                "family_id": fid,
                "name": proj["name"],
                "content": proj["content"],
                "source_document": (proj.get("entity").source_document
                                    if proj.get("entity") else ""),
                "version_count": proj.get("version_count", 1),
                "entity": proj.get("entity"),
                "lexical_score": 0.90 if name_match else 0.0,
                "dense_score": dense_score,
                "combined_score": max(dense_score, 0.90 if name_match else 0.0),
                "merge_safe": name_match or dense_score >= self.merge_safe_embedding_threshold,
                "name_match_type": "exact" if name_match else "embedding",
            })

        # ── 2. Exact name / core-name lookup ──
        for lookup_name, lookup_dict in ((ee_name, name_to_proj),
                                         (ee_core, core_to_proj)):
            if not lookup_name or len(lookup_name) < 2:
                continue
            proj = lookup_dict.get(lookup_name)
            if not proj or proj["family_id"] in seen_fids:
                continue
            fid = proj["family_id"]
            seen_fids.add(fid)
            candidates.append({
                "family_id": fid,
                "name": proj["name"],
                "content": proj["content"],
                "source_document": (proj.get("entity").source_document
                                    if proj.get("entity") else ""),
                "version_count": proj.get("version_count", 1),
                "entity": proj.get("entity"),
                "lexical_score": 0.90,
                "dense_score": 0.0,
                "combined_score": 0.90,
                "merge_safe": True,
                "name_match_type": "exact",
            })

        return candidates

    # ------------------------------------------------------------------
    # Neo4j vector search (unchanged)
    # ------------------------------------------------------------------

    def _search_embedding_top_k(
        self,
        extracted_entities: List[Dict[str, str]],
        name_embeddings,
        full_embeddings,
        top_k: int,
    ) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
        """Neo4j vector index top-K per extracted entity."""
        name_scores: Dict[int, Dict[str, float]] = {}
        full_scores: Dict[int, Dict[str, float]] = {}

        if not hasattr(self.storage, '_session'):
            return name_scores, full_scores

        name_queries: List[Tuple[int, List[float]]] = []
        full_queries: List[Tuple[int, List[float]]] = []

        for idx in range(len(extracted_entities)):
            if name_embeddings is not None:
                query_emb = np.asarray(
                    name_embeddings[idx]
                    if name_embeddings.ndim == 1 or idx < len(name_embeddings)
                    else None,
                    dtype=np.float32,
                )
                if query_emb is not None and query_emb.size > 0:
                    norm = np.linalg.norm(query_emb)
                    if norm > 0:
                        query_emb = query_emb / norm
                    name_queries.append((idx, query_emb.tolist()))

            if full_embeddings is not None:
                query_emb = np.asarray(
                    full_embeddings[idx]
                    if full_embeddings.ndim == 1 or idx < len(full_embeddings)
                    else None,
                    dtype=np.float32,
                )
                if query_emb is not None and query_emb.size > 0:
                    norm = np.linalg.norm(query_emb)
                    if norm > 0:
                        query_emb = query_emb / norm
                    full_queries.append((idx, query_emb.tolist()))

        if name_queries:
            batch_results = self._neo4j_vector_search_batch(
                [qv for _, qv in name_queries], top_k
            )
            for (orig_idx, _), scores in zip(name_queries, batch_results):
                if scores:
                    name_scores[orig_idx] = scores

        if full_queries:
            batch_results = self._neo4j_vector_search_batch(
                [qv for _, qv in full_queries], top_k
            )
            for (orig_idx, _), scores in zip(full_queries, batch_results):
                if scores:
                    full_scores[orig_idx] = scores

        return name_scores, full_scores

    def _neo4j_vector_search_batch(
        self, query_vectors: List[List[float]], top_k: int,
    ) -> List[Dict[str, float]]:
        """Execute multiple Neo4j vector queries in a single session."""
        all_results: List[Dict[str, float]] = [{} for _ in query_vectors]
        if not query_vectors:
            return all_results
        try:
            with self.storage._session() as session:
                graph_id = getattr(self.storage, '_graph_id', 'default')
                for i, qv in enumerate(query_vectors):
                    result = session.run(
                        """
                        CALL db.index.vector.queryNodes('entity_embedding', $k, $queryVector)
                        YIELD node, score
                        WHERE node.graph_id = $graph_id AND node.invalid_at IS NULL
                        RETURN node.family_id AS family_id, score
                        ORDER BY score DESC
                        """,
                        k=top_k, queryVector=qv, graph_id=graph_id,
                    )
                    for record in result:
                        fid = record["family_id"]
                        if fid:
                            all_results[i][fid] = float(record["score"])
        except Exception as e:
            logger.debug("Neo4j vector search failed: %s", e)
        return all_results
