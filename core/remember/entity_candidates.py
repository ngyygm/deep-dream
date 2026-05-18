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
from ._shared import (
    normalize_entity_name_for_matching,
    _get_bm25_pool, BM25_POOL_MAX,
)
from .entity_candidates_enrich import _EnrichMixin

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate table builder
# ---------------------------------------------------------------------------

class EntityCandidateBuilder(_EnrichMixin):
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

        # Vectorized similarity via embedding top-K queries
        top_k = max(len(projections), self.max_alignment_candidates or 50, 50)
        name_emb_scores, full_emb_scores = self._search_embedding_top_k(
            extracted_entities, name_embeddings, full_embeddings, top_k,
        )

        _t_vec = time.monotonic()
        wprint_info(f"[candidate_timing] embedding vector top-K search: {_t_vec - _t_encode:.3f}s")

        # Pre-compute core names for all projections (avoids E × P calls to normalize function)
        for p in projections:
            p["_core_name"] = normalize_entity_name_for_matching(p["name"])

        # Pre-compute core names + bigram sets for all extracted entities (avoids E × P recomputation)
        ext_bigrams = []
        ext_core_bigrams = []
        ext_core_names: List[str] = []
        for ee in extracted_entities:
            _n = ee["name"]
            ext_bigrams.append(_bigrams(_n.lower().strip()) if _n else _EMPTY_FROZENSET)
            _c = normalize_entity_name_for_matching(_n)
            ext_core_names.append(_c)
            ext_core_bigrams.append(_bigrams(_c.lower().strip()) if _c else _EMPTY_FROZENSET)
        proj_bigrams = []
        proj_core_bigrams = []
        for p in projections:
            _n = p["name"]
            proj_bigrams.append(_bigrams(_n.lower().strip()) if _n else _EMPTY_FROZENSET)
            proj_core_bigrams.append(_bigrams(p["_core_name"].lower().strip()) if p["_core_name"] else _EMPTY_FROZENSET)

        # Build initial candidate rows
        _t_matrix = time.monotonic()
        wprint_info(f"[candidate_timing] matrix build + precompute: {_t_matrix - _t_encode:.3f}s")

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
    # Internal: embedding vector top-K search
    # ------------------------------------------------------------------

    def _search_embedding_top_k(
        self,
        extracted_entities: List[Dict[str, str]],
        name_embeddings,
        full_embeddings,
        top_k: int,
    ) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
        """Use embedding search to find top-K similar entities per extracted entity.

        Returns:
            (name_scores, full_scores) — each is {extracted_idx: {family_id: cosine_score}}
        """
        name_scores: Dict[int, Dict[str, float]] = {}
        full_scores: Dict[int, Dict[str, float]] = {}

        if not hasattr(self.storage, '_session') and not hasattr(self.storage, 'search_entities_by_similarity'):
            return name_scores, full_scores

        for idx in range(len(extracted_entities)):
            # Name-based search
            if name_embeddings is not None:
                query_emb = np.asarray(name_embeddings[idx] if name_embeddings.ndim == 1 or idx < len(name_embeddings) else None, dtype=np.float32)
                if query_emb is not None and query_emb.size > 0:
                    norm = np.linalg.norm(query_emb)
                    if norm > 0:
                        query_emb = query_emb / norm
                    name_scores[idx] = self._backend_vector_search(query_emb.tolist(), top_k)

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
                    full_scores[idx] = self._backend_vector_search(query_emb.tolist(), top_k)

        return name_scores, full_scores

    def _backend_vector_search(self, query_vector: List[float], top_k: int) -> Dict[str, float]:
        """Execute a vector search using the storage backend, returning {family_id: score}."""
        results = {}
        try:
            if hasattr(self.storage, 'search_entities_by_similarity'):
                # SQLite or other backends with a native similarity search method
                hits = self.storage.search_entities_by_similarity(query_vector, limit=top_k)
                for hit in hits:
                    fid = hit.family_id if hasattr(hit, 'family_id') else hit.get('family_id')
                    score = hit.score if hasattr(hit, 'score') else hit.get('score', 0.0)
                    if fid:
                        results[fid] = float(score)
        except Exception as e:
            logger.debug("Vector search in alignment failed: %s", e)
        return results

    def _compute_sim_matrix(self, query_embeddings, stored_emb_matrix, stored_dim, label):
        """Compute normalized cosine similarity matrix between query and stored embeddings."""
        if query_embeddings is None:
            return None
        query_mat = np.array(query_embeddings, dtype=np.float32)
        if query_mat.ndim == 1:
            query_mat = query_mat.reshape(1, -1)
        if query_mat.shape[1] == 0 or query_mat.shape[1] != stored_dim:
            logger.warning(
                "entity alignment: %s embedding dim mismatch (query=%d, stored=%d)",
                label, query_mat.shape[1], stored_dim,
            )
            return None
        query_norms = np.linalg.norm(query_mat, axis=1, keepdims=True)
        query_norms = np.where(query_norms == 0, 1.0, query_norms)
        query_mat = query_mat / query_norms
        return query_mat @ stored_emb_matrix.T

    # ------------------------------------------------------------------
    # Internal: per-entity row building
    # ------------------------------------------------------------------

    def _build_rows_for_entity(
        self, idx, extracted_entity, projections,
        name_emb_scores: Dict[str, float], full_emb_scores: Dict[str, float],
        jaccard_threshold, embedding_name_threshold, embedding_full_threshold,
        ext_name_bigrams, ext_core_bigrams, proj_name_bigrams, proj_core_bigrams,
        ext_core_name: str = "",
    ) -> List[Dict[str, Any]]:
        """Build candidate rows for a single extracted entity."""
        candidate_rows: List[Dict[str, Any]] = []
        ext_name = extracted_entity["name"]
        ext_core = ext_core_name or normalize_entity_name_for_matching(ext_name)

        for j, projection in enumerate(projections):
            lexical_score = _jaccard_from_bigrams(ext_name_bigrams, proj_name_bigrams[j])
            proj_core = projection["_core_name"]
            core_score = 0.0
            name_match_type = "none"

            # Substring detection — cache lengths to avoid repeated len() calls
            if ext_core and proj_core:
                ext_cl = len(ext_core)
                proj_cl = len(proj_core)
                if ext_cl >= 2 and proj_cl >= 2:
                    if ext_core in proj_core or proj_core in ext_core:
                        if ext_cl <= proj_cl:
                            ratio = ext_cl / proj_cl
                        else:
                            ratio = proj_cl / ext_cl
                        substring_score = 0.65 + ratio * 0.30
                        core_score = max(core_score, min(substring_score, 0.95))
                        name_match_type = "substring"
                elif ext_cl == 1 and proj_cl >= 2 and ext_core in proj_core:
                    # Single-char core name (e.g., "张" from "张教授"): allow
                    # substring match with penalty (higher false-positive risk).
                    # Score intentionally above jaccard_threshold so the candidate
                    # is generated for LLM-based final decision.
                    ratio = ext_cl / proj_cl
                    substring_score = 0.60 + ratio * 0.15
                    core_score = max(core_score, min(substring_score, 0.75))
                    name_match_type = "substring"

            # Exact core-name match
            if ext_core and proj_core and ext_core == proj_core:
                core_score = max(core_score, 0.85)
                if name_match_type == "none":
                    name_match_type = "exact"

            # Jaccard fallback
            if core_score == 0 and lexical_score < jaccard_threshold:
                core_score = _jaccard_from_bigrams(ext_core_bigrams, proj_core_bigrams[j])

            lexical_score = max(lexical_score, core_score)

            fid = projection["family_id"]
            dense_name_score = name_emb_scores.get(fid, 0.0)
            dense_full_score = full_emb_scores.get(fid, 0.0)

            if (
                lexical_score >= jaccard_threshold
                or dense_name_score >= embedding_name_threshold
                or dense_full_score >= embedding_full_threshold
            ):
                best_dense = max(dense_name_score, dense_full_score)
                core_name_match = (
                    ext_core
                    and proj_core == ext_core
                )
                candidate_rows.append({
                    "family_id": projection["family_id"],
                    "name": projection["name"],
                    "content": projection["content"],
                    "source_document": projection["entity"].source_document if projection.get("entity") else "",
                    "version_count": projection["version_count"],
                    "entity": projection.get("entity"),
                    "lexical_score": lexical_score,
                    "dense_score": best_dense,
                    "combined_score": max(lexical_score, dense_name_score, dense_full_score),
                    "merge_safe": core_name_match or (best_dense >= self.merge_safe_embedding_threshold and lexical_score >= self.merge_safe_jaccard_threshold),
                    "name_match_type": name_match_type,
                })

        return candidate_rows

    # ------------------------------------------------------------------
    # Supplement: BM25 concept search
    # ------------------------------------------------------------------

    def _supplement_candidates_from_concepts(
        self,
        candidate_table: Dict[int, List[Dict[str, Any]]],
        extracted_entities: List[Dict[str, str]],
        jaccard_threshold: float,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Supplement candidate table with BM25 matches from the unified concepts table."""
        _t0 = time.monotonic()
        if not extracted_entities:
            return candidate_table

        name_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, ee in enumerate(extracted_entities):
            name = ee.get("name", "").strip()
            if name:
                name_to_indices[name].append(idx)
        if not name_to_indices:
            return candidate_table

        existing_fids_per_idx: Dict[int, set] = {}
        for idx in range(len(extracted_entities)):
            existing_fids_per_idx[idx] = {
                c["family_id"] for c in (candidate_table.get(idx) or ())
            }

        new_candidates_by_idx: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        all_new_fids: set = set()

        # Parallel BM25 concept search — one per unique name
        _bm25_items = list(name_to_indices.items())

        def _search_concept(name):
            try:
                return (name, self.storage.search_concepts_by_bm25(name, role="entity", limit=5))
            except Exception as exc:
                logger.debug("concept BM25 supplement failed for '%s': %s", name, exc)
                return (name, [])

        if len(_bm25_items) > 1:
            _pool = _get_bm25_pool(min(len(_bm25_items), BM25_POOL_MAX))
            _bm25_results = list(_pool.map(lambda item: _search_concept(item[0]), _bm25_items))
        else:
            _bm25_results = [_search_concept(_bm25_items[0][0])] if _bm25_items else []

        for name, bm25_results in _bm25_results:
            indices = name_to_indices.get(name, [])
            for concept in bm25_results:
                concept_fid = concept.get("family_id", "")
                concept_name = concept.get("name", "")
                if not concept_fid or not concept_name:
                    continue
                jaccard = self._calculate_jaccard_similarity(name, concept_name)
                if jaccard < jaccard_threshold:
                    continue
                for idx in indices:
                    if concept_fid in existing_fids_per_idx.get(idx, set()):
                        continue
                    new_candidates_by_idx[idx].append({
                        "family_id": concept_fid,
                        "name": concept_name,
                        "jaccard_score": jaccard,
                    })
                    all_new_fids.add(concept_fid)
                    if idx not in existing_fids_per_idx:
                        existing_fids_per_idx[idx] = set()
                    existing_fids_per_idx[idx].add(concept_fid)

        if not all_new_fids:
            return candidate_table

        _t_bm25 = time.monotonic()
        wprint_info(f"[candidate_timing] BM25 search: {_t_bm25 - _t0:.3f}s ({len(name_to_indices)} names)")

        fid_list = list(all_new_fids)
        entity_map = self.storage.get_entities_by_family_ids(fid_list)
        version_counts = self.storage.get_entity_version_counts(fid_list)

        for idx, raw_candidates in new_candidates_by_idx.items():
            rows = candidate_table.get(idx) or []
            for rc in raw_candidates:
                fid = rc["family_id"]
                entity_obj = entity_map.get(fid)
                rows.append({
                    "family_id": fid,
                    "name": rc["name"],
                    "content": entity_obj.content if entity_obj else (rc.get("name", "")),
                    "source_document": entity_obj.source_document if entity_obj else "",
                    "version_count": version_counts.get(fid, 1),
                    "entity": entity_obj,
                    "lexical_score": rc["jaccard_score"],
                    "dense_score": 0.0,
                    "combined_score": rc["jaccard_score"],
                    "merge_safe": False,
                })
            rows.sort(key=lambda r: r["combined_score"], reverse=True)
            limit = self.max_alignment_candidates or self.max_similar_entities
            candidate_table[idx] = rows[:limit]

        _t_fetch = time.monotonic()
        wprint_info(f"[candidate_timing] BM25 entity fetch: {_t_fetch - _t_bm25:.3f}s ({len(fid_list)} fids)")

        return candidate_table
