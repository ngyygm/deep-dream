"""
Entity candidate generation: building, supplementing, and enriching candidate tables
for entity alignment. Extracted from entity.py for maintainability.
"""
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
import threading

import numpy as np

from core.utils import wprint_info, calculate_jaccard_similarity, _bigrams, _jaccard_from_bigrams
from core.content_schema import ENTITY_SECTIONS
from .helpers import _PAREN_ANNOTATION_RE
from functools import lru_cache
import re

logger = logging.getLogger(__name__)

# Sentinel: reusable empty frozenset for bigram fallbacks (avoids per-call allocation)
_EMPTY_FROZENSET = frozenset()

# Shared pool for BM25 concept search (avoids per-call pool creation)
_BM25_POOL: ThreadPoolExecutor | None = None
_BM25_POOL_LOCK = threading.Lock()
_BM25_POOL_MAX = 4

def _get_bm25_pool(max_workers: int) -> ThreadPoolExecutor:
    global _BM25_POOL, _BM25_POOL_MAX
    with _BM25_POOL_LOCK:
        if _BM25_POOL is not None:
            if max_workers > _BM25_POOL_MAX:
                try: _BM25_POOL.shutdown(wait=False)
                except Exception: pass
                _BM25_POOL = None
            else:
                return _BM25_POOL
        _BM25_POOL_MAX = max(max_workers, _BM25_POOL_MAX)
        _BM25_POOL = ThreadPoolExecutor(max_workers=_BM25_POOL_MAX, thread_name_prefix="cand-bm25")
        return _BM25_POOL

# Shared pool for supplement phase (avoids per-window pool creation)
_SUPP_POOL: ThreadPoolExecutor | None = None
_SUPP_POOL_LOCK = threading.Lock()

def _get_supp_pool() -> ThreadPoolExecutor:
    global _SUPP_POOL
    with _SUPP_POOL_LOCK:
        if _SUPP_POOL is None:
            _SUPP_POOL = ThreadPoolExecutor(max_workers=3, thread_name_prefix="cand-supp")
        return _SUPP_POOL


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

_TITLE_SUFFIXES_RE = re.compile(
    r'(?:教授|博士|先生|女士|同学|老师|工程师|经理|总监|院长|所长|主任|校长|站长|馆长|主编|首席|总裁'
    r'|部长|省长|市长|县长|区长|镇长|村长|将军|上校|中校|少校|大校|司令|参谋|政委|舰长|机长)$'
)


@lru_cache(maxsize=4096)
def normalize_entity_name_for_matching(name: str) -> str:
    """去掉括号注释和称谓后缀，返回用于匹配的核心名称。"""
    core = _PAREN_ANNOTATION_RE.sub('', name).strip()
    core = _TITLE_SUFFIXES_RE.sub('', core).strip()
    return core


# ---------------------------------------------------------------------------
# Candidate table builder
# ---------------------------------------------------------------------------

class EntityCandidateBuilder:
    """Builds and enriches entity candidate tables for alignment.

    Responsibilities:
    - Build initial candidate table from projections (Jaccard + embedding)
    - Supplement with BM25 concept matches
    - Supplement by content-mention alias detection
    - Cross-check within-batch aliases
    - Enrich with graph neighborhood data
    - Expand via neighbor overlap
    - Enrich with source text snippets
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
        self.merge_safe_jaccard_threshold = merge_safe_jaccard_threshold
        self.verbose = verbose
        self.entity_progress_verbose = entity_progress_verbose

    def _entity_tree_log(self) -> bool:
        return self.verbose and self.entity_progress_verbose

    @staticmethod
    def _calculate_jaccard_similarity(text1: str, text2: str) -> float:
        return calculate_jaccard_similarity(text1, text2)

    def build_candidate_table(
        self,
        extracted_entities: List[Dict[str, str]],
        similarity_threshold: float,
        jaccard_search_threshold: Optional[float] = None,
        embedding_name_search_threshold: Optional[float] = None,
        embedding_full_search_threshold: Optional[float] = None,
        prefetched_embeddings: Optional[Tuple[Optional[Any], Optional[Any]]] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Build the full candidate table with all supplement/enrichment stages.

        This is the main entry point — replaces the old _build_entity_candidate_table.
        """
        projections = self.storage.get_latest_entities_projection(
            self.llm_client.effective_entity_snippet_length()
        )
        if not projections:
            wprint_info("[candidate_table] ⚠️ projections is EMPTY — no existing entities found for alignment")
            return {}
        else:
            proj_names = [p["name"] for p in projections[:10]]
            wprint_info(f"[candidate_table] projections: {len(projections)} existing entities. First 10: {proj_names}")

        jaccard_threshold = jaccard_search_threshold if jaccard_search_threshold is not None else min(similarity_threshold, 0.6)
        embedding_name_threshold = embedding_name_search_threshold if embedding_name_search_threshold is not None else min(similarity_threshold, 0.6)
        embedding_full_threshold = embedding_full_search_threshold if embedding_full_search_threshold is not None else min(similarity_threshold, 0.6)

        # Encode extracted entities
        name_embeddings: Optional[Any] = None
        full_embeddings: Optional[Any] = None
        if prefetched_embeddings is not None:
            name_embeddings, full_embeddings = prefetched_embeddings
        elif self.storage.embedding_client and self.storage.embedding_client.is_available():
            _N = len(extracted_entities)
            _snippet_len = self.llm_client.effective_entity_snippet_length()
            _name_texts = [entity["name"] for entity in extracted_entities]
            _full_texts = [
                f"{entity['name']} {entity['content'][:_snippet_len]}"
                for entity in extracted_entities
            ]
            _all_embs = self.storage.embedding_client.encode(_name_texts + _full_texts)
            name_embeddings = _all_embs[:_N]
            full_embeddings = _all_embs[_N:]

        # Vectorized similarity computation
        stored_emb_matrix = self._build_stored_embedding_matrix(projections)

        # Precompute similarity matrices
        name_sim_matrix = None
        full_sim_matrix = None
        if stored_emb_matrix is not None and stored_emb_matrix.shape[1] > 1:
            stored_dim = stored_emb_matrix.shape[1]
            name_sim_matrix = self._compute_sim_matrix(name_embeddings, stored_emb_matrix, stored_dim, "name")
            full_sim_matrix = self._compute_sim_matrix(full_embeddings, stored_emb_matrix, stored_dim, "full")

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
        candidate_table: Dict[int, List[Dict[str, Any]]] = {}
        for idx, extracted_entity in enumerate(extracted_entities):
            candidate_rows = self._build_rows_for_entity(
                idx, extracted_entity, projections,
                stored_emb_matrix, name_sim_matrix, full_sim_matrix,
                jaccard_threshold, embedding_name_threshold, embedding_full_threshold,
                ext_bigrams[idx], ext_core_bigrams[idx],
                proj_bigrams, proj_core_bigrams,
                ext_core_name=ext_core_names[idx],
            )
            candidate_rows.sort(key=lambda row: row["combined_score"], reverse=True)
            limit = self.max_alignment_candidates or self.max_similar_entities
            candidate_table[idx] = candidate_rows[:limit]

        # Phase 1: parallel supplement stages (1-3 only add candidates, no inter-dependencies)
        _pool = _get_supp_pool()
        _f1 = _pool.submit(
            self._supplement_candidates_from_concepts,
            {k: list(v) for k, v in candidate_table.items()},
            extracted_entities, jaccard_threshold,
        )
        _f2 = _pool.submit(
            self._supplement_candidates_by_content_mention,
            {k: list(v) for k, v in candidate_table.items()},
            extracted_entities, projections,
        )
        _f3 = _pool.submit(
            self._cross_check_within_batch,
            {k: list(v) for k, v in candidate_table.items()},
            extracted_entities,
        )
        _ct_1, _ct_2, _ct_3 = _f1.result(), _f2.result(), _f3.result()

        # Merge parallel results: dedup by family_id (first-seen wins)
        _limit = self.max_alignment_candidates or self.max_similar_entities
        for idx in range(len(extracted_entities)):
            _seen = set()
            _merged = []
            for src in (candidate_table, _ct_1, _ct_2, _ct_3):
                for c in src.get(idx) or ():
                    fid = c.get("family_id", "")
                    if fid not in _seen:
                        _merged.append(c)
                        _seen.add(fid)
            _merged.sort(key=lambda r: r["combined_score"], reverse=True)
            candidate_table[idx] = _merged[:_limit]

        # Phase 2: sequential enrichment (depends on merged phase-1 results)
        candidate_table, graph_context = self._enrich_candidates_with_neighbors(candidate_table)
        candidate_table = self._expand_candidates_via_neighbor_overlap(
            candidate_table, extracted_entities, graph_context=graph_context
        )
        candidate_table = self._enrich_candidates_with_source_text(candidate_table)

        # Debug logging
        if self.entity_progress_verbose:
            self._log_candidate_summary(candidate_table, extracted_entities, projections)

        return candidate_table

    # ------------------------------------------------------------------
    # Internal: embedding matrix
    # ------------------------------------------------------------------

    def _build_stored_embedding_matrix(self, projections: List[Dict]) -> Optional[np.ndarray]:
        """Build a normalized embedding matrix from stored projections.
        Single-pass: convert + find min-dim, then truncate + stack."""
        arrays: List[Optional[np.ndarray]] = []
        dim = None
        for p in projections:
            ea = p.get("embedding_array")
            if ea is not None:
                arr = np.asarray(ea, dtype=np.float32).ravel()
                arrays.append(arr)
                if dim is None or arr.shape[0] < dim:
                    dim = arr.shape[0]
            else:
                arrays.append(None)

        if dim is None or dim == 0:
            return None

        rows = [arr[:dim] if arr is not None else np.zeros(dim, dtype=np.float32) for arr in arrays]

        stored_emb_matrix = np.stack(rows)
        norms = np.linalg.norm(stored_emb_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return stored_emb_matrix / norms

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
        stored_emb_matrix, name_sim_matrix, full_sim_matrix,
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

            # Exact core-name match
            if ext_core and proj_core and ext_core == proj_core:
                core_score = max(core_score, 0.85)
                if name_match_type == "none":
                    name_match_type = "exact"

            # Jaccard fallback
            if core_score == 0 and lexical_score < jaccard_threshold:
                core_score = _jaccard_from_bigrams(ext_core_bigrams, proj_core_bigrams[j])

            lexical_score = max(lexical_score, core_score)

            dense_name_score = float(name_sim_matrix[idx, j]) if name_sim_matrix is not None else 0.0
            dense_full_score = float(full_sim_matrix[idx, j]) if full_sim_matrix is not None else 0.0

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
            _pool = _get_bm25_pool(min(len(_bm25_items), _BM25_POOL_MAX))
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

        return candidate_table

    # ------------------------------------------------------------------
    # Supplement: content-mention alias detection
    # ------------------------------------------------------------------

    def _supplement_candidates_by_content_mention(
        self,
        candidate_table: Dict[int, List[Dict[str, Any]]],
        extracted_entities: List[Dict[str, str]],
        projections: List[Dict[str, Any]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Supplement candidates by checking content mentions for alias detection."""
        if not extracted_entities or not projections:
            return candidate_table

        proj_name_to_idx: Dict[str, List[int]] = defaultdict(list)
        for i, p in enumerate(projections):
            pname = p.get("name", "")
            if pname and len(pname) >= 2:
                proj_name_to_idx[pname].append(i)

        existing_fids_per_idx: Dict[int, set] = {}
        for idx in range(len(extracted_entities)):
            existing_fids_per_idx[idx] = {
                c["family_id"] for c in (candidate_table.get(idx) or ())
            }

        # Pre-compute projection contents (avoid repeated dict.get in O(E*P) loop)
        _proj_contents = [p.get("content", "") for p in projections]

        # Pre-group projection names by first character for fast filtering
        _pname_by_first_char: Dict[str, List[tuple]] = defaultdict(list)
        for pname, proj_indices in proj_name_to_idx.items():
            if pname and len(pname) >= 2:
                fc = pname[0]
                _pname_by_first_char[fc].append((pname, proj_indices))

        _supplemented = 0
        for idx, ee in enumerate(extracted_entities):
            existing = candidate_table.get(idx) or ()
            if existing and any(c.get("merge_safe") and c.get("combined_score", 0) >= 0.7 for c in existing):
                continue

            ee_name = ee.get("name", "")
            ee_content = ee.get("content", "")
            if not ee_name or len(ee_name) < 2:
                continue

            new_candidates = []
            _existing_fids = existing_fids_per_idx.get(idx, set())

            # Phase 1: check if projection names appear in entity content
            if ee_content:
                # Only check names whose first char appears in content (cheap pre-filter)
                _content_chars = set(ee_content)
                for fc, entries in _pname_by_first_char.items():
                    if fc not in _content_chars:
                        continue
                    for pname, proj_indices in entries:
                        if pname in ee_content:
                            for pi in proj_indices:
                                proj = projections[pi]
                                fid = proj["family_id"]
                                if fid in _existing_fids:
                                    continue
                                new_candidates.append({
                                    "family_id": fid,
                                    "name": proj["name"],
                                    "content": _proj_contents[pi],
                                    "source_document": proj.get("entity").source_document if proj.get("entity") else "",
                                    "version_count": proj.get("version_count", 1),
                                    "entity": proj.get("entity"),
                                    "lexical_score": 0.0,
                                    "dense_score": 0.0,
                                    "combined_score": 0.3,
                                    "merge_safe": False,
                                    "name_match_type": "content_mention",
                                })
                                _existing_fids.add(fid)

            # Phase 2: check if entity name appears in projection content
            # Pre-filter: only check projections whose content contains first char of ee_name
            _ee_first_char = ee_name[0]
            _ee_len = len(ee_name)
            for pi, proj in enumerate(projections):
                proj_content = _proj_contents[pi]
                if not proj_content:
                    continue
                # Fast pre-filter: content shorter than name can't contain it
                if len(proj_content) < _ee_len:
                    continue
                # Fast pre-filter: first char must be in content
                if _ee_first_char not in proj_content:
                    continue
                fid = proj["family_id"]
                if fid in _existing_fids:
                    continue
                if ee_name in proj_content:
                    new_candidates.append({
                        "family_id": fid,
                        "name": proj["name"],
                        "content": proj_content,
                        "source_document": proj.get("entity").source_document if proj.get("entity") else "",
                        "version_count": proj.get("version_count", 1),
                        "entity": proj.get("entity"),
                        "lexical_score": 0.0,
                        "dense_score": 0.0,
                        "combined_score": 0.3,
                        "merge_safe": False,
                        "name_match_type": "content_mention",
                    })
                    _existing_fids.add(fid)

            if new_candidates:
                existing = candidate_table.get(idx) or []
                existing.extend(new_candidates)
                existing.sort(key=lambda r: r["combined_score"], reverse=True)
                candidate_table[idx] = existing
                _supplemented += 1

        if _supplemented > 0:
            wprint_info(f"[candidate_table] Content-mention alias supplement: {_supplemented} entities got new candidates")

        return candidate_table

    # ------------------------------------------------------------------
    # Supplement: within-batch alias cross-check
    # ------------------------------------------------------------------

    def _cross_check_within_batch(
        self,
        candidate_table: Dict[int, List[Dict[str, Any]]],
        extracted_entities: List[Dict[str, str]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Cross-check extracted entities within the same batch for alias pairs."""
        n = len(extracted_entities)
        if n < 2:
            return candidate_table

        # Pre-compute cores and names once (O(n) instead of O(n^2) normalization)
        _cores = [normalize_entity_name_for_matching(e["name"]) for e in extracted_entities]
        _names = [e["name"] for e in extracted_entities]

        _alias_pairs = 0
        for i in range(n):
            core_i = _cores[i]
            if not core_i or len(core_i) < 2:
                continue

            for j in range(i + 1, n):
                core_j = _cores[j]
                if not core_j or len(core_j) < 2:
                    continue

                is_alias = False
                if core_i in core_j or core_j in core_i:
                    is_alias = True
                elif len(core_i) >= 2 and len(core_j) >= 2:
                    jaccard = self._calculate_jaccard_similarity(core_i, core_j)
                    len_diff = abs(len(core_i) - len(core_j))
                    if jaccard >= 0.6 and len_diff <= 2:
                        is_alias = True

                if is_alias:
                    for src_idx, tgt_idx, src_name, tgt_name, tgt_core in [
                        (j, i, _names[j], _names[i], core_i),
                        (i, j, _names[i], _names[j], core_j),
                    ]:
                        existing = candidate_table.get(tgt_idx) or ()
                        already = any(
                            c.get("family_id") == f"__batch_{src_idx}"
                            for c in existing
                        )
                        if not already:
                            ratio = min(len(tgt_core), len(src_name)) / max(len(tgt_core), len(src_name))
                            synthetic_score = 0.65 + ratio * 0.30
                            existing.append({
                                "family_id": f"__batch_{src_idx}",
                                "name": src_name,
                                "content": extracted_entities[src_idx].get("content", ""),
                                "source_document": extracted_entities[src_idx].get("source_document", ""),
                                "version_count": 0,
                                "lexical_score": synthetic_score,
                                "dense_score": 0.0,
                                "combined_score": synthetic_score,
                                "merge_safe": True,
                                "name_match_type": "within_batch_alias",
                            })
                            candidate_table[tgt_idx] = existing
                            _alias_pairs += 1

        if _alias_pairs > 0:
            wprint_info(f"[candidate_table] Within-batch alias cross-check: {_alias_pairs} alias pairs found")

        return candidate_table

    # ------------------------------------------------------------------
    # Enrich: graph neighborhood
    # ------------------------------------------------------------------

    def _enrich_candidates_with_neighbors(
        self,
        candidate_table: Dict[int, List[Dict[str, Any]]],
    ) -> tuple:
        """Enrich candidates with graph neighborhood data for better alignment.

        Returns (candidate_table, graph_context) where graph_context is a dict
        of shared data that _expand_candidates_via_neighbor_overlap can reuse.
        """
        family_ids = set()
        for candidates in candidate_table.values():
            for c in candidates:
                fid = c.get("family_id", "")
                if fid and not fid.startswith("__batch_"):
                    family_ids.add(fid)
        if not family_ids:
            return candidate_table, {}

        # Batch fetch entities and relations in parallel (independent queries)
        fid_list = list(family_ids)
        fid_to_abs_ids: Dict[str, set] = defaultdict(set)
        all_relations = []

        def _fetch_entities():
            return self.storage.get_entities_by_family_ids(fid_list)

        def _fetch_relations():
            return self.storage.get_relations_by_family_ids(fid_list, limit=10)

        _ent_fut = _get_supp_pool().submit(_fetch_entities)
        _rel_fut = _get_supp_pool().submit(_fetch_relations)

        try:
            entity_map = _ent_fut.result()
            for fid, entity in entity_map.items():
                fid_to_abs_ids[fid].add(entity.absolute_id)
        except Exception:
            return candidate_table, {}
        if not fid_to_abs_ids:
            return candidate_table, {}

        try:
            all_relations = _rel_fut.result() or []
        except Exception:
            all_relations = []
        if not all_relations:
            return candidate_table, {}

        other_abs_ids = set()
        for rel in all_relations:
            other_abs_ids.add(rel.entity1_absolute_id)
            other_abs_ids.add(rel.entity2_absolute_id)

        # Batch fetch all neighbor entities by absolute_id (replaces N individual calls)
        abs_id_to_entity: Dict[str, Any] = {}
        try:
            neighbor_entities = self.storage.get_entities_by_absolute_ids(list(other_abs_ids))
            for ent in neighbor_entities:
                if ent:
                    abs_id_to_entity[ent.absolute_id] = ent
        except Exception:
            pass

        # Build reverse mapping: absolute_id -> family_id (O(total_abs_ids), one-time)
        abs_to_fid: Dict[str, str] = {}
        for fid, abs_ids in fid_to_abs_ids.items():
            for aid in abs_ids:
                abs_to_fid[aid] = fid

        fid_to_neighbors: Dict[str, List[Dict]] = defaultdict(list)
        fid_to_neighbor_ents: Dict[str, List[Any]] = defaultdict(list)
        for rel in all_relations:
            e1, e2 = rel.entity1_absolute_id, rel.entity2_absolute_id
            fid1 = abs_to_fid.get(e1)
            fid2 = abs_to_fid.get(e2)
            if fid1 and not fid2:
                # e1 is a candidate, e2 is the neighbor
                other_ent = abs_id_to_entity.get(e2)
                if other_ent:
                    fid_to_neighbor_ents[fid1].append(other_ent)
                    if other_ent.name:
                        fid_to_neighbors[fid1].append({
                            "name": other_ent.name,
                            "relation_summary": (rel.content or "")[:60],
                        })
            elif fid2 and not fid1:
                # e2 is a candidate, e1 is the neighbor
                other_ent = abs_id_to_entity.get(e1)
                if other_ent:
                    fid_to_neighbor_ents[fid2].append(other_ent)
                    if other_ent.name:
                        fid_to_neighbors[fid2].append({
                            "name": other_ent.name,
                            "relation_summary": (rel.content or "")[:60],
                        })

        _enriched = 0
        for candidates in candidate_table.values():
            for c in candidates:
                fid = c.get("family_id", "")
                if not fid or fid.startswith("__batch_"):
                    continue
                neighbors = fid_to_neighbors.get(fid) or ()
                if neighbors:
                    c["neighbors"] = neighbors[:5]
                    _enriched += 1

        if _enriched > 0:
            wprint_info(f"[candidate_table] Neighbor enrichment: {_enriched} candidates enriched with graph neighbors")

        # Build shared graph context for _expand_candidates_via_neighbor_overlap to reuse
        graph_context = {
            "abs_to_fid": abs_to_fid,
            "abs_id_to_entity": abs_id_to_entity,
            "fid_to_neighbor_ents": fid_to_neighbor_ents,
            "fid_to_abs_ids": fid_to_abs_ids,
        }

        return candidate_table, graph_context

    # ------------------------------------------------------------------
    # Expand: neighbor overlap
    # ------------------------------------------------------------------

    def _expand_candidates_via_neighbor_overlap(
        self,
        candidate_table: Dict[int, List[Dict[str, Any]]],
        extracted_entities: List[Dict[str, str]],
        graph_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Expand candidates by adding graph neighbors of existing candidates.

        When graph_context is provided (from _enrich_candidates_with_neighbors),
        reuses the pre-fetched data instead of making duplicate DB queries.
        """
        max_neighbor_expansion = 8
        neighbor_score = 0.25

        entities_needing_expansion: Dict[int, List[Dict[str, Any]]] = {}
        for idx, candidates in candidate_table.items():
            if not candidates:
                continue
            has_strong = any(
                c.get("merge_safe") and c.get("combined_score", 0) >= 0.7
                for c in candidates
            )
            if not has_strong:
                entities_needing_expansion[idx] = candidates

        if not entities_needing_expansion:
            return candidate_table

        # Reuse graph_context from enrich step if available
        if graph_context and graph_context.get("fid_to_neighbor_ents"):
            fid_to_neighbor_ents = graph_context["fid_to_neighbor_ents"]
        else:
            # Fallback: fetch data ourselves
            expansion_fids = set()
            for candidates in entities_needing_expansion.values():
                for c in candidates:
                    fid = c.get("family_id", "")
                    if fid and not fid.startswith("__batch_"):
                        expansion_fids.add(fid)
            if not expansion_fids:
                return candidate_table

            fid_to_abs_ids: Dict[str, set] = defaultdict(set)
            try:
                entity_map = self.storage.get_entities_by_family_ids(list(expansion_fids))
                for fid, entity in entity_map.items():
                    fid_to_abs_ids[fid].add(entity.absolute_id)
            except Exception:
                return candidate_table
            if not fid_to_abs_ids:
                return candidate_table

            try:
                all_relations = self.storage.get_relations_by_family_ids(
                    list(fid_to_abs_ids), limit=10
                )
            except Exception:
                return candidate_table
            if not all_relations:
                return candidate_table

            other_abs_ids = set()
            for rel in all_relations:
                other_abs_ids.add(rel.entity1_absolute_id)
                other_abs_ids.add(rel.entity2_absolute_id)

            abs_id_to_entity: Dict[str, Any] = {}
            try:
                neighbor_entities = self.storage.get_entities_by_absolute_ids(list(other_abs_ids))
                for ent in neighbor_entities:
                    if ent:
                        abs_id_to_entity[ent.absolute_id] = ent
            except Exception:
                pass

            abs_to_fid: Dict[str, str] = {}
            for fid, abs_ids in fid_to_abs_ids.items():
                for aid in abs_ids:
                    abs_to_fid[aid] = fid

            fid_to_neighbor_ents: Dict[str, List[Any]] = defaultdict(list)
            for rel in all_relations:
                e1, e2 = rel.entity1_absolute_id, rel.entity2_absolute_id
                fid1 = abs_to_fid.get(e1)
                fid2 = abs_to_fid.get(e2)
                if fid1 and not fid2:
                    neighbor_ent = abs_id_to_entity.get(e2)
                    if neighbor_ent:
                        fid_to_neighbor_ents[fid1].append(neighbor_ent)
                elif fid2 and not fid1:
                    neighbor_ent = abs_id_to_entity.get(e1)
                    if neighbor_ent:
                        fid_to_neighbor_ents[fid2].append(neighbor_ent)

        _expanded = 0
        # Pre-fetch version counts for all neighbor entities (batch)
        all_neighbor_fids = set()
        for ents in fid_to_neighbor_ents.values():
            for ent in ents:
                nfid = ent.family_id or ""
                if nfid:
                    all_neighbor_fids.add(nfid)
        version_counts_map: Dict[str, int] = {}
        if all_neighbor_fids:
            try:
                version_counts_map = self.storage.get_entity_version_counts(
                    list(all_neighbor_fids)
                )
            except Exception:
                pass

        for idx, candidates in entities_needing_expansion.items():
            existing_fids = {
                c.get("family_id", "")
                for c in candidates
                if c.get("family_id") and not c["family_id"].startswith("__batch_")
            }
            existing_names = {c.get("name", "") for c in candidates}
            new_candidates = []
            for c in candidates:
                fid = c.get("family_id", "")
                if not fid or fid.startswith("__batch_"):
                    continue
                neighbor_ents = fid_to_neighbor_ents.get(fid) or ()
                for neighbor_ent in neighbor_ents:
                    nfid = neighbor_ent.family_id or ""
                    nname = neighbor_ent.name or ""
                    if nfid and nfid in existing_fids:
                        continue
                    if nname and nname in existing_names:
                        continue

                    new_candidates.append({
                        "family_id": nfid,
                        "name": nname,
                        "content": (_nc := neighbor_ent.content or "")[:200] if len(_nc) > 200 else _nc,
                        "source_document": neighbor_ent.source_document or "",
                        "version_count": version_counts_map.get(nfid, 0),
                        "entity": neighbor_ent,
                        "lexical_score": 0.0,
                        "dense_score": 0.0,
                        "combined_score": neighbor_score,
                        "merge_safe": False,
                        "name_match_type": "neighbor_expansion",
                    })
                    existing_fids.add(nfid)
                    existing_names.add(nname)

            new_candidates = new_candidates[:max_neighbor_expansion]
            if new_candidates:
                candidate_table[idx].extend(new_candidates)
                _expanded += len(new_candidates)

        if _expanded > 0:
            wprint_info(f"[candidate_table] Neighbor expansion: {_expanded} new candidates added across {len(entities_needing_expansion)} entities")

        return candidate_table

    # ------------------------------------------------------------------
    # Enrich: source text snippets
    # ------------------------------------------------------------------

    def _enrich_candidates_with_source_text(
        self,
        candidate_table: Dict[int, List[Dict[str, Any]]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Enrich candidates with source text snippets from their origin Episodes."""
        SNIPPET_LENGTH = 200

        all_episode_ids: set = set()
        for candidates in candidate_table.values():
            for c in candidates:
                ent = c.get("entity")
                if ent:
                    eid = ent.episode_id or ""
                    if eid:
                        all_episode_ids.add(eid)
        if not all_episode_ids:
            return candidate_table

        try:
            snippets = self.storage.batch_get_source_text_snippets(
                list(all_episode_ids), snippet_length=SNIPPET_LENGTH
            )
        except Exception as e:
            logger.debug("Failed to batch-fetch episode source texts: %s", e)
            return candidate_table
        if not snippets:
            return candidate_table

        enriched = 0
        for candidates in candidate_table.values():
            for c in candidates:
                ent = c.get("entity")
                if ent:
                    eid = ent.episode_id or ""
                    if eid and eid in snippets:
                        c["source_text_snippet"] = snippets[eid]
                        enriched += 1

        if enriched > 0:
            wprint_info(f"[candidate_table] Source text enrichment: {enriched} candidates enriched")

        return candidate_table

    # ------------------------------------------------------------------
    # Debug logging
    # ------------------------------------------------------------------

    def _log_candidate_summary(self, candidate_table, extracted_entities, projections):
        _with_cands = 0
        for idx in range(len(extracted_entities)):
            _ename = extracted_entities[idx]["name"]
            _cands = candidate_table.get(idx) or ()
            if _cands:
                _with_cands += 1
            _has_substring = any(c.get("name_match_type") == "substring" for c in _cands)
            _is_short = 2 <= len(_ename) <= 3
            if (_is_short or _has_substring) and _cands:
                _cand_summary = [(c["name"], f"j{c['lexical_score']:.2f}/d{c['dense_score']:.2f}", c.get("name_match_type", "none"), f"safe={c.get('merge_safe', False)}") for c in _cands]
                wprint_info(f"[candidate_table] FULL '{_ename}' -> {_cand_summary}")
        if _with_cands > 0:
            _sample_idx = next(idx for idx in range(len(extracted_entities)) if candidate_table.get(idx))
            _sample_name = extracted_entities[_sample_idx]["name"]
            _sample_cands = [(c["name"], f"j{c['lexical_score']:.2f}/d{c['dense_score']:.2f}", c["name_match_type"]) for c in candidate_table[_sample_idx][:3]]
            wprint_info(f"[candidate_table] {_with_cands}/{len(extracted_entities)} entities have candidates. Sample: '{_sample_name}' -> {_sample_cands}")
        else:
            wprint_info(f"[candidate_table] ⚠️ NO candidates found for {len(extracted_entities)} extracted entities (vs {len(projections)} projections)")
            _ext_names = [e["name"] for e in extracted_entities[:10]]
            _proj_names = [p["name"] for p in projections[:10]]
            wprint_info(f"[candidate_table] extracted names: {_ext_names}")
            wprint_info(f"[candidate_table] projection names: {_proj_names}")
