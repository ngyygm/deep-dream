"""
Entity candidate generation — simplified embedding-first approach.

Retrieval strategy:
1. Neo4j vector index top-K search (primary)
2. Exact name dict lookup from projections (supplement)

That's it. No Jaccard matrix, BM25, content-mention, neighbor expansion, etc.
"""
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.storage.sqlite.manager import SQLiteGraphStorageManager
from core.llm.client import LLMClient
from core.debug_log import log_struct as _dbg_struct
from core.utils import wprint_info, _bigrams, calculate_jaccard_similarity, cosine_similarity
from ._shared import normalize_entity_name_for_matching

logger = logging.getLogger(__name__)


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
                f"# {e['name']}\n{e.get('content', '')[:_snippet_len]}"
                for e in extracted_entities
            ]
            _all_embs = self.storage.embedding_client.encode(_name_texts + _full_texts)
            name_embeddings = _all_embs[:_N]
            full_embeddings = _all_embs[_N:]

        _t_encode = time.monotonic()
        wprint_info(f"[candidate_timing] projections + encode: {_t_encode - _t0:.3f}s")

        # Vectorized similarity via graph-local embedding matrix. Keep the
        # retrieval width bounded; exact/core-name matches are added separately.
        top_k = max(self.max_alignment_candidates or self.max_similar_entities, len(projections), 10)
        name_emb_scores, full_emb_scores = self._search_embedding_top_k(
            extracted_entities, name_embeddings, full_embeddings, top_k,
        )

        _t_vec = time.monotonic()
        wprint_info(f"[candidate_timing] embedding vector top-K search: {_t_vec - _t_encode:.3f}s")

        # Pre-compute core names for all projections (avoids E × P calls to normalize function)
        for p in projections:
            p["_core_name"] = normalize_entity_name_for_matching(p["name"])

        # Pre-compute core names + bigram sets for all extracted entities (avoids E × P recomputation)
        _empty_fs = frozenset()
        ext_bigrams = []
        ext_core_bigrams = []
        ext_core_names: List[str] = []
        for ee in extracted_entities:
            _n = ee["name"]
            ext_bigrams.append(_bigrams(_n.lower().strip()) if _n else _empty_fs)
            _c = normalize_entity_name_for_matching(_n)
            ext_core_names.append(_c)
            ext_core_bigrams.append(_bigrams(_c.lower().strip()) if _c else _empty_fs)
        proj_bigrams = []
        proj_core_bigrams = []
        for p in projections:
            _n = p["name"]
            proj_bigrams.append(_bigrams(_n.lower().strip()) if _n else _empty_fs)
            proj_core_bigrams.append(_bigrams(p["_core_name"].lower().strip()) if p["_core_name"] else _empty_fs)

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

        cache_fn = getattr(self.storage, "_vector_cache_for_role", None)
        if cache_fn:
            try:
                cache = cache_fn("entity")
                matrix = cache.get("matrix")
                rows = cache.get("rows") or []
                if matrix is not None and rows:
                    fid_by_row = [row.get("family_id") for row in rows]

                    def _score_queries(query_embeddings) -> Dict[int, Dict[str, float]]:
                        out: Dict[int, Dict[str, float]] = {}
                        if query_embeddings is None:
                            return out
                        qmat = np.asarray(query_embeddings, dtype=np.float32)
                        if qmat.ndim == 1:
                            qmat = qmat.reshape(1, -1)
                        if qmat.size == 0 or qmat.shape[1] != matrix.shape[1]:
                            return out
                        norms = np.linalg.norm(qmat, axis=1, keepdims=True)
                        norms = np.where(norms == 0, 1.0, norms)
                        qmat = qmat / norms
                        scores = qmat @ matrix.T
                        k = min(max(1, int(top_k or 10)), scores.shape[1])
                        for idx in range(min(len(extracted_entities), scores.shape[0])):
                            row_scores = scores[idx]
                            if row_scores.size <= k:
                                candidate_idx = np.arange(row_scores.size)
                            else:
                                candidate_idx = np.argpartition(row_scores, -k)[-k:]
                            ordered = candidate_idx[np.argsort(row_scores[candidate_idx])[::-1]]
                            out[idx] = {
                                fid_by_row[int(j)]: float(row_scores[int(j)])
                                for j in ordered
                                if fid_by_row[int(j)]
                            }
                        return out

                    return _score_queries(name_embeddings), _score_queries(full_embeddings)
            except Exception as e:
                logger.debug("Vector cache search in alignment failed: %s", e)

        return name_scores, full_scores

# ---------------------------------------------------------------------------
# Search, filtering, and alignment guard helpers (moved from entity_search.py)
# ---------------------------------------------------------------------------

def _calculate_jaccard_similarity(text1: str, text2: str) -> float:
    return calculate_jaccard_similarity(text1, text2)


def _cosine_similarity(embedding1, embedding2) -> float:
    return cosine_similarity(embedding1, embedding2)


def _alignment_guard(
    llm_client: LLMClient,
    alignment_guard_cache: OrderedDict,
    name_a: str, content_a: str, name_b: str, content_b: str,
    *, name_match_type: str = "none", require_content: bool = True,
) -> Optional[Tuple[str, float]]:
    """Three-way alignment check. Returns (verdict, confidence) if reject, None if same (proceed)."""
    if not hasattr(llm_client, 'judge_entity_alignment'):
        return None
    if require_content and not content_b:
        return None
    # Trivial content_b (e.g. "是", "no") carries no alignment signal — skip LLM call
    if content_b is not None and len(content_b) < 3 and not require_content:
        return ("different", 0.9)
    # Check instance cache (keyed by name + content prefix for bounded size)
    _ca = content_a or ""
    _cb = content_b or ""
    _cache_key = (name_a, _ca[:200] if len(_ca) > 200 else _ca, name_b, _cb[:200] if len(_cb) > 200 else _cb)
    if _cache_key in alignment_guard_cache:
        return alignment_guard_cache[_cache_key]
    result = llm_client.judge_entity_alignment(
        name_a, content_a, name_b, content_b, name_match_type=name_match_type,
    )
    verdict = result.get("verdict", "uncertain")
    confidence = result.get("confidence", 0.5)
    _dbg_struct("alignment_guard",
                name_a=name_a, name_b=name_b,
                content_a_snippet=(content_a or "")[:80],
                content_b_snippet=(content_b or "")[:80],
                verdict=verdict, confidence=f"{confidence:.2f}",
                name_match_type=name_match_type)
    if verdict in ("different", "uncertain"):
        ans = (verdict, confidence)
    else:
        ans = None
    # LRU eviction: remove oldest entry when cache exceeds limit
    if len(alignment_guard_cache) > 500:
        alignment_guard_cache.popitem(last=False)
    alignment_guard_cache[_cache_key] = ans
    alignment_guard_cache.move_to_end(_cache_key)
    return ans


def _try_context_alias_merge(
    storage: SQLiteGraphStorageManager,
    llm_client: LLMClient,
    alignment_guard_cache: OrderedDict,
    merge_two_contents_fn,  # callable: (old_entity, entity_name, entity_content, source_document, episode_id, base_time) -> str
    build_entity_version_fn,  # callable: same signature as _build_entity_version
    mark_versioned_fn,  # callable: (family_id, already_versioned, lock)
    entity_tree_log: bool,
    entity_name: str,
    entity_content: str,
    candidates: List[Dict[str, Any]],
    context_text: Optional[str],
    episode_id: str,
    source_document: str,
    base_time: Optional[Any],
    already_versioned_family_ids: Optional[set],
    _version_lock: Optional[Any],
    entity_name_to_id: Optional[Dict[str, str]] = None,
) -> Optional[Tuple]:
    """Check if top candidate is an alias and merge after LLM verification.

    Gate: EITHER name Jaccard >= 0.3 OR embedding(name+content) >= 0.5.
    Then checks content-mention alias evidence, and finally verifies with
    _alignment_guard before merging.

    Returns a result tuple if alias verified, None otherwise.
    """
    if not candidates or not context_text:
        return None

    top = candidates[0]
    cand_name = top.get("name", "")
    cand_content = top.get("content", "")

    # Skip if exact name match (already handled by fast path above)
    if cand_name == entity_name:
        return None

    # Gate: name Jaccard OR embedding similarity must pass threshold.
    # Either signal independently justifies trying LLM verification.
    _name_jaccard = _calculate_jaccard_similarity(entity_name, cand_name)
    _dense_score = top.get("dense_score", 0)
    _lexical_score = top.get("lexical_score", 0)

    _jaccard_ok = _name_jaccard >= 0.3
    _embedding_ok = _dense_score >= 0.5

    if not _jaccard_ok and not _embedding_ok:
        return None

    # Check alias evidence
    is_alias = False
    alias_reason = ""

    # Check 1: Candidate content mentions the extracted name
    # e.g., 刘备 content: "刘备,字玄德" → mentions "玄德"
    if entity_name in cand_content and len(entity_name) >= 2:
        is_alias = True
        alias_reason = f"候选内容提及'{entity_name}'"

    # Check 2: Extracted content mentions the candidate name
    if not is_alias and cand_name in entity_content and len(cand_name) >= 2:
        is_alias = True
        alias_reason = f"当前内容提及'{cand_name}'"

    if not is_alias:
        return None

    # Alias evidence found — verify with _alignment_guard before committing.
    # Content-mention alone is insufficient: "打听" appearing as a verb in
    # "周瑞家的" content is not alias evidence.
    _guard = _alignment_guard(
        llm_client, alignment_guard_cache,
        entity_name, entity_content, cand_name, cand_content or "",
        name_match_type=top.get("name_match_type", "none"),
    )
    if _guard:
        _guard_verdict, _guard_conf = _guard
        _dbg_struct("alias_merge_guard_reject",
                    entity_name=entity_name, cand_name=cand_name,
                    alias_reason=alias_reason,
                    name_jaccard=f"{_name_jaccard:.3f}",
                    dense_score=f"{_dense_score:.3f}",
                    guard_verdict=_guard_verdict, guard_conf=f"{_guard_conf:.2f}")
        if entity_tree_log:
            wprint_info(f"  │  别名合并被 guard 拒绝: '{entity_name}' ≁ '{cand_name}' (verdict={_guard_verdict}, conf={_guard_conf:.2f})")
        return None

    # Alias verified by guard — proceed with merge.
    _combined = top.get("combined_score", 0)
    match_existing_id = top.get("family_id", "")
    if not match_existing_id:
        return None

    latest_entity = top.get("entity") or storage.get_entity_by_family_id(match_existing_id)
    if not latest_entity:
        return None

    if entity_tree_log:
        wprint_info(f"  │  别名合并: '{entity_name}' = '{cand_name}' ({alias_reason}, jaccard={_name_jaccard:.2f}, emb={_dense_score:.2f}, guard=passed)")

    # Use the longer/more standard name as the merged name
    merged_name = cand_name  # Default: keep existing entity's name
    # Heuristic: if the existing entity's name is a full name and the new one is an alias, keep full name
    if len(entity_name) > len(cand_name):
        merged_name = entity_name
    # If the candidate's content explicitly states the entity's name as an alias
    # (e.g., "刘备,字玄德"), keep the first name (the actual name)
    if cand_content and entity_name in cand_content:
        # The candidate is likely the full-name entity, keep its name
        merged_name = cand_name

    # Prevent same-window duplicate versioning
    if already_versioned_family_ids and latest_entity.family_id in already_versioned_family_ids:
        if entity_tree_log:
            wprint_info(f"  │  别名合并: 同窗口复用 {latest_entity.family_id}")
        return latest_entity, [], {
            entity_name: latest_entity.family_id,
            latest_entity.name: latest_entity.family_id,
        }, None

    # Merge content (fast-forward)
    merged_content = merge_two_contents_fn(
        latest_entity, entity_name, entity_content,
        source_document, episode_id, base_time=base_time,
    )

    entity_version = build_entity_version_fn(
        latest_entity.family_id,
        merged_name,
        merged_content,
        episode_id,
        source_document,
        base_time=base_time,
        old_content=latest_entity.content or "",
        old_content_format=latest_entity.content_format or "plain",
    )
    mark_versioned_fn(latest_entity.family_id, already_versioned_family_ids, _version_lock)

    if entity_tree_log:
        wprint_info(f"  │  别名合并: '{entity_name}' → {latest_entity.family_id} (merged_name='{merged_name}')")

    return entity_version, [], {
        entity_name: latest_entity.family_id,
        merged_name: latest_entity.family_id,
    }, entity_version
