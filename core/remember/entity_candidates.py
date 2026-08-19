"""
Entity candidate generation — simplified embedding-first approach.

Retrieval strategy:
1. Neo4j vector index top-K search (primary)
2. Exact name dict lookup from projections (supplement)

That's it. No Jaccard matrix, BM25, content-mention, neighbor expansion, etc.
"""
import logging
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.models import Entity
from core.storage.sqlite.manager import SQLiteGraphStorageManager
from core.llm.client import LLMClient
from core.debug_log import log_struct as _dbg_struct
from core.utils import wprint_info, _bigrams, _jaccard_from_bigrams, calculate_jaccard_similarity, cosine_similarity
from .helpers import _PAREN_ANNOTATION_RE
from ._shared import (
    normalize_entity_name_for_matching,
    _get_bm25_pool, BM25_POOL_MAX,
    _get_supp_pool,
    _TITLE_SUFFIXES_RE,
    _doc_basename,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate table builder
# ---------------------------------------------------------------------------

class _EnrichMixin:
    """Mixin providing candidate enrichment, supplementation, and logging methods."""

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
                        existing = candidate_table.get(tgt_idx) or []
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
                        "content": (neighbor_ent.content or "")[:200],
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

        if not hasattr(self.storage, 'search_entities_by_similarity'):
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
                    full_scores[idx] = self._backend_vector_search(query_emb.tolist(), top_k)

        return name_scores, full_scores

    def _backend_vector_search(self, query_vector: List[float], top_k: int) -> Dict[str, float]:
        """Execute a vector search using the storage backend, returning {family_id: score}."""
        results = {}
        try:
            if hasattr(self.storage, 'search_entities_by_similarity'):
                # SQLite or other backends with a native similarity search method
                # search_entities_by_similarity expects query_text (str), not a vector.
                # Skip this backend path when we only have a precomputed vector.
                hits = []
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


def _search_entity_candidates(
    storage: SQLiteGraphStorageManager,
    llm_client: LLMClient,
    max_similar_entities: int,
    entity_tree_log: bool,
    entity_name: str,
    entity_content: str,
    similarity_threshold: float,
    jaccard_search_threshold: Optional[float] = None,
    embedding_name_search_threshold: Optional[float] = None,
    embedding_full_search_threshold: Optional[float] = None,
    extracted_entity_names: Optional[set] = None,
    extracted_relation_pairs: Optional[set] = None,
) -> List[Entity]:
    """混合搜索候选实体：Jaccard + Embedding（name / name+content），去重合并后返回。

    3-4 个搜索查询并行执行，结果去重后返回。
    """
    from core.remember._shared import _get_entity_pool, _ENTITY_POOL_MAX

    jaccard_threshold = jaccard_search_threshold if jaccard_search_threshold is not None else min(similarity_threshold, 0.6)
    embedding_name_threshold = embedding_name_search_threshold if embedding_name_search_threshold is not None else min(similarity_threshold, 0.6)
    embedding_full_threshold = embedding_full_search_threshold if embedding_full_search_threshold is not None else min(similarity_threshold, 0.6)

    snippet_len = llm_client.effective_entity_snippet_length()

    # Build search tasks — all independent, can run in parallel
    def _search_jaccard():
        return storage.search_entities_by_similarity(
            entity_name, query_content=None, threshold=jaccard_threshold,
            max_results=max_similar_entities,
            content_snippet_length=snippet_len,
            text_mode="name_only", similarity_method="jaccard"
        )

    # 补充搜索：去称谓核心名称
    _core_name = _TITLE_SUFFIXES_RE.sub('', entity_name).strip()
    _has_title_suffix = _core_name != entity_name and len(_core_name) >= 2

    def _search_core_jaccard():
        return storage.search_entities_by_similarity(
            _core_name, query_content=None, threshold=jaccard_threshold,
            max_results=max_similar_entities,
            content_snippet_length=snippet_len,
            text_mode="name_only", similarity_method="jaccard"
        )

    def _search_name_embedding():
        return storage.search_entities_by_similarity(
            entity_name, query_content=None, threshold=embedding_name_threshold,
            max_results=max_similar_entities,
            content_snippet_length=snippet_len,
            text_mode="name_only", similarity_method="embedding"
        )

    def _search_full_embedding():
        return storage.search_entities_by_similarity(
            entity_name, query_content=entity_content, threshold=embedding_full_threshold,
            max_results=max_similar_entities,
            content_snippet_length=snippet_len,
            text_mode="name_and_content", similarity_method="embedding"
        )

    # Execute searches in parallel
    search_fns = [_search_jaccard, _search_name_embedding, _search_full_embedding]
    if _has_title_suffix:
        search_fns.append(_search_core_jaccard)

    if len(search_fns) > 1 and _ENTITY_POOL_MAX[0] > 1:
        pool = _get_entity_pool(min(len(search_fns), _ENTITY_POOL_MAX[0]))
        futures = [pool.submit(fn) for fn in search_fns]
        search_results = [fut.result() for fut in futures]
    else:
        search_results = [fn() for fn in search_fns]

    # Unpack results (core_jaccard is last if present)
    candidates_jaccard = search_results[0]
    candidates_name_embedding = search_results[1]
    candidates_full_embedding = search_results[2]
    candidates_core_jaccard = search_results[3] if _has_title_suffix else []

    if entity_tree_log:
        wprint_info(f"  │  ├─ Jaccard搜索（name_only）: {len(candidates_jaccard)} 个")
        if _has_title_suffix:
            wprint_info(f"  │  ├─ 核心名称Jaccard搜索（{_core_name}）: {len(candidates_core_jaccard)} 个")
        wprint_info(f"  │  ├─ Embedding搜索（name_only）: {len(candidates_name_embedding)} 个")
        wprint_info(f"  │  ├─ Embedding搜索（name+content）: {len(candidates_full_embedding)} 个")

    # 按 family_id 去重，保留最新版本
    entity_dict: Dict[str, Entity] = {}
    all_candidates = candidates_jaccard + candidates_core_jaccard + candidates_name_embedding + candidates_full_embedding
    for entity in all_candidates:
        existing = entity_dict.get(entity.family_id)
        if existing is None or entity.processed_time > existing.processed_time:
            entity_dict[entity.family_id] = entity
    similar_entities = list(entity_dict.values())

    # 过滤：已在当前抽取列表且已有关系的候选跳过
    if extracted_entity_names and extracted_relation_pairs:
        similar_entities = _filter_candidates_by_existing_relations(
            similar_entities, entity_name,
            extracted_entity_names, extracted_relation_pairs,
            entity_tree_log=entity_tree_log,
        )

    return similar_entities


def _filter_candidates_by_existing_relations(
    candidates: List[Entity],
    entity_name: str,
    extracted_entity_names: set,
    extracted_relation_pairs: set,
    *,
    entity_tree_log: bool = False,
) -> List[Entity]:
    """过滤掉已有关系的候选实体（步骤3已处理）。"""
    # Pre-extract pair keys into a set for O(1) lookup (avoids O(C*R) any() scan)
    _pair_keys = {pair[0] for pair in extracted_relation_pairs} if extracted_relation_pairs else set()
    filtered = []
    skipped = 0
    for candidate in candidates:
        if candidate.name == entity_name:
            filtered.append(candidate)
        elif candidate.name not in extracted_entity_names:
            filtered.append(candidate)
        else:
            pair_key = (entity_name, candidate.name) if entity_name <= candidate.name else (candidate.name, entity_name)
            if pair_key in _pair_keys:
                skipped += 1
                if entity_tree_log:
                    wprint_info(f"  │  │  ├─ {candidate.name}: 跳过已有关系（步骤3已处理）")
            else:
                filtered.append(candidate)
    if entity_tree_log and skipped > 0:
        wprint_info(f"  │  跳过 {skipped} 个已在当前抽取列表且已存在关系的候选实体（步骤3已处理）")
    return filtered


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

    # Handle within-batch alias (__batch_ prefixed IDs)
    if match_existing_id.startswith("__batch_"):
        batch_name = top.get("name", "")
        if batch_name:
            # Resolve via entity_name_to_id dict (populated incrementally)
            resolved_id = (entity_name_to_id or {}).get(batch_name)
            if resolved_id:
                match_existing_id = resolved_id
            else:
                return None  # Not yet resolved, can't merge
        else:
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
