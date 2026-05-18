"""
Entity candidate enrichment mixin — supplement, cross-check, neighbor expansion, and logging.

Extracted from entity_candidates.py to keep file sizes manageable.
"""
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from core.utils import wprint_info
from ._shared import normalize_entity_name_for_matching, _get_supp_pool

logger = logging.getLogger(__name__)


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
