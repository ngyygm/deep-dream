"""Entity name resolution sub-mixin for _PipelineExtractionMixin."""
from __future__ import annotations

from typing import Dict, List

from core.utils import wprint_info
from .helpers import (
    _core_entity_name,
)


class _ResolutionMixin:
    """Same-name conflict resolution, missing-name resolution, and name-to-ID conversion."""

    def _resolve_same_name_conflicts(self, entity_name_to_ids, verbose=False):
        """Detect and resolve same-name entity conflicts by merging into primary."""
        duplicate_names = {name: ids for name, ids in entity_name_to_ids.items() if len(ids) > 1}
        ambiguous_duplicate_names = set()

        if not duplicate_names:
            entity_name_to_id = {name: ids[0] for name, ids in entity_name_to_ids.items()}
            return entity_name_to_id, ambiguous_duplicate_names

        if verbose:
            wprint_info(f"【步骤9】警告｜同名｜{len(duplicate_names)}处")
            for name, ids in duplicate_names.items():
                wprint_info(
                    f"【步骤9】冲突｜详情｜{name} {len(ids)}id {ids[:3]}{'...' if len(ids) > 3 else ''}"
                )

        entity_name_to_id = {}
        # Batch-fetch version counts for all duplicate-name entities
        _all_dup_fids = [fid for ids in entity_name_to_ids.values() if len(ids) > 1 for fid in ids]
        _dup_vc_map = self.storage.get_entity_version_counts(_all_dup_fids) if _all_dup_fids else {}
        for name, ids in entity_name_to_ids.items():
            if len(ids) > 1:
                versions_map = {fid: _dup_vc_map.get(fid, 0) for fid in ids}

                # Same-name entities: always merge — name match is strong signal.
                primary_id = max(ids, key=lambda fid: versions_map.get(fid, 0))
                entity_name_to_id[name] = primary_id
                duplicate_pairs = [(fid, primary_id) for fid in ids if fid and fid != primary_id]
                if duplicate_pairs:
                    batch_fn = getattr(self.storage, 'register_entity_redirects_batch', None)
                    if batch_fn:
                        batch_fn(dict(duplicate_pairs))
                    else:
                        for fid, pid in duplicate_pairs:
                            self.storage.register_entity_redirect(fid, pid)
                if verbose:
                    wprint_info(
                        f"【步骤9】冲突｜主实体｜{name}->{primary_id} v{versions_map.get(primary_id, 0)}"
                    )
            else:
                entity_name_to_id[name] = ids[0]

        return entity_name_to_id, ambiguous_duplicate_names

    def _resolve_missing_relation_entity_names(self, pending_relations, entity_name_to_id,
                                                 ambiguous_duplicate_names):
        """Resolve entity names referenced in relations but missing from the name-to-id map.

        Runs 4 rounds: DB exact match → core-name fuzzy → case-insensitive → substring.
        Returns (entity_name_to_id, db_matched, fuzzy_matched).
        """
        _rel_entity_names = set()
        # _core_entity_name is already @lru_cache(maxsize=2048) — no local cache needed

        for rel_info in pending_relations:
            n1 = rel_info.get("entity1_name", "")
            n2 = rel_info.get("entity2_name", "")
            if n1:
                _rel_entity_names.add(n1)
            if n2:
                _rel_entity_names.add(n2)

        _missing_names = [n for n in _rel_entity_names
                          if n not in entity_name_to_id and n not in ambiguous_duplicate_names]
        _db_matched = 0
        _fuzzy_matched = 0

        # Rounds 1+2 merged: single DB query with both exact and core names
        if _missing_names:
            # Build combined name set: original names + core names for fuzzy match
            _core_name_map: Dict[str, str] = {}
            for name, eid in entity_name_to_id.items():
                core = _core_entity_name(name)
                if core and core not in _core_name_map:
                    _core_name_map[core] = eid

            _query_names = set(_missing_names)
            for missing_name in _missing_names:
                core_missing = _core_entity_name(missing_name)
                if core_missing and core_missing not in _core_name_map:
                    _query_names.add(core_missing)

            _db_map = self.storage.get_family_ids_by_names(list(_query_names))

            # Round 1: resolve exact matches
            for name in _missing_names:
                if name in _db_map and name not in entity_name_to_id:
                    entity_name_to_id[name] = _db_map[name]
                    _db_matched += 1

            # Round 2: resolve core-name fuzzy matches
            for core_name, eid in _db_map.items():
                if core_name not in _core_name_map:
                    _core_name_map[core_name] = eid

            for missing_name in _missing_names:
                if missing_name in entity_name_to_id:
                    continue
                core_missing = _core_entity_name(missing_name)
                if core_missing and core_missing in _core_name_map:
                    entity_name_to_id[missing_name] = _core_name_map[core_missing]
                    _fuzzy_matched += 1

        # Rounds 3+4: Build lookup structures once, then iterate remaining missing names once
        _still_missing = [n for n in _rel_entity_names if n not in entity_name_to_id]
        if _still_missing:
            # Round 3 structures: case-insensitive lookup
            _lower_map: Dict[str, str] = {}
            # Round 4 structures: core name + substring matching
            _known_cores = []
            _core_to_known: Dict[str, str] = {}
            for name, eid in entity_name_to_id.items():
                low = name.lower()
                if low not in _lower_map:
                    _lower_map[low] = eid
                core = _core_entity_name(name).lower()
                if core and len(core) >= 2:
                    _known_cores.append((name, core))
                    if core not in _core_to_known:
                        _core_to_known[core] = name

            for missing_name in _still_missing:
                # Round 3: case-insensitive
                low_missing = missing_name.lower()
                if low_missing in _lower_map:
                    entity_name_to_id[missing_name] = _lower_map[low_missing]
                    _fuzzy_matched += 1
                    continue
                # Round 4: substring fuzzy match
                core_miss = _core_entity_name(missing_name).lower()
                if not core_miss or len(core_miss) < 2:
                    continue
                if core_miss in _core_to_known:
                    entity_name_to_id[missing_name] = entity_name_to_id[_core_to_known[core_miss]]
                    _fuzzy_matched += 1
                    continue
                best_match = None
                best_len = 0
                for known, core_known in _known_cores:
                    if core_miss in core_known or core_known in core_miss:
                        match_len = min(len(core_miss), len(core_known))
                        if match_len > best_len:
                            best_len = match_len
                            best_match = known
                if best_match:
                    entity_name_to_id[missing_name] = entity_name_to_id[best_match]
                    _fuzzy_matched += 1

        return entity_name_to_id, _db_matched, _fuzzy_matched

    def _convert_pending_relations_to_ids(self, pending_relations, entity_name_to_id,
                                           verbose=False):
        """Convert relation endpoint names to family_ids. Returns (updated_relations, skipped, self_rels)."""
        updated_pending_relations = []
        _skipped_relations = []
        _self_relations = 0
        for rel_info in pending_relations:
            entity1_name = rel_info.get("entity1_name", "")
            entity2_name = rel_info.get("entity2_name", "")
            content = rel_info.get("content", "")
            relation_type = rel_info.get("relation_type", "normal")

            entity1_id = entity_name_to_id.get(entity1_name)
            entity2_id = entity_name_to_id.get(entity2_name)

            if entity1_id and entity2_id:
                if entity1_id == entity2_id:
                    _self_relations += 1
                    continue
                updated_pending_relations.append({
                    "entity1_id": entity1_id,
                    "entity2_id": entity2_id,
                    "entity1_name": entity1_name,
                    "entity2_name": entity2_name,
                    "content": content,
                    "relation_type": relation_type
                })
            else:
                _reason = []
                if not entity1_id:
                    _reason.append(f"entity1='{entity1_name}'")
                if not entity2_id:
                    _reason.append(f"entity2='{entity2_name}'")
                _skipped_relations.append(f"  {entity1_name} <-> {entity2_name} (无法解析: {', '.join(_reason)})")

        return updated_pending_relations, _skipped_relations, _self_relations
