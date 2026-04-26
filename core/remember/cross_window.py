"""
Cross-window deduplication — post-processing after all windows complete.

Handles same-name dedup, content similarity checks, and name-alias dedup
across windows within a single remember call.
"""

import re as _re
import uuid as _uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional

from core.models import Entity
from core.utils import wprint_info, cosine_similarity
from .entity_candidates import _TITLE_SUFFIXES_RE
from .helpers import _PAREN_ANNOTATION_RE


class _CrossWindowDedupMixin:
    """Cross-window dedup methods. Expects `self.storage` and `self.llm_client`."""

    def _cross_window_dedup(self, align_results, verbose=True):
        """After all windows complete, find and merge same-name entities with different family_ids.

        Uses embedding similarity to distinguish genuine duplicates from same-name-different-meaning entities.
        Only merges when embedding similarity is above threshold (default 0.75).
        """
        # Collect all unique entities across windows
        all_entities = []
        for ar in align_results:
            if ar is None:
                continue
            all_entities.extend(ar.unique_entities)

        # Group by name
        name_to_entities = defaultdict(list)
        for entity in all_entities:
            name_to_entities[entity.name.strip()].append(entity)

        # Find same-name duplicates with different family_ids
        dupes = {name: ents for name, ents in name_to_entities.items()
                 if len(set(e.family_id for e in ents)) > 1}

        if not dupes:
            return

        if verbose:
            wprint_info(f"【后处理】同名检查｜{len(dupes)}组")

        # Pre-fetch content for entities missing it — batch query to avoid N+1
        _content_cache: Dict[str, str] = {}
        _fids_needing_content = set()
        # Single pass: populate cache from entities with content, track missing ones
        for name, ents in dupes.items():
            for e in ents:
                if (e.content or "")[:1]:
                    if e.family_id not in _content_cache:
                        _content_cache[e.family_id] = (e.content or "")[:500]
                else:
                    _fids_needing_content.add(e.family_id)
        if _fids_needing_content:
            batch_fn = getattr(self.storage, 'get_entities_by_family_ids', None)
            if batch_fn:
                try:
                    _fetched = batch_fn(list(_fids_needing_content)) or {}
                    for fid, ent in _fetched.items():
                        if ent and ent.content:
                            _content_cache[fid] = (ent.content or "")[:500]
                except Exception:
                    pass
            else:
                for fid in _fids_needing_content:
                    try:
                        versions = self.storage.get_entity_versions(fid)
                        if versions and versions[0].content:
                            _content_cache[fid] = (versions[0].content or "")[:500]
                    except Exception:
                        pass
        # Batch pre-compute embeddings for all unique content strings
        _emb_cache: Dict[str, list] = {}  # content_text -> embedding vector
        _unique_contents = list(set(_content_cache.values()))
        if _unique_contents and self.storage.embedding_client and self.storage.embedding_client.is_available():
            try:
                all_embs = self.storage.embedding_client.encode(_unique_contents)
                for txt, emb in zip(_unique_contents, all_embs):
                    _emb_cache[txt] = emb
            except Exception:
                pass

        # Phase 1: Collect all merge pairs with similarity check (no DB writes yet)
        _merge_pairs: List[tuple] = []  # (old_fid, primary_fid, name, sim)
        for name, ents in dupes.items():
            # Group by family_id (in case same fid appears multiple times)
            fid_groups = defaultdict(list)
            for e in ents:
                fid_groups[e.family_id].append(e)

            fids = list(fid_groups)
            if len(fids) < 2:
                continue

            # Check embedding similarity between pairs
            primary_fid = fids[0]
            primary = fid_groups[primary_fid][0]

            for other_fid in fids[1:]:
                # Compute content similarity using embeddings (uses pre-built cache)
                try:
                    sim = self._compute_entity_content_similarity(primary, fid_groups[other_fid][0], content_cache=_content_cache, emb_cache=_emb_cache)
                except Exception:
                    sim = 0.0

                # Same-name entities: threshold 0.75 (documented default).
                # Name identity is a strong signal but 0.5 is too aggressive — it caused
                # false merges of same-name-different-meaning entities (e.g. "曹操" the person
                # vs "曹操" a poem title).
                if sim >= 0.75:
                    _merge_pairs.append((other_fid, primary_fid, name, sim))
                else:
                    if verbose:
                        wprint_info(f"【后处理】同名保留｜{name} sim={sim:.2f} (不同概念)")

        # Phase 2: Batch execute all merges in a single session (if supported)
        if _merge_pairs:
            _batch_fn = getattr(self.storage, 'dedup_merge_batch', None)
            if _batch_fn:
                # Single batch call — all redirect+delete+register in one session
                try:
                    total_deleted = _batch_fn([(p[0], p[1]) for p in _merge_pairs])
                    if verbose:
                        for old_fid, primary_fid, name, sim in _merge_pairs:
                            wprint_info(f"【后处理】同名合并｜{name} sim={sim:.2f} {old_fid}→{primary_fid} (batch, total del={total_deleted}v)")
                except Exception as e:
                    if verbose:
                        wprint_info(f"【后处理】同名批量合并失败，回退逐条处理: {e}")
                    # Fallback: individual processing
                    for old_fid, primary_fid, name, sim in _merge_pairs:
                        try:
                            self.storage.redirect_entity_relations(old_fid, primary_fid)
                            deleted = self.storage.delete_entity_all_versions(old_fid)
                            self.storage.register_entity_redirect(old_fid, primary_fid)
                            if verbose:
                                wprint_info(f"【后处理】同名合并｜{name} sim={sim:.2f} {old_fid}→{primary_fid} (deleted {deleted}v)")
                        except Exception as e2:
                            if verbose:
                                wprint_info(f"【后处理】同名合并｜{name} sim={sim:.2f} {old_fid}→{primary_fid} (redirect only, merge failed: {e2})")
            else:
                # No batch method — individual processing
                for old_fid, primary_fid, name, sim in _merge_pairs:
                    try:
                        self.storage.redirect_entity_relations(old_fid, primary_fid)
                        deleted = self.storage.delete_entity_all_versions(old_fid)
                        self.storage.register_entity_redirect(old_fid, primary_fid)
                        if verbose:
                            wprint_info(f"【后处理】同名合并｜{name} sim={sim:.2f} {old_fid}→{primary_fid} (deleted {deleted}v)")
                    except Exception as e:
                        if verbose:
                            wprint_info(f"【后处理】同名合并｜{name} sim={sim:.2f} {old_fid}→{primary_fid} (redirect only, merge failed: {e})")

        # ---- Cross-window content-mention alias dedup ----
        # DISABLED: In literary text, entities in the same scene almost always
        # mention each other in their content, causing massive false positives
        # (十里街↔仁清巷, 太虚幻境↔对联, 丫鬟↔巨眼英雄 etc.)
        # Real alias detection is handled by Step 6 (alignment with LLM verification)
        # and name-substring dedup below.
        # self._cross_window_content_mention_dedup(all_entities, verbose=verbose)

        # ---- Cross-window name-substring alias dedup ----
        # Catches alias pairs like "甄士隐"/"士隐", "Docker容器"/"Docker"
        self._cross_window_name_alias_dedup(all_entities, verbose=verbose)

    def _compute_entity_content_similarity(self, entity1, entity2, content_cache: Optional[Dict[str, str]] = None,
                                             emb_cache: Optional[Dict[str, list]] = None):
        """Compute content similarity between two entities using embeddings.

        Uses pre-built content_cache and emb_cache to avoid per-pair encode calls.
        """
        c1 = (entity1.content or "")[:500]
        c2 = (entity2.content or "")[:500]

        # Use cache (preferred — avoids DB calls in O(n^2) loops)
        if not c1 and content_cache:
            c1 = content_cache.get(entity1.family_id, "")
        if not c2 and content_cache:
            c2 = content_cache.get(entity2.family_id, "")

        if not c1 or not c2:
            # Fall back to name-only comparison
            return 1.0 if entity1.name == entity2.name else 0.0

        try:
            if emb_cache:
                emb1 = emb_cache.get(c1)
                emb2 = emb_cache.get(c2)
                if emb1 is not None and emb2 is not None:
                    return max(0.0, min(1.0, cosine_similarity(emb1, emb2)))
            # Fallback: encode on demand if no cache or cache miss
            if self.storage.embedding_client and self.storage.embedding_client.is_available():
                emb1, emb2 = self.storage.embedding_client.encode([c1, c2])
                return max(0.0, min(1.0, cosine_similarity(emb1, emb2)))
            return 1.0 if entity1.name == entity2.name else 0.0
        except Exception as e:
            wprint_info(f"【后处理】相似度计算失败｜{entity1.name} {type(e).__name__}: {e}")
            return 0.0

    def _cross_window_name_alias_dedup(self, all_entities, verbose=True):
        """Detect and merge alias pairs across windows by name substring matching.

        Catches aliases like "甄士隐"/"士隐" or "Docker容器"/"Docker" where
        one name is a substring of the other. Uses LLM verification to confirm
        the two entities truly describe the same concept, preventing false
        positives like "甄家"/"甄家丫鬟" where substring is coincidental.
        """
        if not all_entities or len(all_entities) < 2:
            return

        # Deduplicate by family_id
        fid_to_entity = {}
        for e in all_entities:
            if e.family_id not in fid_to_entity:
                fid_to_entity[e.family_id] = e

        entities = list(fid_to_entity.values())
        if len(entities) < 2:
            return

        # Batch-fetch content for entities missing it (alignment results may not have content)
        _fids_needing_content = [e.family_id for e in entities if not (e.content or "")[:1]]
        if _fids_needing_content:
            try:
                batch_fn = getattr(self.storage, 'get_entities_by_family_ids', None)
                if batch_fn:
                    _fetched = batch_fn(_fids_needing_content) or {}
                    for fid, ent in _fetched.items():
                        if fid in fid_to_entity and ent and ent.content:
                            fid_to_entity[fid] = ent  # replace with version that has content
                else:
                    for fid in _fids_needing_content:
                        try:
                            versions = self.storage.get_entity_versions(fid)
                            if versions and versions[0].content:
                                e = fid_to_entity[fid]
                                e.content = versions[0].content
                        except Exception:
                            pass
            except Exception:
                pass

        # Pre-build content cache to avoid per-pair get_entity_versions calls in the O(n^2) loop
        _content_cache: Dict[str, str] = {}
        for e in entities:
            c = (e.content or "")[:500]
            if c:
                _content_cache[e.family_id] = c

        def _normalize(name):
            core = _PAREN_ANNOTATION_RE.sub('', name).strip()
            core = _TITLE_SUFFIXES_RE.sub('', core).strip()
            return core

        merged_fids = set()
        _merged_count = 0

        # Pre-compute normalized cores and stripped names once (O(n) instead of O(n^2))
        _cores = [_normalize((e.name or "").strip()) for e in entities]
        _names = [(e.name or "").strip() for e in entities]

        # Phase 1: Collect all candidate pairs (substring matches)
        # Skip pairs with same-length core names (equal length → only containment is equality, handled by same-name dedup)
        _candidates = []  # (ent_a, ent_b, core_a, core_b, name_a, name_b)
        _core_lens = [len(c) if c else 0 for c in _cores]
        for i, ent_a in enumerate(entities):
            if ent_a.family_id in merged_fids:
                continue
            core_a = _cores[i]
            if not core_a or len(core_a) < 2:
                continue
            len_a = _core_lens[i]

            for j, ent_b in enumerate(entities):
                if j <= i:
                    continue
                if ent_b.family_id in merged_fids:
                    continue
                core_b = _cores[j]
                if not core_b or len(core_b) < 2:
                    continue

                # Same-length cores can't have non-equal substring containment
                if len_a == _core_lens[j]:
                    continue

                # Check substring relationship
                if core_a in core_b or core_b in core_a:
                    _candidates.append((ent_a, ent_b, core_a, core_b, _names[i], _names[j]))

        # Phase 2: Parallel LLM verification for all candidates
        _llm_results: Dict[tuple, dict] = {}  # (fid_a, fid_b) → {verdict, confidence}
        if _candidates and hasattr(self.llm_client, 'judge_entity_alignment'):
            
            def _verify_pair(ent_a, ent_b, core_a, core_b):
                """Verify a single alias pair via LLM."""
                content_a = _content_cache.get(ent_a.family_id, "")
                content_b = _content_cache.get(ent_b.family_id, "")
                try:
                    result = self.llm_client.judge_entity_alignment(
                        name_a=core_a, content_a=content_a,
                        name_b=core_b, content_b=content_b,
                        name_match_type="substring",
                    )
                    return (ent_a.family_id, ent_b.family_id, result)
                except Exception as e:
                    return (ent_a.family_id, ent_b.family_id, {"verdict": "error", "confidence": 0.0, "error": str(e)})

            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="alias-llm") as pool:
                futures = [
                    pool.submit(_verify_pair, ea, eb, ca, cb)
                    for ea, eb, ca, cb, _, _ in _candidates
                ]
                for future in futures:
                    fid_a, fid_b, result = future.result()
                    _llm_results[(fid_a, fid_b)] = result

        # Phase 3: Apply merges — batch content saves then batch dedup operations
        _content_merges: List[Entity] = []  # Entities to save with merged content
        _dedup_pairs: List[tuple] = []      # (other_fid, primary_fid) for batch dedup
        _merge_info: List[tuple] = []       # (other_name, primary_name, llm_conf, other_fid, primary_fid) for logging

        for ent_a, ent_b, core_a, core_b, name_a, name_b in _candidates:
            if ent_a.family_id in merged_fids or ent_b.family_id in merged_fids:
                continue

            result = _llm_results.get((ent_a.family_id, ent_b.family_id), {})
            llm_confirmed = result.get("verdict") == "same"
            llm_conf = result.get("confidence", 0.0)

            if verbose and not llm_confirmed and result.get("verdict"):
                wprint_info(f"【后处理】名称别名｜'{core_a}' vs '{core_b}' LLM verdict={result.get('verdict')} — 不合并")

            if not llm_confirmed:
                continue

            # Determine primary: keep the longer name (likely the full name)
            if len(core_a) >= len(core_b):
                primary_fid = ent_a.family_id
                other_fid = ent_b.family_id
                primary_name = name_a
                other_name = name_b
                _primary_ent = ent_a
                _alias_ent = ent_b
            else:
                primary_fid = ent_b.family_id
                other_fid = ent_a.family_id
                primary_name = name_b
                other_name = name_a
                _primary_ent = ent_b
                _alias_ent = ent_a

            if primary_fid == other_fid:
                continue

            # Check if already queued for merge (avoid double-processing)
            if other_fid in merged_fids:
                continue
            merged_fids.add(other_fid)

            # Merge content: use entity objects directly
            primary_content = (_primary_ent.content or "")
            alias_content = (_alias_ent.content or "")

            if not primary_content:
                primary_content = _content_cache.get(primary_fid, "")
            if not alias_content:
                alias_content = _content_cache.get(other_fid, "")

            # Queue merged content entity
            if alias_content and alias_content not in primary_content:
                merged_content = primary_content + "\n" + alias_content if primary_content else alias_content
                _content_merges.append(Entity(
                    absolute_id=f"{primary_fid}_v{_uuid.uuid4().hex[:8]}",
                    family_id=primary_fid,
                    name=primary_name,
                    content=merged_content,
                    event_time=_primary_ent.event_time,
                    processed_time=datetime.now(),
                    episode_id="alias_merge",
                    source_document=_primary_ent.source_document or "",
                    confidence=_primary_ent.confidence or 0.5,
                ))

            _dedup_pairs.append((other_fid, primary_fid))
            _merge_info.append((other_name, primary_name, llm_conf, other_fid, primary_fid))
            _merged_count += 1

        # Batch save merged content entities
        if _content_merges:
            _bulk_fn = getattr(self.storage, 'bulk_save_entities', None)
            if _bulk_fn:
                try:
                    _bulk_fn(_content_merges)
                except Exception as me:
                    if verbose:
                        wprint_info(f"【后处理】名称别名合并｜批量content写入失败: {me}")
            else:
                for entity in _content_merges:
                    try:
                        self.storage.save_entity(entity)
                    except Exception as me:
                        if verbose:
                            wprint_info(f"【后处理】名称别名合并｜content合并写入失败: {me}")

        # Batch redirect+delete+register
        if _dedup_pairs:
            _batch_fn = getattr(self.storage, 'dedup_merge_batch', None)
            if _batch_fn:
                try:
                    total_deleted = _batch_fn(_dedup_pairs)
                    if verbose:
                        for other_name, primary_name, llm_conf, other_fid, primary_fid in _merge_info:
                            wprint_info(f"【后处理】名称别名合并｜'{other_name}' → '{primary_name}' LLM_conf={llm_conf:.2f} {other_fid}→{primary_fid} (batch, total del={total_deleted}v)")
                except Exception as e:
                    if verbose:
                        wprint_info(f"【后处理】名称别名批量合并失败，回退逐条处理: {e}")
                    for other_name, primary_name, llm_conf, other_fid, primary_fid in _merge_info:
                        try:
                            self.storage.redirect_entity_relations(other_fid, primary_fid)
                            deleted = self.storage.delete_entity_all_versions(other_fid)
                            self.storage.register_entity_redirect(other_fid, primary_fid)
                            if verbose:
                                wprint_info(f"【后处理】名称别名合并｜'{other_name}' → '{primary_name}' LLM_conf={llm_conf:.2f} {other_fid}→{primary_fid} (deleted {deleted}v)")
                        except Exception as e2:
                            if verbose:
                                wprint_info(f"【后处理】名称别名合并｜'{other_name}' → '{primary_name}' LLM_conf={llm_conf:.2f} {other_fid}→{primary_fid} (redirect only: {e2})")
            else:
                for other_name, primary_name, llm_conf, other_fid, primary_fid in _merge_info:
                    try:
                        self.storage.redirect_entity_relations(other_fid, primary_fid)
                        deleted = self.storage.delete_entity_all_versions(other_fid)
                        self.storage.register_entity_redirect(other_fid, primary_fid)
                        if verbose:
                            wprint_info(f"【后处理】名称别名合并｜'{other_name}' → '{primary_name}' LLM_conf={llm_conf:.2f} {other_fid}→{primary_fid} (deleted {deleted}v)")
                    except Exception as e:
                        if verbose:
                            wprint_info(f"【后处理】名称别名合并｜'{other_name}' → '{primary_name}' LLM_conf={llm_conf:.2f} {other_fid}→{primary_fid} (redirect only: {e})")

        if _merged_count > 0 and verbose:
            wprint_info(f"【后处理】名称别名合并｜共合并 {_merged_count} 对")
