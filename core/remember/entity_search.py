"""
Entity search, filtering, and alignment guard helpers.
Extracted from EntityProcessor for modularity.
"""
from typing import List, Dict, Optional, Tuple, Any
from collections import OrderedDict
import logging

import numpy as np

from core.models import Entity
from core.storage.sqlite.manager import SQLiteGraphStorageManager as Neo4jStorageManager
from core.llm.client import LLMClient
from core.utils import wprint_info, calculate_jaccard_similarity, cosine_similarity
from core.debug_log import log_struct as _dbg_struct
from core.remember.entity_candidates import _TITLE_SUFFIXES_RE
from core.remember._shared import _doc_basename

logger = logging.getLogger(__name__)


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
    storage: Neo4jStorageManager,
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
    storage: Neo4jStorageManager,
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
