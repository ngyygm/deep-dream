"""
Entity candidate generation — simplified embedding-first approach.

Retrieval strategy:
1. Neo4j vector index top-K search (primary)
2. Exact name dict lookup from projections (supplement)

That's it. No Jaccard matrix, BM25, content-mention, neighbor expansion, etc.
"""
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.storage.sqlite.manager import SQLiteGraphStorageManager
from core.llm.client import LLMClient
from core.debug_log import log_struct as _dbg_struct
from core.utils import wprint_info, calculate_jaccard_similarity, cosine_similarity
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
        # run 级投影索引缓存（P3.3）：同一 remember run 内窗口间库内容不变
        # （除本 run 新建/更新的实体），全量投影扫描只做一次，后续窗口按
        # observation rowid 增量并入；结构性变更（合并/重定向，向量缓存代数
        # 变化）触发整体重建。缓存以 run 为界（token 取 storage._current_run_id，
        # 与 save_entity 的 run_id 传递同款做法），run 结束由管线显式释放。
        self._run_cache_lock = threading.Lock()
        self._run_projection_cache: Optional[Dict[str, Any]] = None

    def _entity_tree_log(self) -> bool:
        return self.verbose and self.entity_progress_verbose

    # ------------------------------------------------------------------
    # Run-level projection index cache
    # ------------------------------------------------------------------

    def release_run_cache(self) -> None:
        """释放 run 级投影缓存（run 结束调用；下一次构建时按新 run 重建）。"""
        with self._run_cache_lock:
            self._run_projection_cache = None

    def _build_projection_cache_entries(self, projections: List[Dict]) -> Dict[str, Any]:
        """把投影行组装为 run 缓存条目（fid/name/core 三索引 + 计数）。"""
        fid_to_proj: Dict[str, Dict] = {}
        name_to_proj: Dict[str, Dict] = {}
        core_to_proj: Dict[str, Dict] = {}
        for p in projections:
            p["_core_name"] = normalize_entity_name_for_matching(p["name"])
            fid_to_proj[p["family_id"]] = p
            # 与逐窗口重建时的语义一致：name 后写覆盖，core 首见保留
            name_to_proj[p["name"]] = p
            core_to_proj.setdefault(p["_core_name"], p)
        return {
            "fid_to_proj": fid_to_proj,
            "name_to_proj": name_to_proj,
            "core_to_proj": core_to_proj,
            "n": len(fid_to_proj),
        }

    def _refresh_run_cache_entries(self, cache: Dict[str, Any], fresh: List[Dict]) -> bool:
        """把增量刷新取回的投影行并入 run 缓存（替换旧条目或新增）。

        返回 True 表示检测到改名（name/core 匹配键被赢家腾空，次级同名
        family 的回退映射无法从增量信息恢复）→ 调用方应丢弃缓存整体重建
        （与全量重扫语义保持严格等价；run 内改名罕见，重建开销可忽略）。

        键的覆盖语义（与全量重扫 DESC 序严格一致）：
        - name 键：全局 updated_at 最旧的赢 → 缓存已有键保留（setdefault），
          批内（fresh 均新于缓存条目）同名校晚者覆盖较早者；
        - core 键：全局 updated_at 最新的赢 → 批内首见（最新）直接覆盖
          缓存旧键（fresh 的 updated_at 必然新于缓存快照内所有条目）。
        """
        fid_to_proj = cache["fid_to_proj"]
        name_to_proj = cache["name_to_proj"]
        core_to_proj = cache["core_to_proj"]
        # 批内聚合：name 后见（批内较旧）覆盖、core 首见（批内最新）保留
        batch_name: Dict[str, Dict] = {}
        batch_core: Dict[str, Dict] = {}
        renamed = False
        for p in fresh:
            p["_core_name"] = normalize_entity_name_for_matching(p["name"])
            old = fid_to_proj.get(p["family_id"])
            if old is not None:
                if old["name"] != p["name"] or old.get("_core_name") != p["_core_name"]:
                    renamed = True
                # 移除旧名映射（改名后旧名不应继续命中）
                if name_to_proj.get(old["name"]) is old:
                    del name_to_proj[old["name"]]
                if core_to_proj.get(old.get("_core_name", "")) is old:
                    del core_to_proj[old.get("_core_name", "")]
                cache["n"] -= 1
            fid_to_proj[p["family_id"]] = p
            batch_name[p["name"]] = p
            batch_core.setdefault(p["_core_name"], p)
            cache["n"] += 1
        if renamed:
            return True
        # name：缓存已有键（更旧）赢，批内新键补充
        for name, p in batch_name.items():
            name_to_proj.setdefault(name, p)
        # core：批内首见（更新）直接覆盖缓存旧键
        core_to_proj.update(batch_core)
        return False

    def _load_projection_index(self, snippet_len: int) -> Dict[str, Any]:
        """加载（或增量刷新）run 级投影索引。step9 逐窗口链式调用，锁竞争可忽略。"""
        token = getattr(self.storage, "_current_run_id", "") or ""
        vgen = getattr(self.storage, "_vector_cache_generation", 0)
        with self._run_cache_lock:
            cache = self._run_projection_cache
            if cache is not None and (
                    cache.get("token") != token
                    or cache.get("snippet_len") != snippet_len
                    or cache.get("vgen") != vgen):
                # run 边界 / 切片长度变化 / 结构性变更（合并、重定向、删除）→ 丢弃重建
                cache = None
            if cache is None:
                # marker 先于全量扫描取：扫描期间并发生效的新行下一窗口会被增量
                # 重复并入（无害），反之（marker 后取）则可能被永久漏掉
                _, max_obs_rowid = self.storage.get_changed_entity_families_since_obs_rowid()
                projections = self.storage.get_latest_entities_projection(snippet_len)
                cache = self._build_projection_cache_entries(projections)
                cache.update({"token": token, "snippet_len": snippet_len,
                              "vgen": vgen, "max_obs_rowid": max_obs_rowid})
                self._run_projection_cache = cache
                cache["rebuild"] = True
            else:
                cache["rebuild"] = False
                changed_fids, max_obs_rowid = (
                    self.storage.get_changed_entity_families_since_obs_rowid(cache["max_obs_rowid"]))
                if changed_fids:
                    fresh = self.storage.get_entities_projection_for_families(changed_fids, snippet_len)
                    renamed = self._refresh_run_cache_entries(cache, fresh)
                    cache["max_obs_rowid"] = max_obs_rowid
                    cache["refreshed"] = len(fresh)
                    if renamed:
                        # 改名使 name/core 回退映射不可增量恢复 → 丢弃整体重建
                        cache = None
                else:
                    cache["refreshed"] = 0
                if cache is None:
                    projections = self.storage.get_latest_entities_projection(snippet_len)
                    cache = self._build_projection_cache_entries(projections)
                    cache.update({"token": token, "snippet_len": snippet_len,
                                  "vgen": vgen, "max_obs_rowid": max_obs_rowid})
                    self._run_projection_cache = cache
                    cache["rebuild"] = True
            return cache

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

        # ── Load run-level projection index (full scan once per run, incremental after) ──
        _snippet_len = self.llm_client.effective_entity_snippet_length()
        index = self._load_projection_index(_snippet_len)
        n_projections = index["n"]
        if not n_projections:
            wprint_info("[candidate_table] ⚠️ No existing entities for alignment")
            return {}
        name_to_proj: Dict[str, Dict] = index["name_to_proj"]
        core_to_proj: Dict[str, Dict] = index["core_to_proj"]
        fid_to_proj: Dict[str, Dict] = index["fid_to_proj"]

        if index.get("rebuild"):
            wprint_info(f"[candidate_table] {n_projections} existing entities (run 缓存重建)")
        else:
            wprint_info(f"[candidate_table] {n_projections} existing entities (run 缓存命中, 增量并入 {index.get('refreshed', 0)})")

        # ── Encode extracted entities ──
        name_embeddings: Optional[Any] = None
        full_embeddings: Optional[Any] = None
        if prefetched_embeddings is not None:
            name_embeddings, full_embeddings = prefetched_embeddings
        elif self.storage.embedding_client and self.storage.embedding_client.is_available():
            _N = len(extracted_entities)
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
        top_k = max(self.max_alignment_candidates or self.max_similar_entities, n_projections, 10)
        name_emb_scores, full_emb_scores = self._search_embedding_top_k(
            extracted_entities, name_embeddings, full_embeddings, top_k,
        )

        _t_vec = time.monotonic()
        wprint_info(f"[candidate_timing] embedding vector top-K search: {_t_vec - _t_encode:.3f}s")

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
