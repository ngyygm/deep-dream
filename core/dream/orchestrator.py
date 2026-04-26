"""梦境编排器 — 种子选择、图探索、隐含关联发现。

将 api.py 中的 dream_run 逻辑提取为独立模块，便于：
- 独立测试
- 后台调度
- 可配置策略
- 并发 LLM 调用
"""

import json
import logging
import time
import uuid
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


def _trunc(s: str, n: int = 200) -> str:
    """Truncate string only if needed — avoids slicing copy for short strings."""
    return s[:n] if len(s) > n else s

from core.llm.prompts import (
    JUDGE_AND_GENERATE_RELATION_SYSTEM_PROMPT,
)
from core.find.graph_traversal import GraphTraversalSearcher

logger = logging.getLogger(__name__)

# 合法种子策略
VALID_STRATEGIES = frozenset([
    "random", "orphan", "hub", "time_gap",
    "cross_community", "low_confidence",
])
_STRATEGY_DISPLAY_ORDER = ["random", "orphan", "hub", "time_gap",
                           "cross_community", "low_confidence"]


class DreamHistory:
    """跨周期探索历史 — 避免重复检查相同的实体对。

    使用 LRU 淘汰策略：保留最近 _max_entries 条检查记录，
    超出时淘汰最早的记录，允许过期对在足够多的周期后被重新探索。
    同时追踪策略使用记录，支持跨周期策略轮换。
    """

    def __init__(self, max_entries: int = 2000):
        # key: frozenset(entity1_fid, entity2_fid), value: cycle_id
        self._checked_pairs: OrderedDict = OrderedDict()
        # 记录每个 cycle 探索过的实体 family_ids
        self._explored_entities: Dict[str, Set[str]] = {}
        self._max_entries = max_entries
        # 策略使用历史：记录最近使用的策略（用于轮换）
        self._strategy_history: List[str] = []
        # 每个策略的效果统计：strategy -> {"cycles": int, "relations": int}
        self._strategy_stats: Dict[str, Dict[str, int]] = {}

    def mark_checked(self, fid1: str, fid2: str, cycle_id: str) -> None:
        """记录一对实体已被检查。"""
        key = (fid1, fid2) if fid1 <= fid2 else (fid2, fid1)
        self._checked_pairs[key] = cycle_id
        # LRU 淘汰
        if len(self._checked_pairs) > self._max_entries:
            self._checked_pairs.popitem(last=False)

    def was_checked(self, fid1: str, fid2: str) -> bool:
        """判断一对实体是否已被检查过。"""
        key = (fid1, fid2) if fid1 <= fid2 else (fid2, fid1)
        if key in self._checked_pairs:
            # 移到末尾（最近访问）
            self._checked_pairs.move_to_end(key)
            return True
        return False

    def mark_explored(self, cycle_id: str, entity_ids: Set[str]) -> None:
        """记录一个周期探索过的实体。"""
        self._explored_entities[cycle_id] = entity_ids
        # 只保留最近 10 个周期
        if len(self._explored_entities) > 10:
            oldest = next(iter(self._explored_entities))
            del self._explored_entities[oldest]

    def get_recently_explored(self, last_n: int = 3) -> Set[str]:
        """获取最近 N 个周期探索过的所有实体 family_id。"""
        recent_keys = list(self._explored_entities.keys())[-last_n:]
        result: Set[str] = set()
        for k in recent_keys:
            result.update(self._explored_entities[k])
        return result

    def next_strategy(self, current: Optional[str] = None) -> str:
        """选择下一个策略：轮换使用，优先使用效果好的策略。

        策略选择逻辑：
        1. 如果有从未使用过的策略，优先使用
        2. 否则选最近最少使用（LRU）的策略
        3. 同等 LRU 时，优先选效果好的（relations/cycles 比率高的）
        """
        used = set(self._strategy_history)
        unused = [s for s in VALID_STRATEGIES if s not in used]
        if unused:
            return unused[0]

        # 所有策略都用过 — 选最近最少使用的
        recent_set = set(self._strategy_history[-len(VALID_STRATEGIES):])
        lru_candidates = [s for s in VALID_STRATEGIES if s not in recent_set]

        if lru_candidates:
            # 多个 LRU 候选时，选效果最好的
            return self._best_by_stats(lru_candidates)

        # 所有策略都在最近窗口内 — 选使用次数最少的
        counts = Counter(self._strategy_history)
        min_count = min(counts.get(s, 0) for s in VALID_STRATEGIES)
        least_used = [s for s in VALID_STRATEGIES if counts.get(s, 0) == min_count]
        return self._best_by_stats(least_used)

    def _best_by_stats(self, candidates: List[str]) -> str:
        """从候选策略中选效果最好的（relations_per_cycle 最高的）。"""
        if len(candidates) == 1:
            return candidates[0]

        best = candidates[0]
        best_ratio = -1.0
        for s in candidates:
            stats = self._strategy_stats.get(s, {})
            cycles = stats.get("cycles", 0)
            rels = stats.get("relations", 0)
            ratio = rels / cycles if cycles > 0 else 0.0
            if ratio > best_ratio:
                best_ratio = ratio
                best = s
        return best

    def record_strategy_result(self, strategy: str, relations_found: int) -> None:
        """记录一次策略执行的結果。"""
        self._strategy_history.append(strategy)
        # 只保留最近 60 条策略记录（原地裁剪）
        if len(self._strategy_history) > 60:
            del self._strategy_history[:-60]
        # 更新统计
        stats = self._strategy_stats.setdefault(strategy, {"cycles": 0, "relations": 0})
        stats["cycles"] += 1
        stats["relations"] += relations_found

    def get_strategy_stats(self) -> Dict[str, Dict[str, int]]:
        """获取策略效果统计。"""
        return dict(self._strategy_stats)

    def reset(self) -> None:
        """清空历史。"""
        self._checked_pairs.clear()
        self._explored_entities.clear()
        self._strategy_history.clear()
        self._strategy_stats.clear()


@dataclass(slots=True)
class DreamConfig:
    """梦境配置参数。"""
    strategy: str = "random"
    seed_count: int = 3
    max_depth: int = 2
    max_relations: int = 5
    min_confidence: float = 0.5
    max_explore_entities: int = 50
    max_neighbors_per_seed: int = 10
    exclude_ids: List[str] = field(default_factory=list)
    llm_timeout: int = 60
    llm_concurrency: int = 3
    min_pair_similarity: float = 0.0

    def __post_init__(self):
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"无效策略: {self.strategy}，可选: {', '.join(_STRATEGY_DISPLAY_ORDER)}"
            )
        self.seed_count = min(max(self.seed_count, 1), 10)
        self.max_depth = min(max(self.max_depth, 1), 4)
        self.max_relations = min(max(self.max_relations, 1), 20)
        self.min_confidence = max(0.0, min(1.0, self.min_confidence))
        self.min_pair_similarity = max(0.0, min(1.0, self.min_pair_similarity))


@dataclass(slots=True)
class DreamResult:
    """梦境运行结果。"""
    cycle_id: str
    strategy: str
    seeds: List[Dict[str, str]]
    explored: List[Dict[str, Any]]
    relations_created: List[Dict[str, Any]]
    stats: Dict[str, Any]
    cycle_summary: str


class DreamOrchestrator:
    """梦境编排器：种子选择 → 图探索 → 关联发现 → 结果保存。

    支持手动触发 run()。
    """

    def __init__(self, storage: Any, llm_client: Any, config: Optional[DreamConfig] = None):
        self.storage = storage
        self.llm_client = llm_client
        self.config = config or DreamConfig()
        self._searcher = GraphTraversalSearcher(storage)
        self._history = DreamHistory()
        self._cycle_count = 0
        self._pool = ThreadPoolExecutor(max_workers=getattr(config, 'llm_concurrency', 2), thread_name_prefix="dream")

    def run(self, auto_rotate: bool = False) -> DreamResult:
        """执行一轮完整的梦境周期。

        Args:
            auto_rotate: 如果为 True，自动轮换策略（忽略 config.strategy，
                         使用 DreamHistory 推荐的下一个策略）。
        """
        config = self.config
        self._cycle_count += 1
        cycle_id = f"dream_{uuid.uuid4().hex[:12]}"

        # Step 0: 策略轮换（如果启用）
        effective_strategy = config.strategy
        if auto_rotate:
            effective_strategy = self._history.next_strategy(config.strategy)
            logger.info("Dream: 策略轮换 → %s", effective_strategy)

        # 创建临时 config（不修改 self.config，保证线程安全）
        # Short-circuit when no rotation and no exclude_ids — config is identical
        if not auto_rotate and not config.exclude_ids:
            run_config = config
        else:
            run_config = DreamConfig(
                strategy=effective_strategy,
                seed_count=config.seed_count,
                max_depth=config.max_depth,
                max_relations=config.max_relations,
                min_confidence=config.min_confidence,
                max_explore_entities=config.max_explore_entities,
                max_neighbors_per_seed=config.max_neighbors_per_seed,
                exclude_ids=list(config.exclude_ids) if config.exclude_ids else [],
                llm_timeout=config.llm_timeout,
                llm_concurrency=config.llm_concurrency,
                min_pair_similarity=config.min_pair_similarity,
            )

        # Step 1: 种子选择（排除近期探索过的实体）
        recently_explored = self._history.get_recently_explored(last_n=3)
        seeds = self._select_seeds(run_config, recently_explored)
        if not seeds:
            self._history.record_strategy_result(effective_strategy, 0)
            return DreamResult(
                cycle_id=cycle_id,
                strategy=effective_strategy,
                seeds=[],
                explored=[],
                relations_created=[],
                stats={"seeds_count": 0, "entities_explored": 0,
                       "pairs_checked": 0, "relations_created_count": 0,
                       "strategy_rotated": auto_rotate},
                cycle_summary="图谱为空或无可用种子，梦境结束",
            )

        # Step 2: BFS 图探索（含已有关系上下文）
        entity_lookup, seen_ids, explored, relation_context = self._explore_graph(seeds, run_config)

        # Step 3: 关联发现
        relations_created, pairs_checked = self._discover_relations(
            seeds, explored, entity_lookup, cycle_id, run_config,
            relation_context=relation_context,
        )

        # Step 4: 保存梦境记录
        cycle_summary = (
            f"梦境周期 {cycle_id}：策略={effective_strategy}，种子={len(seeds)}，"
            f"探索实体={len(seen_ids)}，检查配对={pairs_checked}，"
            f"创建关系={len(relations_created)}"
        )
        self._save_episode(
            cycle_id, cycle_summary, seen_ids, relations_created, run_config,
        )

        # Step 5: 更新跨周期历史
        self._history.mark_explored(cycle_id, seen_ids)
        for r in relations_created:
            self._history.mark_checked(r["entity1_id"], r["entity2_id"], cycle_id)

        # Step 6: 记录策略效果
        self._history.record_strategy_result(effective_strategy, len(relations_created))

        return DreamResult(
            cycle_id=cycle_id,
            strategy=effective_strategy,
            seeds=[{"family_id": s.get("family_id"), "name": s.get("name", "")} for s in seeds],
            explored=explored,
            relations_created=relations_created,
            stats={
                "seeds_count": len(seeds),
                "entities_explored": len(seen_ids),
                "pairs_checked": pairs_checked,
                "relations_created_count": len(relations_created),
                "strategy_rotated": auto_rotate,
            },
            cycle_summary=cycle_summary,
        )

    # ------------------------------------------------------------------
    # Step 1: 种子选择
    # ------------------------------------------------------------------

    def _select_seeds(self, config: DreamConfig, recently_explored: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """从存储层获取梦境种子（排除近期已探索的实体）。"""
        exclude = set(config.exclude_ids)
        if recently_explored:
            exclude.update(recently_explored)
        try:
            return self.storage.get_dream_seeds(
                strategy=config.strategy,
                count=config.seed_count,
                exclude_ids=list(exclude),
            )
        except Exception as e:
            logger.warning("Dream: 种子选择失败: %s", e)
            return []

    # ------------------------------------------------------------------
    # Step 2: 图探索
    # ------------------------------------------------------------------

    def _explore_graph(
        self,
        seeds: List[Dict[str, Any]],
        config: DreamConfig,
    ) -> tuple:
        """BFS 扩展获取邻居实体及已有关系。

        Returns:
            (entity_lookup, seen_ids, explored_list, relation_context)
            relation_context: family_id -> list of "neighbor_name — relation_snippet"
        """
        seed_family_ids = [s["family_id"] for s in seeds if s.get("family_id")]

        try:
            bfs_entities, bfs_relations, _ = self._searcher.bfs_expand_with_relations(
                seed_family_ids,
                max_depth=config.max_depth,
                max_nodes=config.max_explore_entities,
            )
        except Exception as exc:
            logger.warning("Dream: BFS遍历失败: %s", exc)
            bfs_entities, bfs_relations = [], []

        # 构建 entity_lookup + abs/fid/name 映射（单次遍历）
        entity_lookup: Dict[str, Dict[str, str]] = {}
        seen_ids: Set[str] = set()
        abs_to_fid: Dict[str, str] = {}
        fid_to_name: Dict[str, str] = {}

        for ent in bfs_entities:
            fid = ent.family_id
            abs_id = ent.absolute_id
            name = ent.name
            if fid:
                entity_lookup[fid] = {
                    "family_id": fid,
                    "name": name,
                    "content": _trunc(ent.content or '', 500),
                }
                seen_ids.add(fid)
            if abs_id:
                abs_to_fid[abs_id] = fid or ''
            if fid:
                fid_to_name[fid] = name

        # 补充种子自身
        for s in seeds:
            fid = s.get("family_id")
            if fid and fid not in entity_lookup:
                entity_lookup[fid] = {
                    "family_id": fid,
                    "name": s.get("name", ""),
                    "content": _trunc(s.get("content") or "", 500),
                }
                seen_ids.add(fid)
                fid_to_name[fid] = s.get("name", "")

        # 构建关系上下文: family_id -> ["neighbor_name — relation_snippet"]
        # 同时构建邻接表: family_id -> set of neighbor family_ids
        relation_context: Dict[str, List[str]] = defaultdict(list)
        adjacency: Dict[str, set] = defaultdict(set)
        for rel in bfs_relations:
            e1_abs = rel.entity1_absolute_id
            e2_abs = rel.entity2_absolute_id
            e1_fid = abs_to_fid.get(e1_abs)
            e2_fid = abs_to_fid.get(e2_abs)
            content_snippet = (rel.content or '')[:80]
            e1_name = fid_to_name.get(e1_fid, '') if e1_fid else ''
            e2_name = fid_to_name.get(e2_fid, '') if e2_fid else ''

            if e1_fid:
                relation_context[e1_fid].append(
                    f"{e2_name or e2_abs[:12]} — {content_snippet}"
                )
                if e2_fid:
                    adjacency[e1_fid].add(e2_fid)
            if e2_fid:
                relation_context[e2_fid].append(
                    f"{e1_name or e1_abs[:12]} — {content_snippet}"
                )
                if e1_fid:
                    adjacency[e2_fid].add(e1_fid)

        # 为每个种子构建 explored 信息
        explored: List[Dict[str, Any]] = []
        for seed in seeds:
            fid = seed.get("family_id")
            if not fid:
                continue
            # Use adjacency dict for O(1) neighbor lookup per seed
            neighbor_fids = adjacency.get(fid, set())
            neighbor_data = []
            for nfid in neighbor_fids:
                info = entity_lookup.get(nfid)
                if info:
                    neighbor_data.append({
                        "family_id": nfid,
                        "name": info["name"],
                        "content": _trunc(info["content"], 200),
                    })

            # Fallback for isolated seeds: use embedding similarity to find neighbors
            if not neighbor_data and hasattr(self.storage, 'search_entities_by_similarity'):
                try:
                    seed_name = seed.get("name", "")
                    seed_content = _trunc(seed.get("content") or "", 200)
                    sim_results = self.storage.search_entities_by_similarity(
                        query_name=seed_name,
                        query_content=seed_content,
                        limit=config.max_neighbors_per_seed,
                    )
                    for ent in sim_results:
                        efid = ent.family_id
                        if efid and efid != fid and ent.invalid_at is None:
                            info = {
                                "family_id": efid,
                                "name": ent.name,
                                "content": _trunc(ent.content or '', 200),
                            }
                            neighbor_data.append(info)
                            if efid not in entity_lookup:
                                entity_lookup[efid] = {
                                    "family_id": efid,
                                    "name": info["name"],
                                    "content": info["content"],
                                }
                                seen_ids.add(efid)
                    if neighbor_data:
                        logger.info("Dream: 孤立种子 %s 通过embedding找到 %d 个候选邻居", seed_name, len(neighbor_data))
                except Exception as exc:
                    logger.warning("Dream: embedding fallback failed for seed %s: %s", fid, exc)

            explored.append({
                "seed": {"family_id": fid, "name": seed.get("name", "")},
                "neighbors": neighbor_data[:20],
                "neighbor_count": len(neighbor_data),
            })

        return entity_lookup, seen_ids, explored, relation_context

    # ------------------------------------------------------------------
    # Step 3: 关联发现（并发 LLM 判断）
    # ------------------------------------------------------------------

    def _discover_relations(
        self,
        seeds: List[Dict[str, Any]],
        explored: List[Dict[str, Any]],
        entity_lookup: Dict[str, Dict[str, str]],
        cycle_id: str,
        config: DreamConfig,
        relation_context: Optional[Dict[str, List[str]]] = None,
    ) -> tuple:
        """并发发现实体间的隐含关联。

        Returns:
            (relations_created, pairs_checked)
        """
        # 收集所有待检查的配对（跳过历史已检查的）
        pairs: List[tuple] = []
        skipped_by_history = 0
        involved_fids_set: set = set()
        _history = self._history
        for exp in explored:
            seed_info = exp["seed"]
            seed_fid = seed_info["family_id"]
            seed_name = seed_info["name"]
            for neighbor in exp["neighbors"][:config.max_neighbors_per_seed]:
                nb_fid = neighbor["family_id"]
                if _history.was_checked(seed_fid, nb_fid):
                    skipped_by_history += 1
                    continue
                pairs.append((seed_fid, seed_name, nb_fid, neighbor["name"]))
                involved_fids_set.add(seed_fid)
                involved_fids_set.add(nb_fid)

        if skipped_by_history:
            logger.info("Dream: 跳过 %d 对历史已检查的配对", skipped_by_history)

        if not pairs:
            return [], 0

        # 语义预过滤：跳过语义相似度过低的配对
        pairs = self._prefilter_pairs_by_similarity(pairs, entity_lookup, config, involved_fids_set)

        if not pairs:
            return [], 0

        # 批量预取所有配对的已有关系，避免 _judge_pair 中逐对查询
        try:
            pair_keys = [(p[0], p[2]) for p in pairs]
            existing_map = self.storage.get_relations_by_entity_pairs(pair_keys)
            existing_pairs = {k for k, v in existing_map.items() if v}
        except Exception as exc:
            logger.debug("Dream: 批量关系预取失败，回退逐对查询: %s", exc)
            existing_pairs = None

        relations_created: List[Dict[str, Any]] = []
        pairs_checked = 0

        # Batch pre-fetch all entities involved in pairs (avoid N+1 in save_dream_relation)
        _all_pair_fids = involved_fids_set  # already computed above (superset of filtered pairs' fids)
        _entity_lookup = {}
        if _all_pair_fids:
            try:
                _entity_lookup = self.storage.get_entities_by_family_ids(list(_all_pair_fids))
            except Exception:
                pass

        # Backfill entity_lookup for any fids missing from BFS-built lookup
        _missing_fids = involved_fids_set - set(entity_lookup.keys())
        if _missing_fids:
            try:
                _backfill = self.storage.get_entities_by_family_ids(list(_missing_fids))
                for fid, ent in _backfill.items():
                    if fid not in entity_lookup:
                        entity_lookup[fid] = {
                            "family_id": fid,
                            "name": ent.name if hasattr(ent, "name") else ent.get("name", ""),
                            "content": _trunc(
                                (ent.content or "") if hasattr(ent, "content") else ent.get("content", ""),
                                500,
                            ),
                        }
            except Exception:
                pass

        # 使用共享线程池并发判断
        _pool = self._pool
        futures = {}
        for pair in pairs:
            seed_fid, seed_name, nb_fid, nb_name = pair
            future = _pool.submit(
                self._judge_pair,
                seed_fid, seed_name, nb_fid, nb_name, config,
                entity_lookup, existing_pairs, relation_context,
            )
            futures[future] = pair

        early_break = False
        for future in as_completed(futures):
            pair = futures[future]
            seed_fid, seed_name, nb_fid, nb_name = pair
            # Always mark checked to prevent re-checking in future cycles
            self._history.mark_checked(seed_fid, nb_fid, cycle_id)
            pairs_checked += 1

            if early_break:
                # Already hit max_relations — just drain futures and mark history
                continue
            if len(relations_created) >= config.max_relations:
                early_break = True
                continue

            try:
                result = future.result()
                if result is None:
                    continue

                # 保存 dream relation
                confidence = result["confidence"]
                if confidence < config.min_confidence:
                    continue

                reasoning = f"梦境发现：{seed_name} 与 {nb_name} 存在潜在关联（策略: {config.strategy}）"

                # Use fast path with pre-fetched entities if available
                _e1 = _entity_lookup.get(seed_fid)
                _e2 = _entity_lookup.get(nb_fid)
                if _e1 and _e2:
                    save_result = self.storage.save_dream_relation_fast(
                        entity1=_e1, entity2=_e2,
                        resolved1=seed_fid, resolved2=nb_fid,
                        content=result["content"],
                        confidence=confidence,
                        reasoning=reasoning,
                        dream_cycle_id=cycle_id,
                    )
                else:
                    save_result = self.storage.save_dream_relation(
                        entity1_id=seed_fid,
                        entity2_id=nb_fid,
                        content=result["content"],
                        confidence=confidence,
                        reasoning=reasoning,
                        dream_cycle_id=cycle_id,
                    )
                relations_created.append({
                    "entity1_id": seed_fid,
                    "entity1_name": seed_name,
                    "entity2_id": nb_fid,
                    "entity2_name": nb_name,
                    "content": result["content"],
                    "confidence": confidence,
                    "result": save_result,
                })
            except Exception as exc:
                logger.warning("Dream: 检查关系 %s↔%s 时出错: %s", pair[0], pair[2], exc)

        return relations_created, pairs_checked

    def _prefilter_pairs_by_similarity(
        self,
        pairs: List[tuple],
        entity_lookup: Dict[str, Dict[str, str]],
        config: DreamConfig,
        involved_fids: set = None,
    ) -> List[tuple]:
        """基于 embedding 余弦相似度预过滤配对，跳过语义不相关的配对。

        当 min_pair_similarity > 0 且 embedding 客户端可用时，
        批量计算所有实体的 embedding 并过滤低相似度配对。
        无法获取 embedding 的实体保留（不过滤）。
        """
        if config.min_pair_similarity <= 0:
            return pairs

        ec = getattr(self.storage, 'embedding_client', None)
        if not ec or not getattr(ec, 'is_available', lambda: False)():
            return pairs

        # Use pre-computed fids if provided, otherwise collect from pairs
        if involved_fids is None:
            involved_fids = set()
            for seed_fid, _, nb_fid, _ in pairs:
                involved_fids.add(seed_fid)
                involved_fids.add(nb_fid)

        # 批量计算 embedding
        texts = []
        fid_list = []
        for fid in involved_fids:
            info = entity_lookup.get(fid)
            if info:
                text = f"{info.get('name', '')}: {info.get('content', '')}"
                texts.append(text)
                fid_list.append(fid)

        if not texts:
            return pairs

        try:
            embeddings = ec.encode(texts)
            if embeddings is None:
                return pairs
            fid_to_emb: Dict[str, np.ndarray] = {}
            emb_matrix = np.array(embeddings[:len(fid_list)], dtype=np.float32)
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-9)
            emb_normed = emb_matrix / norms
            for i, fid in enumerate(fid_list):
                fid_to_emb[fid] = emb_normed[i]
        except Exception as exc:
            logger.warning("Dream: embedding 预计算失败，跳过语义过滤: %s", exc)
            return pairs

        # 过滤配对
        filtered = []
        for pair in pairs:
            seed_fid, _, nb_fid, _ = pair
            e1 = fid_to_emb.get(seed_fid)
            e2 = fid_to_emb.get(nb_fid)
            if e1 is not None and e2 is not None:
                similarity = float(np.dot(e1, e2))
                if similarity < config.min_pair_similarity:
                    continue
            # 无 embedding 的实体保留
            filtered.append(pair)

        if len(filtered) < len(pairs):
            logger.info(
                "Dream: 语义预过滤 %d→%d 对 (阈值=%.2f)",
                len(pairs), len(filtered), config.min_pair_similarity,
            )

        return filtered

    def _judge_pair(
        self,
        seed_fid: str,
        seed_name: str,
        nb_fid: str,
        nb_name: str,
        config: DreamConfig,
        entity_lookup: Optional[Dict[str, Dict[str, str]]] = None,
        existing_pairs: Optional[Set[Tuple[str, str]]] = None,
        relation_context: Optional[Dict[str, List[str]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """判断一对实体是否存在隐含关联。

        Returns:
            None 表示无关联，dict 包含 content 和 confidence 表示有关联。
        """
        # 检查是否已有关系（优先使用批量预取结果）
        if existing_pairs is not None:
            pair_key = (seed_fid, nb_fid)
            rev_key = (nb_fid, seed_fid)
            if pair_key in existing_pairs or rev_key in existing_pairs:
                return None
        else:
            try:
                existing = self.storage.get_relations_by_entities(seed_fid, nb_fid)
                if existing:
                    return None
            except Exception as exc:
                logger.debug("Dream: existing relation check failed for %s↔%s: %s", seed_fid, nb_fid, exc)

        # 优先从 entity_lookup 获取实体详情，避免重复 DB 查询
        if entity_lookup:
            seed_info = entity_lookup.get(seed_fid)
            nb_info = entity_lookup.get(nb_fid)
            if not seed_info or not nb_info:
                return None
            seed_name = seed_info.get("name", seed_name)
            seed_content = seed_info.get("content", "")
            nb_name = nb_info.get("name", nb_name)
            nb_content = nb_info.get("content", "")
        else:
            seed_entity = self.storage.get_entity_by_family_id(seed_fid)
            nb_entity = self.storage.get_entity_by_family_id(nb_fid)
            if not seed_entity or not nb_entity:
                return None
            seed_name = seed_entity.name
            seed_content = _trunc(seed_entity.content or "", 500)
            nb_name = nb_entity.name
            nb_content = _trunc(nb_entity.content or "", 500)

        # 构建图拓扑上下文（每个实体已有的关系）
        topology_lines = []
        if relation_context:
            seed_rels = relation_context.get(seed_fid, [])
            nb_rels = relation_context.get(nb_fid, [])
            if seed_rels:
                topology_lines.append(
                    f"实体A的已知关联:\n" + "\n".join(f"  - {r}" for r in seed_rels[:8])
                )
            if nb_rels:
                topology_lines.append(
                    f"实体B的已知关联:\n" + "\n".join(f"  - {r}" for r in nb_rels[:8])
                )
        topology_block = ("\n\n".join(topology_lines) + "\n\n") if topology_lines else ""

        # LLM 判断 + 生成（单次调用）
        judge_messages = [
            {"role": "system", "content": JUDGE_AND_GENERATE_RELATION_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"实体A: {seed_name}\n描述: {seed_content}\n\n"
                f"实体B: {nb_name}\n描述: {nb_content}\n\n"
                f"{topology_block}"
                "判断这两个实体之间是否存在明确的、有意义的关联。如果存在，同时生成关系描述。"
            )},
        ]
        judge_obj, _ = self.llm_client.call_llm_until_json_parses(
            judge_messages,
            parse_fn=json.loads,
            json_parse_retries=1,
            timeout=config.llm_timeout,
        )
        if not judge_obj.get("need_create", False):
            return None

        judge_confidence = float(judge_obj.get("confidence", 0.5))
        rel_content = (judge_obj.get("content") or "").strip()
        if not rel_content or len(rel_content) < 10:
            return None

        return {
            "content": rel_content,
            "confidence": max(0.1, min(1.0, judge_confidence)),
        }

    # ------------------------------------------------------------------
    # Step 4: 保存梦境记录
    # ------------------------------------------------------------------

    def _save_episode(
        self,
        cycle_id: str,
        cycle_summary: str,
        seen_ids: Set[str],
        relations_created: List[Dict[str, Any]],
        config: DreamConfig,
    ) -> None:
        """保存梦境周期记录。"""
        try:
            self.storage.save_dream_episode(
                content=cycle_summary,
                entities_examined=list(seen_ids)[:50],
                relations_created=[
                    r.get("result", {}).get("family_id", "")
                    for r in relations_created if r.get("result")
                ],
                strategy_used=config.strategy,
                dream_cycle_id=cycle_id,
                relations_created_count=len(relations_created),
            )
        except Exception as exc:
            logger.warning("Dream: 保存梦境记录失败: %s", exc)

