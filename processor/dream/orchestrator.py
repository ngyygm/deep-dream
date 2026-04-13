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
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..llm.prompts import (
    JUDGE_AND_GENERATE_RELATION_SYSTEM_PROMPT,
)
from ..search.graph_traversal import GraphTraversalSearcher

logger = logging.getLogger(__name__)

# 合法种子策略
VALID_STRATEGIES = [
    "random", "orphan", "hub", "time_gap",
    "cross_community", "low_confidence",
]


class DreamHistory:
    """跨周期探索历史 — 避免重复检查相同的实体对。

    使用 LRU 淘汰策略：保留最近 _max_entries 条检查记录，
    超出时淘汰最早的记录，允许过期对在足够多的周期后被重新探索。
    """

    def __init__(self, max_entries: int = 2000):
        # key: frozenset(entity1_fid, entity2_fid), value: cycle_id
        self._checked_pairs: OrderedDict = OrderedDict()
        # 记录每个 cycle 探索过的实体 family_ids
        self._explored_entities: Dict[str, Set[str]] = {}
        self._max_entries = max_entries

    def mark_checked(self, fid1: str, fid2: str, cycle_id: str) -> None:
        """记录一对实体已被检查。"""
        key = frozenset((fid1, fid2))
        self._checked_pairs[key] = cycle_id
        # LRU 淘汰
        if len(self._checked_pairs) > self._max_entries:
            self._checked_pairs.popitem(last=False)

    def was_checked(self, fid1: str, fid2: str) -> bool:
        """判断一对实体是否已被检查过。"""
        key = frozenset((fid1, fid2))
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

    def reset(self) -> None:
        """清空历史。"""
        self._checked_pairs.clear()
        self._explored_entities.clear()


@dataclass
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

    def __post_init__(self):
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"无效策略: {self.strategy}，可选: {', '.join(VALID_STRATEGIES)}"
            )
        self.seed_count = min(max(self.seed_count, 1), 10)
        self.max_depth = min(max(self.max_depth, 1), 4)
        self.max_relations = min(max(self.max_relations, 1), 20)
        self.min_confidence = max(0.0, min(1.0, self.min_confidence))


@dataclass
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

    def run(self) -> DreamResult:
        """执行一轮完整的梦境周期。"""
        config = self.config
        self._cycle_count += 1
        cycle_id = f"dream_{uuid.uuid4().hex[:12]}"

        # Step 1: 种子选择（排除近期探索过的实体）
        recently_explored = self._history.get_recently_explored(last_n=3)
        seeds = self._select_seeds(config, recently_explored)
        if not seeds:
            return DreamResult(
                cycle_id=cycle_id,
                strategy=config.strategy,
                seeds=[],
                explored=[],
                relations_created=[],
                stats={"seeds_count": 0, "entities_explored": 0,
                       "pairs_checked": 0, "relations_created_count": 0},
                cycle_summary="图谱为空或无可用种子，梦境结束",
            )

        # Step 2: BFS 图探索
        entity_lookup, seen_ids, explored = self._explore_graph(seeds, config)

        # Step 3: 关联发现
        relations_created, pairs_checked = self._discover_relations(
            seeds, explored, entity_lookup, cycle_id, config,
        )

        # Step 4: 保存梦境记录
        cycle_summary = (
            f"梦境周期 {cycle_id}：策略={config.strategy}，种子={len(seeds)}，"
            f"探索实体={len(seen_ids)}，检查配对={pairs_checked}，"
            f"创建关系={len(relations_created)}"
        )
        self._save_episode(
            cycle_id, cycle_summary, seen_ids, relations_created, config,
        )

        # Step 5: 更新跨周期历史
        self._history.mark_explored(cycle_id, seen_ids)
        for r in relations_created:
            self._history.mark_checked(r["entity1_id"], r["entity2_id"], cycle_id)

        return DreamResult(
            cycle_id=cycle_id,
            strategy=config.strategy,
            seeds=[{"family_id": s.get("family_id"), "name": s.get("name", "")} for s in seeds],
            explored=explored,
            relations_created=relations_created,
            stats={
                "seeds_count": len(seeds),
                "entities_explored": len(seen_ids),
                "pairs_checked": pairs_checked,
                "relations_created_count": len(relations_created),
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
        """BFS 扩展获取邻居实体。

        Returns:
            (entity_lookup, seen_ids, explored_list)
        """
        seed_family_ids = [s["family_id"] for s in seeds if s.get("family_id")]

        try:
            bfs_entities = self._searcher.bfs_expand(
                seed_family_ids,
                max_depth=config.max_depth,
                max_nodes=config.max_explore_entities,
            )
        except Exception as exc:
            logger.warning("Dream: BFS遍历失败: %s", exc)
            bfs_entities = []

        # 构建 entity_lookup: family_id -> 简要信息
        entity_lookup: Dict[str, Dict[str, str]] = {}
        seen_ids: Set[str] = set()

        for ent in bfs_entities:
            fid = getattr(ent, 'family_id', None)
            if fid:
                entity_lookup[fid] = {
                    "family_id": fid,
                    "name": getattr(ent, 'name', ''),
                    "content": (getattr(ent, 'content', '') or '')[:500],
                }
                seen_ids.add(fid)

        # 补充种子自身
        for s in seeds:
            fid = s.get("family_id")
            if fid and fid not in entity_lookup:
                entity_lookup[fid] = {
                    "family_id": fid,
                    "name": s.get("name", ""),
                    "content": (s.get("content") or "")[:500],
                }
                seen_ids.add(fid)

        # 为每个种子构建 explored 信息
        explored: List[Dict[str, Any]] = []
        for seed in seeds:
            fid = seed.get("family_id")
            if not fid:
                continue
            neighbor_data = [
                {"family_id": eid, "name": info["name"], "content": info["content"][:200]}
                for eid, info in entity_lookup.items()
                if eid != fid
            ]
            explored.append({
                "seed": {"family_id": fid, "name": seed.get("name", "")},
                "neighbors": neighbor_data[:20],
                "neighbor_count": len(neighbor_data),
            })

        return entity_lookup, seen_ids, explored

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
    ) -> tuple:
        """并发发现实体间的隐含关联。

        Returns:
            (relations_created, pairs_checked)
        """
        # 收集所有待检查的配对（跳过历史已检查的）
        pairs: List[tuple] = []
        skipped_by_history = 0
        for exp in explored:
            seed_info = exp["seed"]
            seed_fid = seed_info["family_id"]
            seed_name = seed_info["name"]
            for neighbor in exp["neighbors"][:config.max_neighbors_per_seed]:
                nb_fid = neighbor["family_id"]
                if self._history.was_checked(seed_fid, nb_fid):
                    skipped_by_history += 1
                    continue
                pairs.append((seed_fid, seed_name, nb_fid, neighbor["name"]))

        if skipped_by_history:
            logger.info("Dream: 跳过 %d 对历史已检查的配对", skipped_by_history)

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

        # 使用线程池并发判断
        with ThreadPoolExecutor(max_workers=config.llm_concurrency) as executor:
            futures = {}
            for pair in pairs:
                seed_fid, seed_name, nb_fid, nb_name = pair
                future = executor.submit(
                    self._judge_pair,
                    seed_fid, seed_name, nb_fid, nb_name, config,
                    entity_lookup, existing_pairs,
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

    def _judge_pair(
        self,
        seed_fid: str,
        seed_name: str,
        nb_fid: str,
        nb_name: str,
        config: DreamConfig,
        entity_lookup: Optional[Dict[str, Dict[str, str]]] = None,
        existing_pairs: Optional[Set[Tuple[str, str]]] = None,
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
            seed_content = (seed_entity.content or "")[:500]
            nb_name = nb_entity.name
            nb_content = (nb_entity.content or "")[:500]

        # LLM 判断 + 生成（单次调用）
        judge_messages = [
            {"role": "system", "content": JUDGE_AND_GENERATE_RELATION_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"实体A: {seed_name}\n描述: {seed_content}\n\n"
                f"实体B: {nb_name}\n描述: {nb_content}\n\n"
                f"判断这两个实体之间是否存在明确的、有意义的关联。如果存在，同时生成关系描述。"
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

