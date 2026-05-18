"""梦境编排器 — 种子选择、图探索、隐含关联发现。

将 api.py 中的 dream_run 逻辑提取为独立模块，便于：
- 独立测试
- 后台调度
- 可配置策略
- 并发 LLM 调用
"""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set

from core.dream._types import (
    DreamConfig,
    DreamHistory,
    DreamResult,
    VALID_STRATEGIES,
    _STRATEGY_DISPLAY_ORDER,
    _trunc,
)
from core.dream.dream_operations import DreamOperationsMixin
from core.find.graph_traversal import GraphTraversalSearcher

logger = logging.getLogger(__name__)

# Re-export for backward compatibility (consumed by __init__.py)
__all__ = ["DreamOrchestrator", "DreamConfig", "VALID_STRATEGIES"]


class DreamOrchestrator(DreamOperationsMixin):
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
        import time as _time
        cycle_start = _time.time()

        # Step 0: 策略轮换（如果启用）
        effective_strategy = config.strategy
        if auto_rotate:
            effective_strategy = self._history.next_strategy(config.strategy)
            logger.info("Dream: 策略轮换 → %s", effective_strategy)

        # 创建临时 config（不修改 self.config，保证线程安全）
        # Short-circuit when no rotation and no exclude_ids — config is identical
        _discovery_lower_conf = config.discovery_mode and config.min_confidence >= 0.5
        if not auto_rotate and not config.exclude_ids and not _discovery_lower_conf:
            run_config = config
        else:
            run_config = DreamConfig(
                strategy=effective_strategy,
                seed_count=config.seed_count,
                max_depth=config.max_depth,
                max_relations=config.max_relations,
                min_confidence=0.3 if _discovery_lower_conf else config.min_confidence,
                max_explore_entities=config.max_explore_entities,
                max_neighbors_per_seed=config.max_neighbors_per_seed,
                exclude_ids=list(config.exclude_ids) if config.exclude_ids else [],
                llm_timeout=config.llm_timeout,
                llm_concurrency=config.llm_concurrency,
                min_pair_similarity=config.min_pair_similarity,
                discovery_mode=config.discovery_mode,
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
        relations_created, pairs_checked, pair_errors = self._discover_relations(
            seeds, explored, entity_lookup, cycle_id, run_config,
            relation_context=relation_context,
        )

        # Step 4: 保存梦境记录
        cycle_summary = (
            f"梦境周期 {cycle_id}：策略={effective_strategy}，种子={len(seeds)}，"
            f"探索实体={len(seen_ids)}，检查配对={pairs_checked}，"
            f"创建关系={len(relations_created)}"
        )
        cycle_end = _time.time()
        self._save_episode(
            cycle_id, cycle_summary, seen_ids, relations_created, run_config,
            start_time=cycle_start, end_time=cycle_end,
        )

        # Step 5: 更新跨周期历史
        self._history.mark_explored(cycle_id, seen_ids)
        for r in relations_created:
            self._history.mark_checked(r["entity1_id"], r["entity2_id"], cycle_id)

        # Step 6: 记录策略效果
        self._history.record_strategy_result(effective_strategy, len(relations_created))

        # Detect potential LLM failure
        warnings = []
        if pairs_checked > 0 and len(relations_created) == 0 and pair_errors > 0:
            warnings.append(
                f"LLM may be unavailable: {pair_errors}/{pairs_checked} pair judgments failed. "
                "Use health_check_llm to verify."
            )

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
                "pair_errors": pair_errors,
                "strategy_rotated": auto_rotate,
            },
            cycle_summary=cycle_summary,
            warnings=warnings,
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
