"""梦境模块的共享类型、常量和工具函数。

被 orchestrator.py 和 dream_operations.py 共同引用，
避免循环导入。
"""

from collections import OrderedDict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


def _trunc(s: str, n: int = 200) -> str:
    """Truncate string only if needed — avoids slicing copy for short strings."""
    return s[:n] if len(s) > n else s


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
    discovery_mode: bool = False

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
    explored: List[Dict[str, object]]
    relations_created: List[Dict[str, object]]
    stats: Dict[str, object]
    cycle_summary: str
    warnings: List[str] = field(default_factory=list)
