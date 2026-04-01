"""DeepDream 数据模型。"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class DreamConfig:
    """梦境配置"""
    review_window_days: int = 30
    max_entities_per_cycle: int = 100
    similarity_threshold: float = 0.8
    max_new_connections: int = 20


@dataclass
class DreamReport:
    """梦境报告"""
    cycle_id: str
    graph_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    insights: List[dict] = field(default_factory=list)
    new_connections: List[dict] = field(default_factory=list)
    consolidations: List[dict] = field(default_factory=list)
    narrative: str = ""
    strategy: str = ""
    entities_examined: int = 0
    relations_created: int = 0
    episode_ids: List[str] = field(default_factory=list)


# ============================================================
# Dream Agent Models — 基于工具的自主梦境代理
# ============================================================

# 所有策略及其描述
DREAM_STRATEGIES = {
    "free_association": "随机选取 2-3 个实体，自由联想寻找隐藏连接",
    "contrastive": "选取语义相近的实体，发现差异和对比关系",
    "temporal_bridge": "选取时间跨度大的实体，寻找时间演变路径",
    "cross_domain": "跨社区选取实体，发现跨领域连接",
    "orphan_adoption": "选取无关系或极少关系的实体，为它们找到归属",
    "hub_remix": "选取高连接度实体，发现非常规路径",
    "leap": "随机实体出发，经过随机中转，发现创意跳跃连接",
    "narrative": "选取可构成故事线的实体，构建叙事连接",
}

VALID_DREAM_STRATEGIES = list(DREAM_STRATEGIES.keys())


@dataclass
class DreamAgentConfig:
    """Dream Agent 配置。"""
    graph_id: str = "default"
    max_cycles: int = 10
    strategies: List[str] = field(default_factory=lambda: [
        "free_association", "cross_domain", "leap",
    ])
    strategy_mode: str = "round_robin"  # round_robin | random | adaptive
    confidence_threshold: float = 0.6
    max_tool_calls_per_cycle: int = 15
    max_traverse_depth: int = 3
    seed_count: int = 5


@dataclass
class DreamToolCall:
    """一次工具调用。"""
    tool: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DreamActionResult:
    """Dream Agent 动作结果。"""
    success: bool
    data: Any = None
    error: Optional[str] = None


@dataclass
class DreamCycleResult:
    """单个策略周期的结果。"""
    strategy: str = ""
    entities_examined: int = 0
    relations_discovered: int = 0
    relations_saved: int = 0
    tool_calls_made: int = 0
    observations: List[str] = field(default_factory=list)
    proposed_relations: List[Dict[str, Any]] = field(default_factory=list)
    saved_relations: List[Dict[str, Any]] = field(default_factory=list)
    episode_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class DreamAgentState:
    """Dream Agent 运行时状态。"""
    session_id: str = ""
    graph_id: str = "default"
    current_cycle: int = 0
    total_entities_examined: int = 0
    total_relations_discovered: int = 0
    total_relations_saved: int = 0
    total_tool_calls: int = 0
    examined_entity_ids: set = field(default_factory=set)
    cycle_results: List[DreamCycleResult] = field(default_factory=list)
    status: str = "idle"  # idle | running | completed | failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    narrative: str = ""
