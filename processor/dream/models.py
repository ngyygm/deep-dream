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
