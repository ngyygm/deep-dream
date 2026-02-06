"""
Memory Retrieval Agent - 基于 Agent Skills 的记忆检索系统

使用 ReAct 循环（规划-执行-观察-判断）从时序记忆图谱中智能检索相关记忆

新增功能：
- ReasoningCache: 追踪推理状态，包括子目标、已知事实、缺失信息
- Reasoner: 分析问题类型，进行推理规划和结论生成  
- Summarizer: 筛选有用信息，生成推理总结
"""

from .orchestrator import MemoryRetrievalAgent
from .models import (
    AgentConfig,
    QueryResult,
    ToolCall,
    ToolResult,
    RetrievedMemory
)
from .context import ReasoningCache, QuestionType, GoalStatus
from .reasoner import Reasoner
from .summarizer import Summarizer

__all__ = [
    # 主要类
    "MemoryRetrievalAgent",
    "AgentConfig",
    "QueryResult",
    "ToolCall",
    "ToolResult",
    "RetrievedMemory",
    # 推理相关
    "ReasoningCache",
    "QuestionType",
    "GoalStatus",
    "Reasoner",
    "Summarizer"
]

__version__ = "0.2.0"
