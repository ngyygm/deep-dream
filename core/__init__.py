"""
Deep-Dream Core

唯一的代码目录，包含三大功能模块和共享基础设施：
- remember/ — 知识图谱构建管道
- find/ — 混合搜索引擎
- dream/ — 隐含关系发现
- llm/ — LLM 客户端
- storage/ — 双后端存储 (SQLite/Neo4j)
"""

from .models import ContentPatch, Episode, Entity, Relation
from .storage import StorageManager
from .storage import EmbeddingClient
from .storage import create_storage_manager
from .llm import LLMClient
from .llm import ollama_chat
from .remember.orchestrator import TemporalMemoryGraphProcessor

__all__ = [
    "ContentPatch",
    "Episode",
    "Entity",
    "Relation",
    "StorageManager",
    "EmbeddingClient",
    "create_storage_manager",
    "LLMClient",
    "ollama_chat",
    "TemporalMemoryGraphProcessor",
]
