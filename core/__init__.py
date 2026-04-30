"""
Deep-Dream Core

唯一的代码目录，包含三大功能模块和共享基础设施：
- remember/ — 知识图谱构建管道
- find/ — 混合搜索引擎
- dream/ — 隐含关系发现
- llm/ — LLM 客户端
- storage/ — Neo4j 图存储
"""

from .models import ContentPatch, Episode, Entity, Relation
from .storage.neo4j_store import Neo4jStorageManager
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
    "Neo4jStorageManager",
    "EmbeddingClient",
    "create_storage_manager",
    "LLMClient",
    "ollama_chat",
    "TemporalMemoryGraphProcessor",
]
