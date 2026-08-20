"""
Deep-Dream Core

唯一的代码目录，包含核心功能模块和共享基础设施：
- remember/ — 知识图谱构建管道
- find/ — 混合搜索引擎
- llm/ — LLM 客户端
- storage/ — 图存储（SQLite + hnswlib + FTS5）

包级导出全部懒加载（PEP 562）：`import core` 只付包自身开销，
子模块在首次属性访问时才 import。此前急切导入使每次 CLI 调用
（哪怕 `deep-dream version`）都要加载 storage/llm/remember 全链
（实测 ~325ms）。`from core import Entity` 等用法语义不变。
"""

import importlib

# 名字 → (子模块, 属性)
_LAZY_EXPORTS = {
    "ContentPatch": ("core.models", "ContentPatch"),
    "Episode": ("core.models", "Episode"),
    "Entity": ("core.models", "Entity"),
    "Relation": ("core.models", "Relation"),
    "SQLiteGraphStorageManager": ("core.storage.sqlite", "SQLiteGraphStorageManager"),
    "EmbeddingClient": ("core.storage", "EmbeddingClient"),
    "create_storage_manager": ("core.storage", "create_storage_manager"),
    "LLMClient": ("core.llm", "LLMClient"),
    "ollama_chat": ("core.llm", "ollama_chat"),
    "TemporalMemoryGraphProcessor": ("core.remember.orchestrator", "TemporalMemoryGraphProcessor"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(target[0]), target[1])


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
