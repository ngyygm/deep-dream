"""
Deep-Dream Core

唯一的代码目录，包含核心功能模块和共享基础设施：
- remember/ — 知识图谱构建管道
- find/ — 混合搜索引擎
- llm/ — LLM 客户端
- storage/ — 图存储（SQLite + hnswlib + FTS5）

This package uses PEP 562 lazy attribute access so that ``import core`` stays
cheap (no eager openai / sqlite / numpy pulls on the startup path). Symbols
are only imported on first attribute access via module-level ``__getattr__``.
Both ``from core import X`` and ``core.X`` keep working for every name below.
"""

# Static list of exported names. Kept eager so ``import core; core.__all__``
# works without triggering any heavy import.
__all__ = [
    "ContentPatch",
    "Episode",
    "Entity",
    "Relation",
    "SQLiteGraphStorageManager",
    "EmbeddingClient",
    "create_storage_manager",
    "LLMClient",
    "ollama_chat",
    "TemporalMemoryGraphProcessor",
]


# Map each exported name to a (module, attribute) pair, resolved lazily.
# Defined inside __getattr__ below to avoid importing the modules at definition
# time. Keeping it as a plain local keeps ``import core`` truly weightless.

def __getattr__(name):
    # source module -> sequence of attribute names it provides
    _lazy_sources = (
        (".models", ("ContentPatch", "Episode", "Entity", "Relation")),
        (".storage", ("EmbeddingClient", "create_storage_manager")),
        (".storage.sqlite", ("SQLiteGraphStorageManager",)),
        (".llm", ("LLMClient", "ollama_chat")),
        (".remember.orchestrator", ("TemporalMemoryGraphProcessor",)),
    )
    if name in __all__:
        import importlib
        for module_name, attrs in _lazy_sources:
            if name in attrs:
                module = importlib.import_module(module_name, __name__)
                value = getattr(module, name)
                # Cache on this module so subsequent lookups bypass __getattr__.
                globals()[name] = value
                return value
        # Defensive: listed in __all__ but not mapped above — fall through to error.
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
