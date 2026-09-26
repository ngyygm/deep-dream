"""Backward compatibility shim.

SQLiteGraphStorageManager is now LibraryManager (V1.5 schema).
All 5200 lines of the old manager have been replaced by:
  - library_manager.py   — facade class
  - dto_mapping.py       — V1.5 row → DTO mapping
  - graph_traversal.py   — BFS traversal
  - merge.py             — entity merge/redirect
  - vault_indexer.py     — vault file indexing
  - agent_query.py       — SQL sandbox
  - repositories/        — thin SQL functions
"""
from .library_manager import LibraryManager as SQLiteGraphStorageManager

__all__ = ["SQLiteGraphStorageManager"]
