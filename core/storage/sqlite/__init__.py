"""SQLite-based native graph storage engine for Deep Dream.

Replaces Neo4j with an embedded SQLite database + hnswlib for vector search.
All graph operations (CRUD, traversal, search) are implemented in pure Python + SQL.
"""

from .manager import SQLiteGraphStorageManager

__all__ = ["SQLiteGraphStorageManager"]
