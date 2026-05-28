"""SQLite-based native graph storage engine for Deep Dream.

V1.5 LibraryManager replaces the old SQLiteGraphStorageManager.
"""
from .library_manager import LibraryManager

# Backward compat: old name still importable
SQLiteGraphStorageManager = LibraryManager

__all__ = ["LibraryManager", "SQLiteGraphStorageManager"]
