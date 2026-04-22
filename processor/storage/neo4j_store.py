"""Backward compatibility shim — real code lives in processor/storage/neo4j/."""
from .neo4j import Neo4jStorageManager  # noqa: F401

__all__ = ["Neo4jStorageManager"]
