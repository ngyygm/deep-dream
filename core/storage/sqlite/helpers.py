"""Shared helpers for SQLite graph storage."""

import logging
import threading
import time
from datetime import datetime
from typing import Any, Optional

import numpy as np

from ...models import Entity, Relation

logger = logging.getLogger(__name__)


def _parse_dt(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return datetime.fromisoformat(value.isoformat()).replace(tzinfo=None)
        except Exception:
            pass
    return None


def _fmt_dt(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.replace(tzinfo=None).isoformat() if value.tzinfo else value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _row_to_entity(row: dict, _now: Optional[datetime] = None) -> Entity:
    """Convert a SQLite row dict to Entity dataclass."""
    if _now is None:
        _now = datetime.now()
    _emb = row.get("embedding")
    if isinstance(_emb, bytes) and len(_emb) > 0:
        pass  # already bytes
    elif isinstance(_emb, (list, np.ndarray)):
        _emb = np.array(_emb, dtype=np.float32).tobytes()
    else:
        _emb = None
    return Entity(
        absolute_id=row["uuid"],
        family_id=row["family_id"],
        name=row.get("name", ""),
        content=row.get("content", ""),
        event_time=_parse_dt(row.get("event_time")) or _now,
        processed_time=_parse_dt(row.get("processed_time")) or _now,
        episode_id=row.get("episode_id", ""),
        source_document=row.get("source_document") or "",
        embedding=_emb,
        valid_at=_parse_dt(row.get("valid_at")),
        invalid_at=_parse_dt(row.get("invalid_at")),
        summary=row.get("summary"),
        attributes=row.get("attributes"),
        confidence=float(row["confidence"]) if row.get("confidence") is not None else None,
        content_format=row.get("content_format", "plain"),
        community_id=row.get("community_id"),
    )


def _row_to_relation(row: dict, _now: Optional[datetime] = None) -> Relation:
    """Convert a SQLite row dict to Relation dataclass."""
    if _now is None:
        _now = datetime.now()
    _emb = row.get("embedding")
    if isinstance(_emb, bytes) and len(_emb) > 0:
        pass
    elif isinstance(_emb, (list, np.ndarray)):
        _emb = np.array(_emb, dtype=np.float32).tobytes()
    else:
        _emb = None
    return Relation(
        absolute_id=row["uuid"],
        family_id=row["family_id"],
        entity1_absolute_id=row.get("entity1_absolute_id", ""),
        entity2_absolute_id=row.get("entity2_absolute_id", ""),
        content=row.get("content", ""),
        event_time=_parse_dt(row.get("event_time")) or _now,
        processed_time=_parse_dt(row.get("processed_time")) or _now,
        episode_id=row.get("episode_id", ""),
        source_document=row.get("source_document") or "",
        embedding=_emb,
        valid_at=_parse_dt(row.get("valid_at")),
        invalid_at=_parse_dt(row.get("invalid_at")),
        summary=row.get("summary"),
        attributes=row.get("attributes"),
        confidence=float(row["confidence"]) if row.get("confidence") is not None else None,
        provenance=row.get("provenance"),
        content_format=row.get("content_format", "plain"),
    )


def _encode_and_normalize(embedding_client, text: str):
    """Encode text via embedding client, L2-normalize, return (bytes, ndarray) or None."""
    if not embedding_client or not embedding_client.is_available():
        return None
    embedding = embedding_client.encode(text)
    if embedding is None or (isinstance(embedding, (list, tuple)) and len(embedding) == 0):
        return None
    if isinstance(embedding, np.ndarray) and embedding.size == 0:
        return None
    emb_array = np.array(embedding[0] if isinstance(embedding, list) else embedding, dtype=np.float32)
    norm = np.linalg.norm(emb_array)
    if norm > 0:
        emb_array = emb_array / norm
    return emb_array.tobytes(), emb_array


# Cached datetime.now() refreshed every ~1s
_cached_now_time: float = 0.0
_cached_now_val: Optional[datetime] = None
_cached_now_lock = threading.Lock()


def _get_cached_now() -> datetime:
    global _cached_now_time, _cached_now_val
    _t = time.time()
    if _cached_now_val is None or (_t - _cached_now_time) > 1.0:
        _cached_now_val = datetime.now()
        _cached_now_time = _t
    return _cached_now_val


ENTITY_COLUMNS = [
    "uuid", "family_id", "graph_id", "name", "content", "summary",
    "attributes", "confidence", "content_format", "community_id",
    "valid_at", "invalid_at", "event_time", "processed_time",
    "episode_id", "source_document", "embedding",
]

RELATION_COLUMNS = [
    "uuid", "family_id", "graph_id",
    "entity1_absolute_id", "entity2_absolute_id",
    "entity1_family_id", "entity2_family_id",
    "content", "summary", "attributes", "confidence", "provenance",
    "content_format",
    "valid_at", "invalid_at", "event_time", "processed_time",
    "episode_id", "source_document", "embedding",
]

EPISODE_COLUMNS = [
    "uuid", "graph_id", "content", "source_text", "source_document",
    "event_time", "processed_time", "episode_type", "activity_type",
    "doc_hash", "created_at", "embedding",
]
