"""Shared helpers for SQLite graph storage."""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _parse_dt(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo:
            return value.astimezone(timezone.utc)
        return value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo:
                return dt.astimezone(timezone.utc)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            dt = datetime.fromisoformat(value.isoformat())
            if dt.tzinfo:
                return dt.astimezone(timezone.utc)
            return dt.replace(tzinfo=timezone.utc)
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


def _encode_and_normalize(embedding_client, text: str):
    """Encode text via embedding client, L2-normalize, return (bytes, ndarray) or None."""
    if not embedding_client or not embedding_client.is_available():
        return None
    embedding = embedding_client.encode(text)
    if embedding is None or (isinstance(embedding, (list, tuple)) and len(embedding) == 0):
        return None
    if isinstance(embedding, np.ndarray) and embedding.size == 0:
        return None
    emb_array = np.array(embedding[0] if isinstance(embedding, list) else embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(emb_array)
    if norm > 0:
        emb_array = emb_array / norm
    return emb_array.tobytes(), emb_array
