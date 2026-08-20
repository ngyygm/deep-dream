"""Shared helpers for SQLite graph storage."""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def now_utc() -> datetime:
    """当前 UTC 时间（带时区）。DB 时间列的统一时间源。"""
    return datetime.now(timezone.utc)


def now_utc_str() -> str:
    """当前 UTC 时间的 ISO 字符串（+00:00 带时区）。

    created_at/processed_at/updated_at 等 DB 时间列的统一写入格式
    （P4.5 收敛：merge / library_manager / vault_indexer / db backfill
    共用此实现，避免混格式破坏 ORDER BY <时间列> 的字符串排序）。
    """
    return now_utc().isoformat()


def escape_like(value: str) -> str:
    """Escape LIKE wildcard characters (%_) so they match literally.

    Uses '!' as the ESCAPE character to avoid backslash quoting issues
    in Python triple-quoted SQL strings. 与 SQL 侧 ``ESCAPE '!'`` 子句
    成对使用（P4.5 收敛：library_manager / cmd_concept / cmd_find 共用）。
    """
    return value.replace("!", "!!").replace("%", "!%").replace("_", "!_")


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


def _time_bounds_sql(column: str, time_after=None, time_before=None):
    """构建 processed_at 双界（闭区间）过滤 SQL 片段与参数。

    P2.8：time_after/time_before 分别作为下/上界下推到 SQL，
    修复此前路由层把两界折叠成单个 time_point、静默丢弃另一界的问题。
    界值经 _fmt_dt 归一后与 ISO 字符串列直接可比
    （与 get_all_entities_before_time 同一约定）。
    """
    conds, params = [], []
    if time_after is not None:
        lo = _fmt_dt(time_after)
        if lo:
            conds.append(f"{column} >= ?")
            params.append(lo)
    if time_before is not None:
        hi = _fmt_dt(time_before)
        if hi:
            conds.append(f"{column} <= ?")
            params.append(hi)
    return (" AND " + " AND ".join(conds)) if conds else "", params


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
