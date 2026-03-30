"""
TTL 查询缓存，避免短时间内重复查询。
"""
import threading
import time
from typing import Any, Optional


class QueryCache:
    """线程安全的 TTL 缓存。"""

    def __init__(self, default_ttl: float = 30):
        self._cache: dict[str, tuple[float, float, Any]] = {}  # key -> (expire_time, insert_time, value)
        self._default_ttl = default_ttl
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            expire_at, _, value = entry
            if time.monotonic() > expire_at:
                del self._cache[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        with self._lock:
            ttl = ttl if ttl is not None else self._default_ttl
            self._cache[key] = (time.monotonic() + ttl, time.monotonic(), value)

    def invalidate(self, pattern: Optional[str] = None):
        with self._lock:
            if pattern is None:
                self._cache.clear()
            else:
                self._cache = {k: v for k, v in self._cache.items() if pattern not in k}
