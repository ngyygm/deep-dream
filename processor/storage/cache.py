"""
TTL 查询缓存，避免短时间内重复查询。
"""
import threading
import time
from typing import Any, Optional


class QueryCache:
    """线程安全的 TTL 缓存，支持容量上限和惰性淘汰。"""

    def __init__(self, default_ttl: float = 30, max_size: int = 4096):
        self._cache: dict[str, tuple[float, float, Any]] = {}  # key -> (expire_time, insert_time, value)
        self._default_ttl = default_ttl
        self._max_size = max_size
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
            if len(self._cache) >= self._max_size:
                self._evict_locked()
            self._cache[key] = (time.monotonic() + ttl, time.monotonic(), value)

    def invalidate(self, pattern: Optional[str] = None):
        with self._lock:
            if pattern is None:
                self._cache.clear()
            else:
                self._cache = {k: v for k, v in self._cache.items() if pattern not in k}

    def _evict_locked(self):
        """当缓存满时，淘汰已过期条目或最旧的条目（调用时已持有锁）。"""
        now = time.monotonic()
        # 先清除过期条目
        expired = [k for k, (exp, _, _) in self._cache.items() if now > exp]
        for k in expired:
            del self._cache[k]
        # 如果仍然满了，按插入时间淘汰最旧的10%
        if len(self._cache) >= self._max_size:
            sorted_keys = sorted(self._cache, key=lambda k: self._cache[k][1])
            remove_count = max(1, self._max_size // 10)
            for k in sorted_keys[:remove_count]:
                del self._cache[k]

    def size(self) -> int:
        """返回当前缓存条目数。"""
        with self._lock:
            return len(self._cache)
