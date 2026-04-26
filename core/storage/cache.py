"""
TTL 查询缓存，避免短时间内重复查询。
"""
import collections
import threading
import time
from typing import Any, Dict, Optional


class QueryCache:
    """线程安全的 TTL 缓存，支持容量上限和惰性淘汰。

    使用 OrderedDict 维护插入顺序，淘汰时从头部弹出（O(1)），
    避免排序整个缓存（O(n log n)）。
    """

    def __init__(self, default_ttl: float = 30, max_size: int = 4096):
        self._cache: collections.OrderedDict[str, tuple[float, Any]] = collections.OrderedDict()
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._lock = threading.Lock()
        # Statistics counters (atomic via lock)
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._sets: int = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            expire_at, value = entry
            if time.monotonic() > expire_at:
                del self._cache[key]
                self._evictions += 1
                self._misses += 1
                return None
            # Move to end (most recently accessed)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        with self._lock:
            ttl = ttl if ttl is not None else self._default_ttl
            if len(self._cache) >= self._max_size:
                self._evict_locked()
            self._cache[key] = (time.monotonic() + ttl, value)
            self._cache.move_to_end(key)
            self._sets += 1

    def invalidate(self, pattern: Optional[str] = None):
        with self._lock:
            if pattern is None:
                self._cache.clear()
            else:
                # Collect matching keys then delete (can't mutate OrderedDict during iteration)
                for k in [k for k in self._cache if pattern in k]:
                    self._cache.pop(k, None)

    def invalidate_keys(self, keys: list[str]):
        """Remove specific cache keys (O(1) per key). Prefer over pattern invalidate."""
        with self._lock:
            for k in keys:
                self._cache.pop(k, None)

    def _evict_locked(self):
        """当缓存满时，淘汰已过期条目或最旧的条目（调用时已持有锁）。

        OrderedDict 保证插入/访问顺序，从头部弹出即可淘汰最旧条目（O(1)）。
        """
        now = time.monotonic()
        # 先清除过期条目（从前部扫描，遇到未过期即停止）
        while self._cache:
            key = next(iter(self._cache))
            expire_at, _ = self._cache[key]
            if now > expire_at:
                self._cache.popitem(last=False)
                self._evictions += 1
            else:
                break
        # 如果仍然满了，按插入时间淘汰最旧的10%（从头部弹出，O(1) each）
        if len(self._cache) >= self._max_size:
            remove_count = max(1, self._max_size // 10)
            for _ in range(remove_count):
                if self._cache:
                    self._cache.popitem(last=False)
                    self._evictions += 1

    def size(self) -> int:
        """返回当前缓存条目数。"""
        with self._lock:
            return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        """返回缓存统计信息。"""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "sets": self._sets,
                "evictions": self._evictions,
                "default_ttl": self._default_ttl,
            }
