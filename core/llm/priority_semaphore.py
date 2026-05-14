"""
Priority-based semaphore and rate-limit detection extracted from client.py.

PrioritySemaphore: a threading semaphore where lower priority number = higher priority.
_is_rate_limit_tpm_error: detects 429 / TPM / rate-limit errors for retry logic.
"""
import heapq
import threading

try:
    from openai import RateLimitError
except ImportError:  # pragma: no cover
    RateLimitError = None  # type: ignore[misc,assignment]

# Static error keywords — computed once at import time, not per-call
_RATE_LIMIT_KEYWORDS = ("rate_limit", "rate limit", "tpm", "throttl", "capacity", "overloaded")


def _is_rate_limit_tpm_error(exc: BaseException, _pre_lowered: str = None) -> bool:
    """429 / TPM / 速率限制：应长时间退避直至恢复，不计入普通重试上限。"""
    if RateLimitError is not None and isinstance(exc, RateLimitError):
        return True
    code = getattr(exc, "status_code", None)
    if code == 429:
        return True
    s = _pre_lowered if _pre_lowered is not None else str(exc).lower()
    # 检查 429 状态码相关字符串
    if "429" in s and ("error code" in s or "status code" in s):
        return True
    # 检查速率限制关键词（不依赖 429 状态码）
    return any(k in s for k in _RATE_LIMIT_KEYWORDS)


class PrioritySemaphore:
    """带优先级的信号量。priority 越小优先级越高，高优先级先获得锁。"""

    def __init__(self, value: int):
        if value < 1:
            raise ValueError("value must be >= 1")
        self._max_value = value
        self._value = value
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._heap: list = []  # [(priority, seq, event), ...]
        self._seq = 0

    @property
    def active_count(self) -> int:
        """当前正在使用的许可数（= 最大值 - 剩余值）。"""
        with self._lock:
            return self._max_value - self._value

    @property
    def max_value(self) -> int:
        return self._max_value

    def acquire(self, priority: int = 0):
        event = threading.Event()
        with self._cond:
            self._seq += 1
            if self._value > 0:
                self._value -= 1
                return
            heapq.heappush(self._heap, (priority, self._seq, event))
        event.wait()

    def release(self):
        with self._cond:
            while self._heap:
                _, _, event = heapq.heappop(self._heap)
                if not event.is_set():
                    event.set()
                    return
            self._value += 1
