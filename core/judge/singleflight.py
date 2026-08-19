"""同 key 单飞：并发场景下同一判断只执行一次，其余调用者共享结果。"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional, Tuple

_MISS = object()


class _Flight:
    __slots__ = ("event", "result", "error")

    def __init__(self):
        self.event = threading.Event()
        self.result: Any = None
        self.error: Optional[BaseException] = None


class SingleFlight:
    """execute(key, fn)：同 key 并发调用只有 leader 执行 fn，follower 等待结果。

    follower 等待到 leader 异常时返回 _MISS_SENTINEL（调用方自行回退直调），
    而不是抛出——单次失败不应让所有等待者一起失败。
    """

    MISS_SENTINEL = _MISS

    def __init__(self):
        self._lock = threading.Lock()
        self._flights: Dict[str, _Flight] = {}
        self._inflight = 0
        self._coalesced = 0

    def execute(self, key: str, fn: Callable[[], Any]) -> Any:
        with self._lock:
            flight = self._flights.get(key)
            if flight is not None:
                leader = False
                self._coalesced += 1
            else:
                flight = _Flight()
                self._flights[key] = flight
                leader = True
                self._inflight += 1
        if not leader:
            flight.event.wait()
            if flight.error is not None:
                return _MISS
            return flight.result
        try:
            result = fn()
        except BaseException as exc:  # noqa: BLE001 — follower 需要感知失败
            flight.error = exc
            flight.event.set()
            with self._lock:
                self._flights.pop(key, None)
                self._inflight -= 1
            raise
        flight.result = result
        flight.event.set()
        with self._lock:
            self._flights.pop(key, None)
            self._inflight -= 1
        return result

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"inflight": self._inflight, "coalesced": self._coalesced}
