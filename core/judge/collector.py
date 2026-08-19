"""判断请求攒批执行器。

目标：同一时间窗（batch_delay_ms）内到达的多个判断请求一次放出执行，
避免每个请求各自排队等 LLM 信号量。串行场景退化为立即执行单个请求，
行为不变。
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BatchCollector:
    """submit(fn) -> result：阻塞直到该 fn 在某个批次中执行完成。

    实现：懒启动的 flusher 守护线程。队列非空后等 batch_delay_ms 攒批，
    取出至多 batch_max 项，交给小线程池并行执行（每项独立设置结果/异常）。
    """

    def __init__(self, *, batch_delay_ms: int = 200, batch_max: int = 32,
                 pool_size: int = 4):
        self._batch_delay = max(0, int(batch_delay_ms)) / 1000.0
        self._batch_max = max(1, int(batch_max))
        self._pool_size = max(1, int(pool_size))
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._queue: List[Tuple[threading.Event, Callable[[], Any], List]] = []
        self._worker: Optional[threading.Thread] = None
        self._pool: Optional[ThreadPoolExecutor] = None
        self._closed = False
        self._stats = {"batches": 0, "items": 0, "errors": 0}

    # ------------------------------------------------------------------
    def _ensure_worker(self) -> None:
        # 调用方持有 self._lock
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._flush_loop, name="judge-collector", daemon=True)
        self._worker.start()

    def submit(self, fn: Callable[[], Any]) -> Any:
        done = threading.Event()
        slot: List[Any] = []  # [result] 或 [("__error__", exc)]

        with self._not_empty:
            if self._closed:
                # 已关闭：直接同步执行，保证语义不丢
                return fn()
            self._queue.append((done, fn, slot))
            self._ensure_worker()
            self._not_empty.notify_all()
        done.wait()
        if slot and slot[0] == "__error__":
            raise slot[1]
        return slot[0]

    # ------------------------------------------------------------------
    def _flush_loop(self) -> None:
        while True:
            with self._not_empty:
                while not self._queue and not self._closed:
                    self._not_empty.wait(timeout=0.5)
                if self._closed and not self._queue:
                    return
                if not self._queue:
                    continue
            # 队列非空：等一个攒批窗口再取
            if self._batch_delay > 0:
                time.sleep(self._batch_delay)
            with self._not_empty:
                batch = self._queue[:self._batch_max]
                del self._queue[:len(batch)]
            if not batch:
                continue
            self._execute_batch(batch)

    def _execute_batch(self, batch) -> None:
        if self._pool is None:
            try:
                self._pool = ThreadPoolExecutor(
                    max_workers=self._pool_size, thread_name_prefix="judge-batch")
            except RuntimeError:
                self._pool = None
        with self._lock:
            self._stats["batches"] += 1
            self._stats["items"] += len(batch)

        def _run_one(item):
            done, fn, slot = item
            try:
                slot.append(fn())
            except BaseException as exc:  # noqa: BLE001
                slot.append("__error__")
                slot.append(exc)
                with self._lock:
                    self._stats["errors"] += 1
                logger.debug("judge batch item failed: %s", exc)
            finally:
                done.set()

        if self._pool is not None and len(batch) > 1:
            futures = [self._pool.submit(_run_one, item) for item in batch]
            for f in futures:
                try:
                    f.result()
                except Exception:
                    pass
        else:
            for item in batch:
                _run_one(item)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def close(self) -> None:
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=2.0)
        pool = self._pool
        self._pool = None
        if pool is not None:
            pool.shutdown(wait=False)
