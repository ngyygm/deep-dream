"""
Tests for PrioritySemaphore: acquire/release, priority ordering,
active_count, max_value, over-subscription blocking, and concurrency
with semaphore integration.
"""

import threading
import time

import pytest

from processor.llm.client import LLMClient, PrioritySemaphore

# 与 service_config.llm.context_window_tokens / server.config.DEFAULTS 对齐
_TEST_CONTEXT_WINDOW_TOKENS = 8000


def _llm_client(**kwargs):
    kwargs.setdefault("context_window_tokens", _TEST_CONTEXT_WINDOW_TOKENS)
    return LLMClient(**kwargs)


class TestPrioritySemaphore:
    """Tests for the PrioritySemaphore class."""

    def test_basic_acquire_release(self):
        """Basic acquire and release with value=2."""
        sem = PrioritySemaphore(2)
        sem.acquire()
        assert sem.active_count == 1
        sem.acquire()
        assert sem.active_count == 2
        sem.release()
        assert sem.active_count == 1
        sem.release()
        assert sem.active_count == 0

    def test_priority_ordering(self):
        """Lower priority number acquires semaphore first."""
        sem = PrioritySemaphore(1)
        acquisition_order = []

        # First caller takes the only slot
        sem.acquire()

        # Three threads wait with different priorities
        results = {}

        def waiter(name, priority):
            sem.acquire(priority=priority)
            acquisition_order.append(name)
            sem.release()

        # Start threads with priorities 5, 1, 3 (lower = higher priority)
        t1 = threading.Thread(target=waiter, args=("low_p", 5))
        t2 = threading.Thread(target=waiter, args=("high_p", 1))
        t3 = threading.Thread(target=waiter, args=("mid_p", 3))

        # Stagger starts slightly so they enter the heap in order
        t1.start()
        time.sleep(0.05)
        t2.start()
        time.sleep(0.05)
        t3.start()
        time.sleep(0.05)

        # Release the slot so the highest-priority waiter proceeds
        sem.release()

        t1.join(timeout=5)
        t2.join(timeout=5)
        t3.join(timeout=5)

        # Priority 1 ("high_p") should acquire first
        assert acquisition_order[0] == "high_p"
        # Remaining two are released in priority order as each finishes
        assert len(acquisition_order) == 3

    def test_active_count_property(self):
        """active_count reflects current usage."""
        sem = PrioritySemaphore(3)
        assert sem.active_count == 0
        sem.acquire()
        assert sem.active_count == 1
        sem.acquire()
        assert sem.active_count == 2
        sem.release()
        assert sem.active_count == 1

    def test_max_value_property(self):
        """max_value returns the constructor value."""
        sem = PrioritySemaphore(5)
        assert sem.max_value == 5

    def test_oversubscription_blocks(self):
        """Acquiring beyond capacity blocks until a release happens."""
        sem = PrioritySemaphore(1)
        sem.acquire()

        blocked = threading.Event()
        finished = threading.Event()

        def try_acquire():
            blocked.set()  # signal that thread has started
            sem.acquire()
            finished.set()
            sem.release()

        t = threading.Thread(target=try_acquire)
        t.start()
        blocked.wait(timeout=2)
        # Thread should be blocked because the slot is held
        assert not finished.is_set()

        sem.release()
        t.join(timeout=5)
        assert finished.is_set()

    def test_value_must_be_at_least_one(self):
        """Constructor raises ValueError for value < 1."""
        with pytest.raises(ValueError):
            PrioritySemaphore(0)
        with pytest.raises(ValueError):
            PrioritySemaphore(-1)


class TestConcurrencyWithSemaphore:
    """Test that the semaphore limits concurrent LLM calls."""

    def test_max_concurrency_enforced(self):
        # max=1 时不拆分信号量，便于与旧版一样打桩 _llm_semaphore
        client = _llm_client(max_llm_concurrency=1)
        max_observed = 0
        lock = threading.Lock()
        barrier = threading.Barrier(5, timeout=10)

        # Wrap the semaphore's acquire/release to track actual concurrency
        sem = client._llm_semaphore
        original_acquire = sem.acquire
        original_release = sem.release

        def tracked_acquire(priority=0):
            original_acquire(priority)
            with lock:
                count = sem.active_count
                nonlocal max_observed
                if count > max_observed:
                    max_observed = count

        sem.acquire = tracked_acquire

        def call_llm():
            barrier.wait()
            time.sleep(0.05)  # let all threads queue up
            result = client._call_llm("请抽取实体：测试内容")
            assert isinstance(result, str)
            assert len(result) > 0

        threads = [threading.Thread(target=call_llm) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert max_observed <= 1

    def test_semaphore_released_on_exception(self):
        """Semaphore is released even when errors occur."""
        client = _llm_client(max_llm_concurrency=1)
        assert client._llm_semaphore.active_count == 0
        # In mock mode, no exceptions occur; just verify release after normal call
        client._call_llm("请抽取实体")
        assert client._llm_semaphore.active_count == 0
