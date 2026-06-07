"""
Tests for progress callbacks in task_worker.py.

The worker_loop creates closures (_on_main_chunk_done, _on_step9_chunk_done,
_on_chunk_done, _mark_task_running) that call q._update_task_progress and
q._persist.  We extract the closure-creation logic into testable helpers by
re-creating the same closure pattern and mocking the queue methods.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call

import pytest

# ---------------------------------------------------------------------------
# Minimal stand-in for RememberTask (avoids importing the full module which
# pulls in Flask / SQLite etc.).  We only need the progress-tracking fields.
# ---------------------------------------------------------------------------

@dataclass
class FakeTask:
    task_id: str = "test-001"
    text: str = ""
    source_name: str = "test"
    status: str = "queued"
    total_chunks: int = 0
    main_done_chunks: int = 0
    step9_done_chunks: int = 0
    step10_done_chunks: int = 0
    processed_chunks: int = 0
    progress: float = 0.0
    main_progress: float = 0.0
    main_label: str = ""
    step9_progress: float = 0.0
    step9_label: str = ""
    step10_progress: float = 0.0
    step10_label: str = ""
    failed_window_indices: List[int] = field(default_factory=list)
    phase: str = "queued"
    phase_label: str = ""
    phase_current: int = 0
    phase_total: int = 0
    message: str = ""
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    run_start_chunks: int = 0
    chain_started_at: Dict[str, float] = field(default_factory=dict)
    chain_run_start_chunks: Dict[str, int] = field(default_factory=dict)
    control_action: Optional[str] = None
    original_path: str = ""
    cache_document_path: Optional[str] = None
    override_doc_id: str = ""
    load_cache: Optional[bool] = None
    event_time = None
    result: Optional[Dict] = None
    max_retries: int = 3
    retry_attempt: int = 0
    last_update: float = 0.0


# ---------------------------------------------------------------------------
# Fake queue that records calls to _update_task_progress and _persist.
# The real RememberTaskQueue._update_task_progress writes fields onto the task
# object; our fake does the same so closures see the updated state.
# ---------------------------------------------------------------------------

class FakeQueue:
    def __init__(self):
        self._lock = threading.Lock()
        self.progress_calls: List[Dict[str, Any]] = []
        self.persist_calls: List[FakeTask] = []

    def _update_task_progress(self, task, **kwargs):
        """Mimic the real method: store kwargs on the task and record the call."""
        # Mirror the real implementation's field-writing logic.
        for key, value in kwargs.items():
            if value is not None:
                setattr(task, key, value)
        self.progress_calls.append(kwargs)

    def _persist(self, task):
        self.persist_calls.append(task)


# ---------------------------------------------------------------------------
# Closure builders -- extracted from worker_loop's body (lines 304-361)
# so we can test them in isolation without running the full worker.
# ---------------------------------------------------------------------------

def build_closures(
    task: FakeTask,
    q: FakeQueue,
    *,
    is_targeted_retry: bool = False,
    target_indices: Optional[List[int]] = None,
):
    """Re-create the four closures exactly as worker_loop does."""
    _is_targeted_retry = is_targeted_retry
    _target_indices = sorted(target_indices) if target_indices else None

    def _remap_targeted_progress(processed_count, _t=task):
        _tc = max(1, int(_t.total_chunks or 1))
        _pc = max(0, int(processed_count))
        if _is_targeted_retry:
            _n_done = sum(1 for idx in _target_indices if idx + 1 <= _pc)
            _pc = (_tc - len(_target_indices)) + _n_done
            _pc = min(_tc, max(0, _pc))
        return _tc, _pc

    # Imported from task_progress -- used in _on_main_chunk_done
    import re
    _RE_MAIN_1_8_DONE = re.compile(r"步骤\s*1\s*[–-]\s*8\s*/\s*10")

    def _on_main_chunk_done(processed_count: int, _t=task):
        _tc, _pc = _remap_targeted_progress(processed_count, _t)
        _pg = min(1.0, float(_pc) / float(_tc))
        _ml = _t.main_label or ""
        if _pc >= _tc and not _RE_MAIN_1_8_DONE.search(_ml):
            _ml = "步骤1–8/10 已完成"
        q._update_task_progress(
            _t,
            main_done_chunks=max(_pc, int(_t.main_done_chunks or 0)),
            main_progress=max(_pg, float(_t.main_progress or 0.0)),
            main_label=_ml,
        )
        q._persist(_t)

    def _on_step9_chunk_done(processed_count: int, _t=task):
        _tc, _pc = _remap_targeted_progress(processed_count, _t)
        _pg = min(1.0, float(_pc) / float(_tc))
        q._update_task_progress(
            _t,
            step9_done_chunks=max(_pc, int(_t.step9_done_chunks or 0)),
            step9_progress=max(_pg, float(_t.step9_progress or 0.0)),
        )
        q._persist(_t)

    def _on_chunk_done(processed_count: int, _t=task):
        _tc, _pc = _remap_targeted_progress(processed_count, _t)
        _pg = min(1.0, float(_pc) / float(_tc))
        q._update_task_progress(
            _t,
            step10_done_chunks=max(_pc, int(_t.step10_done_chunks or 0)),
            processed_chunks=max(_pc, int(_t.processed_chunks or 0)),
            progress=max(_pg, float(_t.progress or 0.0)),
            step10_progress=max(_pg, float(_t.step10_progress or 0.0)),
        )
        q._persist(_t)

    return _on_main_chunk_done, _on_step9_chunk_done, _on_chunk_done


def build_mark_task_running(
    task: FakeTask,
    q: FakeQueue,
    *,
    existing_main_chunks: int = 0,
    existing_step9_chunks: int = 0,
    existing_step10_chunks: int = 0,
    init_progress: float = 0.0,
    resume_hint: str = "开始处理",
    is_targeted_retry: bool = False,
    target_indices: Optional[List[int]] = None,
):
    """Re-create _mark_task_running as a callable."""
    _is_targeted_retry = is_targeted_retry
    _target_indices = sorted(target_indices) if target_indices else None

    def _mark_task_running():
        task.chain_started_at = {}
        task.chain_run_start_chunks = {
            "main": int(existing_main_chunks or 0),
            "step9": int(existing_step9_chunks or 0),
            "step10": int(existing_step10_chunks or 0),
        }
        _phase_label = resume_hint
        if _is_targeted_retry:
            _phase_label = f"补跑 {len(_target_indices)} 个缺失/失败窗口"
        _tc = max(1, task.total_chunks)
        q._update_task_progress(
            task,
            status="running",
            phase="processing",
            phase_label=_phase_label,
            phase_current=existing_step10_chunks,
            phase_total=_tc,
            main_done_chunks=existing_main_chunks,
            step9_done_chunks=existing_step9_chunks,
            step10_done_chunks=existing_step10_chunks,
            processed_chunks=existing_step10_chunks,
            total_chunks=task.total_chunks,
            run_start_chunks=existing_step10_chunks,
            progress=init_progress,
            step9_progress=(existing_step9_chunks / _tc) if existing_step9_chunks else 0.0,
            step9_label="",
            step10_progress=(existing_step10_chunks / _tc) if existing_step10_chunks else 0.0,
            step10_label="",
            main_progress=(existing_main_chunks / _tc) if existing_main_chunks else 0.0,
            main_label="",
            message=resume_hint,
            started_at=1000.0,
            finished_at=None,
            error=None,
        )
        q._persist(task)

    return _mark_task_running


# ===================================================================
# Test 1: Normal callback
# ===================================================================

class TestNormalCallback:
    """_on_main_chunk_done with total=10, processed_count=5."""

    def test_sets_main_done_chunks(self):
        task = FakeTask(total_chunks=10)
        q = FakeQueue()
        on_main, _, _ = build_closures(task, q)

        on_main(processed_count=5)

        assert task.main_done_chunks == 5

    def test_sets_main_progress(self):
        task = FakeTask(total_chunks=10)
        q = FakeQueue()
        on_main, _, _ = build_closures(task, q)

        on_main(processed_count=5)

        assert task.main_progress == pytest.approx(0.5, abs=1e-6)

    def test_persist_called(self):
        task = FakeTask(total_chunks=10)
        q = FakeQueue()
        on_main, _, _ = build_closures(task, q)

        on_main(processed_count=5)

        assert len(q.persist_calls) == 1
        assert q.persist_calls[0] is task


# ===================================================================
# Test 2: Targeted retry callback
# ===================================================================

class TestTargetedRetryCallback:
    """With _is_targeted_retry=True, _target_indices=[5,13,22], total=30.

    The formula in _remap_targeted_progress is:
        _pc = (total - len(target_indices)) + n_done_target_windows
    where n_done_target_windows = number of target indices whose 1-based
    position (idx + 1) <= processed_count.
    """

    def setup_method(self):
        self.task = FakeTask(total_chunks=30)
        self.q = FakeQueue()
        self.on_main, self.on_step9, self.on_chunk = build_closures(
            self.task,
            self.q,
            is_targeted_retry=True,
            target_indices=[5, 13, 22],
        )

    def test_window_5_gives_pc_28(self):
        """Completing window 5 (processed_count=6): _pc = (30-3) + 1 = 28."""
        self.on_main(processed_count=6)
        assert self.task.main_done_chunks == 28
        assert self.task.main_progress == pytest.approx(28 / 30, abs=1e-6)

    def test_window_13_gives_pc_29(self):
        """Completing window 13 (processed_count=14): _pc = (30-3) + 2 = 29."""
        self.on_main(processed_count=14)
        assert self.task.main_done_chunks == 29
        assert self.task.main_progress == pytest.approx(29 / 30, abs=1e-6)

    def test_window_22_gives_pc_30(self):
        """Completing window 22 (processed_count=23): _pc = (30-3) + 3 = 30."""
        self.on_main(processed_count=23)
        assert self.task.main_done_chunks == 30
        assert self.task.main_progress == pytest.approx(1.0, abs=1e-6)

    def test_non_target_window_does_not_advance(self):
        """processed_count=3 (before first target): _pc = (30-3) + 0 = 27."""
        self.on_main(processed_count=3)
        assert self.task.main_done_chunks == 27

    def test_step9_uses_same_remap(self):
        self.on_step9(processed_count=23)
        assert self.task.step9_done_chunks == 30
        assert self.task.step9_progress == pytest.approx(1.0, abs=1e-6)

    def test_chunk_done_uses_same_remap(self):
        self.on_chunk(processed_count=14)
        assert self.task.step10_done_chunks == 29
        assert self.task.processed_chunks == 29
        assert self.task.progress == pytest.approx(29 / 30, abs=1e-6)
        assert self.task.step10_progress == pytest.approx(29 / 30, abs=1e-6)


# ===================================================================
# Test 3: Monotonic progress
# ===================================================================

class TestMonotonicProgress:
    """Calling with a lower value should not decrease main_done_chunks."""

    def test_main_done_chunks_never_decreases(self):
        task = FakeTask(total_chunks=10)
        q = FakeQueue()
        on_main, _, _ = build_closures(task, q)

        on_main(processed_count=7)
        assert task.main_done_chunks == 7

        # Simulate out-of-order callback with lower value
        on_main(processed_count=3)
        assert task.main_done_chunks == 7  # still 7, not 3

    def test_main_progress_never_decreases(self):
        task = FakeTask(total_chunks=10)
        q = FakeQueue()
        on_main, _, _ = build_closures(task, q)

        on_main(processed_count=8)
        p_after_8 = task.main_progress

        on_main(processed_count=4)
        assert task.main_progress >= p_after_8

    def test_step10_monotonic(self):
        task = FakeTask(total_chunks=10)
        q = FakeQueue()
        _, _, on_chunk = build_closures(task, q)

        on_chunk(processed_count=9)
        assert task.step10_done_chunks == 9

        on_chunk(processed_count=2)
        assert task.step10_done_chunks == 9  # does not go down

    def test_step9_monotonic(self):
        task = FakeTask(total_chunks=10)
        q = FakeQueue()
        _, on_step9, _ = build_closures(task, q)

        on_step9(processed_count=6)
        assert task.step9_done_chunks == 6

        on_step9(processed_count=1)
        assert task.step9_done_chunks == 6


# ===================================================================
# Test 4: step9 / step10 progress floats
# ===================================================================

class TestStep9Step10ProgressFloats:
    """_on_step9_chunk_done updates step9_progress; _on_chunk_done updates step10_progress."""

    def test_step9_callback_sets_step9_progress(self):
        task = FakeTask(total_chunks=20)
        q = FakeQueue()
        _, on_step9, _ = build_closures(task, q)

        on_step9(processed_count=10)
        assert task.step9_progress == pytest.approx(0.5, abs=1e-6)
        # step10_progress should remain untouched (never set by step9 callback)
        assert task.step10_progress == 0.0

    def test_chunk_done_sets_step10_progress(self):
        task = FakeTask(total_chunks=20)
        q = FakeQueue()
        _, _, on_chunk = build_closures(task, q)

        on_chunk(processed_count=15)
        assert task.step10_progress == pytest.approx(0.75, abs=1e-6)

    def test_chunk_done_sets_overall_progress(self):
        task = FakeTask(total_chunks=20)
        q = FakeQueue()
        _, _, on_chunk = build_closures(task, q)

        on_chunk(processed_count=12)
        assert task.progress == pytest.approx(12 / 20, abs=1e-6)

    def test_step9_does_not_touch_step10_fields(self):
        task = FakeTask(total_chunks=10)
        q = FakeQueue()
        _, on_step9, _ = build_closures(task, q)

        on_step9(processed_count=5)
        assert task.step10_done_chunks == 0
        assert task.step10_progress == 0.0
        assert task.step10_label == ""

    def test_chunk_done_does_not_touch_step9_fields(self):
        task = FakeTask(total_chunks=10)
        q = FakeQueue()
        _, _, on_chunk = build_closures(task, q)

        on_chunk(processed_count=5)
        # step9 fields are not written by _on_chunk_done
        assert task.step9_done_chunks == 0
        assert task.step9_progress == 0.0

    def test_independent_chains_different_progress(self):
        """step9 and step10 can have different progress values independently."""
        task = FakeTask(total_chunks=10)
        q = FakeQueue()
        _, on_step9, on_chunk = build_closures(task, q)

        on_step9(processed_count=8)
        assert task.step9_progress == pytest.approx(0.8, abs=1e-6)

        on_chunk(processed_count=3)
        assert task.step10_progress == pytest.approx(0.3, abs=1e-6)
        # step9 should still be at 0.8
        assert task.step9_progress == pytest.approx(0.8, abs=1e-6)


# ===================================================================
# Test 5: _mark_task_running with existing chunks
# ===================================================================

class TestMarkTaskRunningInitialProgress:
    """_mark_task_running with existing chunks should set correct initial progress floats."""

    def test_resume_sets_all_progress_floats(self):
        task = FakeTask(total_chunks=20)
        q = FakeQueue()
        mark = build_mark_task_running(
            task,
            q,
            existing_main_chunks=10,
            existing_step9_chunks=8,
            existing_step10_chunks=6,
            init_progress=6 / 20,
        )
        mark()

        assert task.main_progress == pytest.approx(10 / 20, abs=1e-6)
        assert task.step9_progress == pytest.approx(8 / 20, abs=1e-6)
        assert task.step10_progress == pytest.approx(6 / 20, abs=1e-6)

    def test_resume_sets_done_chunk_counts(self):
        task = FakeTask(total_chunks=20)
        q = FakeQueue()
        mark = build_mark_task_running(
            task,
            q,
            existing_main_chunks=12,
            existing_step9_chunks=10,
            existing_step10_chunks=8,
            init_progress=8 / 20,
        )
        mark()

        assert task.main_done_chunks == 12
        assert task.step9_done_chunks == 10
        assert task.step10_done_chunks == 8
        assert task.processed_chunks == 8

    def test_resume_from_zero(self):
        task = FakeTask(total_chunks=10)
        q = FakeQueue()
        mark = build_mark_task_running(task, q)
        mark()

        assert task.main_progress == 0.0
        assert task.step9_progress == 0.0
        assert task.step10_progress == 0.0
        assert task.main_done_chunks == 0
        assert task.step9_done_chunks == 0
        assert task.step10_done_chunks == 0

    def test_resume_status_running(self):
        task = FakeTask(total_chunks=10)
        q = FakeQueue()
        mark = build_mark_task_running(task, q)
        mark()

        assert task.status == "running"
        assert task.phase == "processing"

    def test_resume_chain_run_start_chunks(self):
        task = FakeTask(total_chunks=20)
        q = FakeQueue()
        mark = build_mark_task_running(
            task,
            q,
            existing_main_chunks=5,
            existing_step9_chunks=4,
            existing_step10_chunks=3,
        )
        mark()

        assert task.chain_run_start_chunks == {
            "main": 5,
            "step9": 4,
            "step10": 3,
        }

    def test_targeted_retry_progress(self):
        """Targeted retry init progress = (total - num_targets) / total."""
        task = FakeTask(total_chunks=30)
        q = FakeQueue()
        mark = build_mark_task_running(
            task,
            q,
            existing_main_chunks=27,
            existing_step9_chunks=27,
            existing_step10_chunks=27,
            init_progress=(30 - 3) / 30,
            is_targeted_retry=True,
            target_indices=[5, 13, 22],
        )
        mark()

        assert task.progress == pytest.approx(27 / 30, abs=1e-6)
        assert task.main_done_chunks == 27
        assert task.step9_done_chunks == 27
        assert task.step10_done_chunks == 27

    def test_persist_called(self):
        task = FakeTask(total_chunks=10)
        q = FakeQueue()
        mark = build_mark_task_running(task, q)
        mark()

        assert len(q.persist_calls) == 1
        assert q.persist_calls[0] is task


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_total_chunks_one(self):
        task = FakeTask(total_chunks=1)
        q = FakeQueue()
        on_main, _, _ = build_closures(task, q)
        on_main(processed_count=1)
        assert task.main_done_chunks == 1
        assert task.main_progress == pytest.approx(1.0, abs=1e-6)

    def test_zero_total_chunks_safe(self):
        """Division by zero should be guarded (tc defaults to max(1, 0) = 1)."""
        task = FakeTask(total_chunks=0)
        q = FakeQueue()
        on_main, _, _ = build_closures(task, q)
        # Should not raise
        on_main(processed_count=1)
        assert task.main_done_chunks == 1
        assert task.main_progress == pytest.approx(1.0, abs=1e-6)

    def test_processed_count_exceeds_total_clamped(self):
        """processed_count > total_chunks clamps progress to 1.0 but
        main_done_chunks reflects the raw value (the source uses max(_pc, ...)
        without a min(tc, ...) clamp in the non-targeted path)."""
        task = FakeTask(total_chunks=5)
        q = FakeQueue()
        on_main, _, _ = build_closures(task, q)
        on_main(processed_count=100)
        assert task.main_progress == pytest.approx(1.0, abs=1e-6)
        # Source does NOT clamp _pc to _tc for non-targeted path; it flows
        # through as max(processed_count, 0) = 100.
        assert task.main_done_chunks == 100

    def test_main_label_set_on_completion(self):
        """When _pc >= _tc, main_label should be set to completion message."""
        task = FakeTask(total_chunks=3)
        q = FakeQueue()
        on_main, _, _ = build_closures(task, q)
        on_main(processed_count=3)
        assert "步骤1" in task.main_label
        assert "8" in task.main_label
        assert "已完成" in task.main_label


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
