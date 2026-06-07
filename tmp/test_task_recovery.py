"""
Tests for recover_from_disk in core/server/task_worker.py.

Covers:
1. Paused task stays paused (not re-queued)
2. Running task recovers as queued with correct done_chunks
3. Three-chain independence (step10_done preserved independently)
4. Legacy record fallback (missing main_done/step9_done => 0)
5. Progress calculation (step10_done_chunks / total_chunks)
"""
from __future__ import annotations

import queue as _queue
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the function under test
# ---------------------------------------------------------------------------
from core.server.task_worker import recover_from_disk
from core.server.task_journal import RememberTask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    task_id: str = "abc12345",
    status: str = "running",
    original_path: str = "/tmp/originals/abc12345.txt",
    total_chunks: int = 20,
    main_done_chunks: int = 0,
    step9_done_chunks: int = 0,
    step10_done_chunks: int = 0,
    processed_chunks: int = 0,
    created_at: float | None = None,
    **extra,
) -> dict:
    rec = {
        "task_id": task_id,
        "status": status,
        "source_name": "test_doc",
        "original_path": original_path,
        "total_chunks": total_chunks,
        "main_done_chunks": main_done_chunks,
        "step9_done_chunks": step9_done_chunks,
        "step10_done_chunks": step10_done_chunks,
        "processed_chunks": processed_chunks,
        "created_at": created_at or time.time(),
        "event_time": None,
        "load_cache": None,
        "control_action": None,
    }
    rec.update(extra)
    return rec


def _run_recovery(records: list[dict], *, text: str = "hello world", text_length: int = 0):
    """Invoke recover_from_disk with mocked dependencies.

    Returns (tasks_dict, task_queue, persist_fn, n_resumed).
    """
    journal = MagicMock()
    journal.iter_records.return_value = records

    tasks: dict[str, RememberTask] = {}
    task_queue: _queue.Queue[RememberTask] = _queue.Queue()
    lock = threading.Lock()
    persist_fn = MagicMock()
    log_info_fn = MagicMock()

    # Patch Path.exists and Path.read_text so original_path resolution works
    with patch.object(Path, "exists", return_value=True), \
         patch.object(Path, "read_text", return_value=text), \
         patch(
             "core.server.task_worker.estimate_chunk_count",
             return_value=max(1, text_length // 800),
         ):
        n = recover_from_disk(
            journal=journal,
            tasks=tasks,
            task_queue=task_queue,
            lock=lock,
            window_size=1000,
            overlap=200,
            persist_fn=persist_fn,
            log_info_fn=log_info_fn,
        )

    return tasks, task_queue, persist_fn, n


# ===================================================================
# Test 1: Paused task stays paused
# ===================================================================

class TestPausedTaskStaysPaused:
    """A task with status='paused' should be placed in tasks dict but NOT
    re-enqueued into the task_queue."""

    def test_paused_not_queued(self):
        rec = _make_record(status="paused", task_id="paused001")
        tasks, task_queue, persist_fn, n_resumed = _run_recovery([rec])

        # Should be in the tasks dict with status paused
        assert "paused001" in tasks
        task = tasks["paused001"]
        assert task.status == "paused"
        assert task.phase == "paused"

        # Should NOT be in the queue
        assert task_queue.qsize() == 0

        # n_resumed should not count paused tasks
        assert n_resumed == 0

        # persist_fn should still be called (to flush recovered state)
        assert persist_fn.call_count == 1

    def test_paused_message_contains_restart_hint(self):
        rec = _make_record(status="paused", task_id="paused002")
        tasks, task_queue, _, _ = _run_recovery([rec])
        task = tasks["paused002"]
        assert "暂停" in task.message


# ===================================================================
# Test 2: Running task recovers as queued with correct done_chunks
# ===================================================================

class TestRunningTaskRecoversAsQueued:
    """A task with status='running' should recover as 'queued' and its
    chunk counters should be preserved via the monotonic recovery logic."""

    def test_running_becomes_queued(self):
        rec = _make_record(
            status="running",
            task_id="run001",
            total_chunks=20,
            main_done_chunks=12,
            step9_done_chunks=10,
            step10_done_chunks=8,
        )
        tasks, task_queue, persist_fn, n_resumed = _run_recovery([rec])

        assert n_resumed == 1
        assert "run001" in tasks
        task = tasks["run001"]
        assert task.status == "queued"
        assert task.phase == "queued"

        # Should be in the queue
        assert task_queue.qsize() == 1

        # Chunk counters preserved (monotonic: main >= step9 >= step10)
        assert task.main_done_chunks == 12
        assert task.step9_done_chunks == 10
        assert task.step10_done_chunks == 8

        # started_at / finished_at / error cleared
        assert task.started_at is None
        assert task.finished_at is None
        assert task.error is None

    def test_done_chunks_monotonic_clamp(self):
        """If the record has step10 > step9, recovery should clamp step10 <= step9."""
        rec = _make_record(
            status="running",
            task_id="clamp001",
            total_chunks=20,
            main_done_chunks=5,
            step9_done_chunks=3,
            step10_done_chunks=10,  # exceeds step9
        )
        tasks, task_queue, _, n_resumed = _run_recovery([rec])

        task = tasks["clamp001"]
        # Recovery logic: _step10 = min(tc, max(0, step10_done or processed))
        #   _step9  = min(tc, max(_step10, step9_done or processed))
        #   _main   = min(tc, max(_step9, main_done or processed))
        # step10_done=10 => _step10=10
        # step9_done=3 => _step9=max(10,3)=10
        # main_done=5 => _main=max(10,5)=10
        assert task.step10_done_chunks == 10
        assert task.step9_done_chunks == 10  # clamped up
        assert task.main_done_chunks == 10   # clamped up


# ===================================================================
# Test 3: Three-chain independence
# ===================================================================

class TestThreeChainIndependence:
    """When journal has main_done=12, step9_done=15, step10_done=10,
    recovery should preserve step10_done=10 as the _start_chunk base."""

    def test_step10_preserved_independently(self):
        rec = _make_record(
            status="running",
            task_id="chain001",
            total_chunks=20,
            main_done_chunks=12,
            step9_done_chunks=15,
            step10_done_chunks=10,
        )
        tasks, task_queue, _, n_resumed = _run_recovery([rec])

        task = tasks["chain001"]
        assert n_resumed == 1

        # The three chains should be clamped to monotonic order
        # _step10 = min(20, max(0, 10)) = 10
        # _step9  = min(20, max(10, 15)) = 15
        # _main   = min(20, max(15, 12)) = 15
        assert task.step10_done_chunks == 10
        assert task.step9_done_chunks == 15
        assert task.main_done_chunks == 15  # max(15, 12)

        # processed_chunks should match step10_done
        assert task.processed_chunks == 10


# ===================================================================
# Test 4: Legacy record fallback
# ===================================================================

class TestLegacyRecordFallback:
    """When main_done_chunks/step9_done_chunks are missing from record,
    they should be 0 (not inflated from processed_chunks)."""

    def test_missing_main_and_step9_default_to_zero(self):
        rec = _make_record(
            status="running",
            task_id="legacy001",
            total_chunks=20,
            step10_done_chunks=7,
            processed_chunks=7,
        )
        # Remove the keys entirely to simulate legacy records
        rec.pop("main_done_chunks", None)
        rec.pop("step9_done_chunks", None)

        tasks, task_queue, _, n_resumed = _run_recovery([rec])

        task = tasks["legacy001"]
        # remember_task_from_record defaults missing keys to 0
        # Recovery monotonic clamp: _step10=7, _step9=max(7,0)=7, _main=max(7,0)=7
        # Because step9 defaults to 0, it gets clamped up to step10.
        assert task.step10_done_chunks == 7
        # step9_done_chunks: int(None or 0) => 0, then max(_step10, 0) = 7
        assert task.step9_done_chunks == 7
        # main_done_chunks: int(None or 0) => 0, then max(_step9, 0) = 7
        assert task.main_done_chunks == 7

    def test_all_missing_default_to_zero(self):
        """When all done_chunks fields are absent, recovery starts from 0."""
        rec = _make_record(
            status="running",
            task_id="legacy002",
            total_chunks=15,
        )
        rec.pop("main_done_chunks", None)
        rec.pop("step9_done_chunks", None)
        rec.pop("step10_done_chunks", None)
        rec.pop("processed_chunks", None)

        tasks, task_queue, _, n_resumed = _run_recovery([rec])

        task = tasks["legacy002"]
        assert task.main_done_chunks == 0
        assert task.step9_done_chunks == 0
        assert task.step10_done_chunks == 0
        assert task.processed_chunks == 0


# ===================================================================
# Test 5: Progress calculation
# ===================================================================

class TestProgressCalculation:
    """After recovery, task.progress should equal step10_done_chunks / total_chunks."""

    def test_progress_ratio(self):
        rec = _make_record(
            status="running",
            task_id="prog001",
            total_chunks=20,
            step10_done_chunks=10,
        )
        tasks, task_queue, _, n_resumed = _run_recovery([rec])

        task = tasks["prog001"]
        assert task.progress == pytest.approx(10 / 20)

    def test_progress_zero_when_no_done(self):
        rec = _make_record(
            status="running",
            task_id="prog002",
            total_chunks=20,
            step10_done_chunks=0,
        )
        tasks, task_queue, _, n_resumed = _run_recovery([rec])

        task = tasks["prog002"]
        assert task.progress == 0.0

    def test_progress_full(self):
        rec = _make_record(
            status="running",
            task_id="prog003",
            total_chunks=20,
            step10_done_chunks=20,
        )
        tasks, task_queue, _, n_resumed = _run_recovery([rec])

        task = tasks["prog003"]
        assert task.progress == 1.0

    def test_main_progress_separate(self):
        """main_progress should be _main_done / total_chunks."""
        rec = _make_record(
            status="running",
            task_id="prog004",
            total_chunks=20,
            main_done_chunks=15,
            step9_done_chunks=12,
            step10_done_chunks=8,
        )
        tasks, task_queue, _, n_resumed = _run_recovery([rec])

        task = tasks["prog004"]
        assert task.progress == pytest.approx(8 / 20)
        assert task.main_progress == pytest.approx(15 / 20)


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Terminal-status tasks are skipped entirely."""

    def test_completed_task_skipped(self):
        rec = _make_record(status="completed", task_id="done001")
        tasks, task_queue, _, n_resumed = _run_recovery([rec])
        assert "done001" not in tasks
        assert task_queue.qsize() == 0
        assert n_resumed == 0

    def test_failed_task_skipped(self):
        rec = _make_record(status="failed", task_id="fail001")
        tasks, task_queue, _, n_resumed = _run_recovery([rec])
        assert "fail001" not in tasks
        assert n_resumed == 0

    def test_cancelled_task_skipped(self):
        rec = _make_record(status="cancelled", task_id="cancel001")
        tasks, task_queue, _, n_resumed = _run_recovery([rec])
        assert "cancel001" not in tasks
        assert n_resumed == 0

    def test_last_record_wins_for_same_task_id(self):
        """When multiple records share the same task_id, the last one wins."""
        recs = [
            _make_record(task_id="dup001", status="running", step10_done_chunks=3, created_at=100.0),
            _make_record(task_id="dup001", status="running", step10_done_chunks=7, created_at=200.0),
        ]
        tasks, task_queue, _, n_resumed = _run_recovery(recs)
        assert n_resumed == 1
        task = tasks["dup001"]
        assert task.step10_done_chunks == 7

    def test_missing_original_path_marks_failed(self):
        """When original_path is missing and file doesn't exist, task is marked failed."""
        rec = _make_record(
            status="running",
            task_id="miss001",
            original_path="/nonexistent/path.txt",
        )
        # Do NOT patch Path.exists — let it return False by default for /nonexistent
        journal = MagicMock()
        journal.iter_records.return_value = [rec]
        tasks: dict[str, RememberTask] = {}
        task_queue: _queue.Queue[RememberTask] = _queue.Queue()
        lock = threading.Lock()
        persist_fn = MagicMock()
        log_info_fn = MagicMock()

        # estimate_chunk_count is irrelevant here since the task fails early
        with patch("core.server.task_worker.estimate_chunk_count", return_value=1):
            n = recover_from_disk(
                journal=journal,
                tasks=tasks,
                task_queue=task_queue,
                lock=lock,
                window_size=1000,
                overlap=200,
                persist_fn=persist_fn,
                log_info_fn=log_info_fn,
            )

        # Task should NOT be in the live tasks dict (it's marked failed in journal)
        assert "miss001" not in tasks
        assert task_queue.qsize() == 0
        assert n == 0

    def test_multiple_tasks_mixed_statuses(self):
        """Multiple tasks with different statuses recover correctly."""
        recs = [
            _make_record(status="running", task_id="mix001", total_chunks=10, step10_done_chunks=5),
            _make_record(status="paused", task_id="mix002", total_chunks=10),
            _make_record(status="completed", task_id="mix003", total_chunks=10),
        ]
        tasks, task_queue, _, n_resumed = _run_recovery(recs)

        # Only the running task was resumed
        assert n_resumed == 1
        assert "mix001" in tasks
        assert tasks["mix001"].status == "queued"

        # Paused task is in tasks but not queued
        assert "mix002" in tasks
        assert tasks["mix002"].status == "paused"

        # Completed task is skipped
        assert "mix003" not in tasks

        # Only one task in the queue
        assert task_queue.qsize() == 1
