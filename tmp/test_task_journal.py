"""
Tests for task_journal serialization: task_to_dict / remember_task_from_record
and RememberJournal.write atomic-write semantics.
"""
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

# Ensure project root is on sys.path so `core` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.server.task_journal import (
    RememberTask,
    RememberJournal,
    remember_task_from_record,
    task_to_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(**overrides) -> RememberTask:
    """Build a RememberTask with sensible defaults, applying *overrides* last."""
    defaults = dict(
        task_id="abc12345-6789-def0-1234-567890abcdef",
        text="sample document text",
        source_name="test_source",
        load_cache=None,
        control_action=None,
        event_time=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        original_path="/tmp/test.md",
        cache_document_path=None,
        override_doc_id="",
        status="running",
        result=None,
        error=None,
        created_at=1750000000.0,
        started_at=1750000001.0,
        finished_at=None,
        phase="main",
        phase_label="处理中",
        phase_current=2,
        phase_total=5,
        main_done_chunks=7,
        step9_done_chunks=3,
        step10_done_chunks=4,
        processed_chunks=14,
        total_chunks=20,
        run_start_chunks=0,
        task_seq=42,
        progress=0.55,
        message="正在处理",
        step9_progress=0.3,
        step9_label="step9 running",
        step10_progress=0.4,
        step10_label="step10 running",
        main_progress=0.5,
        main_label="main running",
        chain_started_at={"chain_a": 1750000010.0},
        chain_run_start_chunks={"chain_a": 5},
        failed_window_indices=[1, 3],
        failed_window_errors=[{"idx": 1, "msg": "err"}],
        repair_window_indices=[2],
        repair_window_statuses=[{"idx": 2, "ok": True}],
        retry_attempt=1,
        max_retries=3,
        last_update=1750000020.0,
    )
    defaults.update(overrides)
    # done_event is not serialized; provide default
    defaults.setdefault("done_event", threading.Event())
    return RememberTask(**defaults)


# ---------------------------------------------------------------------------
# 1. Round-trip: task_to_dict -> remember_task_from_record preserves fields
# ---------------------------------------------------------------------------

def test_roundtrip_preserves_all_fields():
    original = _make_task()
    d = task_to_dict(original)
    restored = remember_task_from_record(d, text=original.text)

    # Scalar fields that survive serialization
    assert restored.task_id == original.task_id
    assert restored.text == original.text
    assert restored.source_name == original.source_name
    assert restored.load_cache == original.load_cache
    assert restored.control_action == original.control_action
    assert restored.event_time == original.event_time
    assert restored.original_path == original.original_path
    assert restored.cache_document_path == original.cache_document_path
    assert restored.override_doc_id == original.override_doc_id
    assert restored.status == original.status
    assert restored.result == original.result
    assert restored.error == original.error
    assert restored.created_at == original.created_at
    assert restored.started_at == original.started_at
    assert restored.finished_at == original.finished_at
    assert restored.phase == original.phase
    assert restored.phase_label == original.phase_label
    assert restored.phase_current == original.phase_current
    assert restored.phase_total == original.phase_total
    assert restored.main_done_chunks == original.main_done_chunks
    assert restored.step9_done_chunks == original.step9_done_chunks
    assert restored.step10_done_chunks == original.step10_done_chunks
    assert restored.processed_chunks == original.processed_chunks
    assert restored.total_chunks == original.total_chunks
    assert restored.run_start_chunks == original.run_start_chunks
    assert restored.task_seq == original.task_seq
    assert restored.progress == original.progress
    assert restored.message == original.message
    assert restored.step9_progress == original.step9_progress
    assert restored.step9_label == original.step9_label
    assert restored.step10_progress == original.step10_progress
    assert restored.step10_label == original.step10_label
    assert restored.main_progress == original.main_progress
    assert restored.main_label == original.main_label
    assert restored.chain_started_at == original.chain_started_at
    assert restored.chain_run_start_chunks == original.chain_run_start_chunks
    assert restored.failed_window_indices == original.failed_window_indices
    assert restored.failed_window_errors == original.failed_window_errors
    assert restored.repair_window_indices == original.repair_window_indices
    assert restored.repair_window_statuses == original.repair_window_statuses
    assert restored.retry_attempt == original.retry_attempt
    assert restored.max_retries == original.max_retries
    assert restored.last_update == original.last_update


# ---------------------------------------------------------------------------
# 2. Legacy record: missing main_done_chunks / step9_done_chunks => 0
#    NOT processed_chunks
# ---------------------------------------------------------------------------

def test_legacy_record_missing_chunk_counters():
    """Old records that lack main_done_chunks / step9_done_chunks should
    default them to 0, NOT to processed_chunks."""
    rec = {
        "task_id": "legacy001",
        "source_name": "old_source",
        "original_path": "/old/doc.md",
        "status": "running",
        "processed_chunks": 15,
        "total_chunks": 20,
        # main_done_chunks and step9_done_chunks intentionally absent
        # step10_done_chunks also absent
    }
    task = remember_task_from_record(rec, text="legacy doc")

    # main_done_chunks should be 0, NOT 15 (processed_chunks)
    assert task.main_done_chunks == 0, (
        f"main_done_chunks should be 0 for legacy records, got {task.main_done_chunks}"
    )
    # step9_done_chunks should be 0, NOT 15
    assert task.step9_done_chunks == 0, (
        f"step9_done_chunks should be 0 for legacy records, got {task.step9_done_chunks}"
    )
    # processed_chunks itself is preserved
    assert task.processed_chunks == 15


# ---------------------------------------------------------------------------
# 3. step10_done_chunks fallback: when missing but processed_chunks is set,
#    step10_done_chunks should fall back to processed_chunks
# ---------------------------------------------------------------------------

def test_step10_fallback_to_processed_chunks():
    rec = {
        "task_id": "step10fb",
        "source_name": "test",
        "original_path": "/doc.md",
        "status": "running",
        "processed_chunks": 10,
        # step10_done_chunks is absent
    }
    task = remember_task_from_record(rec, text="fallback doc")

    assert task.step10_done_chunks == 10, (
        f"step10_done_chunks should fall back to processed_chunks (10), "
        f"got {task.step10_done_chunks}"
    )
    # Sanity: processed_chunks itself is still correct
    assert task.processed_chunks == 10


# ---------------------------------------------------------------------------
# 4. Atomic write: Journal.write uses temp file + rename pattern
# ---------------------------------------------------------------------------

def test_journal_write_uses_atomic_rename(tmp_path):
    """Verify that _write_unlocked writes to a .tmp file then renames."""
    journal = RememberJournal(tmp_path)
    task = _make_task(status="running")

    # We patch Path.replace to track that it is called with the right target
    original_replace = Path.replace
    replace_calls: list = []

    def tracking_replace(self_path: Path, target: Path) -> Path:
        replace_calls.append((str(self_path), str(target)))
        return original_replace(self_path, target)

    with patch.object(Path, "replace", tracking_replace):
        journal.write(task)

    # The tmp file should have been renamed to the final journal file
    expected_tmp = str(journal._file.with_suffix(".jsonl.tmp"))
    expected_final = str(journal._file)

    assert len(replace_calls) == 1, f"Expected exactly 1 rename, got {len(replace_calls)}"
    assert replace_calls[0] == (expected_tmp, expected_final), (
        f"Rename should be from {expected_tmp} to {expected_final}, "
        f"got {replace_calls[0]}"
    )

    # Verify the final file exists and contains valid JSON with our task
    assert journal._file.exists(), "Journal file should exist after write"
    records = journal.iter_records()
    assert len(records) == 1
    assert records[0]["task_id"] == task.task_id


def test_journal_write_tmp_file_created(tmp_path):
    """Verify the .tmp file is created before rename."""
    journal = RememberJournal(tmp_path)
    task = _make_task(status="running", task_id="tmp-check-001")

    write_targets: list = []

    real_open = builtins_open = open  # capture before patching

    def tracking_open(file, mode="r", *args, **kwargs):
        # Only track writes to .tmp files
        file_str = str(file)
        if "w" in mode and ".tmp" in file_str:
            write_targets.append(file_str)
        return real_open(file, mode, *args, **kwargs)

    with patch("builtins.open", tracking_open):
        journal._write_unlocked(task)

    assert any(journal._file.with_suffix(".jsonl.tmp").__str__() in t for t in write_targets), (
        f"Expected a write to .jsonl.tmp, got writes to: {write_targets}"
    )


def test_journal_terminal_status_removes_task(tmp_path):
    """Terminal-status tasks should be removed (not written) from the journal."""
    journal = RememberJournal(tmp_path)

    # First write a running task
    running_task = _make_task(status="running", task_id="term-001")
    journal.write(running_task)
    assert journal.read_record("term-001") is not None, "Task should exist after initial write"

    # Now mark it completed and write again
    completed_task = _make_task(status="completed", task_id="term-001")
    journal.write(completed_task)

    record = journal.read_record("term-001")
    assert record is None, (
        f"Completed task should be removed from journal, but found: {record}"
    )


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("Running test_roundtrip_preserves_all_fields...", end=" ")
    test_roundtrip_preserves_all_fields()
    print("PASSED")

    print("Running test_legacy_record_missing_chunk_counters...", end=" ")
    test_legacy_record_missing_chunk_counters()
    print("PASSED")

    print("Running test_step10_fallback_to_processed_chunks...", end=" ")
    test_step10_fallback_to_processed_chunks()
    print("PASSED")

    # The atomic-write tests need a tmp_path fixture equivalent
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)

        print("Running test_journal_write_uses_atomic_rename...", end=" ")
        test_journal_write_uses_atomic_rename(tp)
        print("PASSED")

    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)

        print("Running test_journal_write_tmp_file_created...", end=" ")
        test_journal_write_tmp_file_created(tp)
        print("PASSED")

    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)

        print("Running test_journal_terminal_status_removes_task...", end=" ")
        test_journal_terminal_status_removes_task(tp)
        print("PASSED")

    print("\nAll tests passed.")
