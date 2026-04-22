from __future__ import annotations

import pytest
from pathlib import Path

from server.task_queue import RememberTask, RememberTaskQueue, _remember_callback_ui_fields


def _make_task(**overrides) -> RememberTask:
    base = dict(
        task_id="task-progress",
        text="demo",
        source_name="demo.txt",
        load_cache=False,
        control_action=None,
        event_time=None,
        original_path="",
        total_chunks=10,
        processed_chunks=7,
        step6_done_chunks=7,
        step7_done_chunks=7,
        step6_progress=0.7,
        step7_progress=0.7,
    )
    base.update(overrides)
    return RememberTask(**base)


class TestRememberCallbackUiFields:
    def test_step6_progress_uses_global_window_position(self):
        task = _make_task()

        fields = _remember_callback_ui_fields(
            task,
            progress=0.7785714285714286,  # 窗口 8/10 的 step6 中点
            phase_label="窗口 8/10 · 步骤6/7: 实体对齐 (3/6)",
            message="实体对齐 3/6",
            chain_id="step6",
        )

        assert fields["step6_progress"] == pytest.approx(0.75)
        assert "step7_progress" not in fields

    def test_step7_progress_keeps_entity_bar_at_completed_window_fraction(self):
        task = _make_task(step6_done_chunks=8, step6_progress=0.8)

        fields = _remember_callback_ui_fields(
            task,
            progress=0.7928571428571428,  # 窗口 8/10 的 step7 中点
            phase_label="窗口 8/10 · 步骤7/7: 关系对齐 (5/10)",
            message="关系对齐 5/10",
            chain_id="step7",
        )

        assert fields["step6_progress"] == pytest.approx(0.8)
        assert fields["step7_progress"] == pytest.approx(0.75)

    def test_step6_update_does_not_reset_existing_relation_progress(self):
        task = _make_task(step7_progress=0.6)

        fields = _remember_callback_ui_fields(
            task,
            progress=0.775,
            phase_label="窗口 8/10 · 步骤6/7: 实体对齐 (1/4)",
            message="实体对齐 1/4",
            chain_id="step6",
        )

        assert "step7_progress" not in fields
        assert "step7_label" not in fields


class _DummyProcessor:
    def __init__(self, load_cache_memory: bool, runtime_stats: dict | None = None):
        self.load_cache_memory = load_cache_memory
        self._runtime_stats = runtime_stats or {}

        class _Doc:
            window_size = 100
            overlap = 10

        self.document_processor = _Doc()

    def get_runtime_stats(self):
        return dict(self._runtime_stats)


class TestRememberTaskQueueLoadCacheSemantics:
    def test_none_uses_processor_default(self, tmp_path: Path):
        processor = _DummyProcessor(load_cache_memory=True)
        q = RememberTaskQueue(
            processor,
            tmp_path,
            processor_factory=lambda: _DummyProcessor(load_cache_memory=False),
            max_workers=0,
        )
        task = _make_task(load_cache=None)
        assert q._task_uses_external_cache(task) is True

    def test_false_means_independent_task(self, tmp_path: Path):
        processor = _DummyProcessor(load_cache_memory=True)
        q = RememberTaskQueue(
            processor,
            tmp_path,
            processor_factory=lambda: _DummyProcessor(load_cache_memory=False),
            max_workers=0,
        )
        task = _make_task(load_cache=False)
        assert q._task_uses_external_cache(task) is False

    def test_runtime_stats_snapshot_falls_back_to_shared_processor(self, tmp_path: Path):
        processor = _DummyProcessor(
            load_cache_memory=True,
            runtime_stats={
                "configured_window_workers": 64,
                "active_main_pipeline_windows": 0,
                "llm_semaphore_active": 0,
                "llm_semaphore_max": 200,
            },
        )
        q = RememberTaskQueue(
            processor,
            tmp_path,
            processor_factory=lambda: _DummyProcessor(load_cache_memory=False),
            max_workers=0,
        )

        stats = q.get_runtime_stats_snapshot()

        assert stats["configured_window_workers"] == 64
        assert stats["llm_semaphore_max"] == 200
        assert stats["active_main_pipeline_windows"] == 0

    def test_runtime_stats_snapshot_aggregates_active_task_processors(self, tmp_path: Path):
        shared = _DummyProcessor(
            load_cache_memory=True,
            runtime_stats={
                "configured_window_workers": 64,
                "active_main_pipeline_windows": 0,
                "active_step6": 0,
                "active_step7": 0,
                "llm_semaphore_active": 0,
                "llm_semaphore_max": 200,
            },
        )
        p1 = _DummyProcessor(
            load_cache_memory=False,
            runtime_stats={
                "configured_window_workers": 64,
                "active_main_pipeline_windows": 3,
                "active_step6": 1,
                "active_step7": 0,
                "llm_semaphore_active": 5,
                "llm_semaphore_max": 200,
            },
        )
        p2 = _DummyProcessor(
            load_cache_memory=False,
            runtime_stats={
                "configured_window_workers": 64,
                "active_main_pipeline_windows": 2,
                "active_step6": 0,
                "active_step7": 1,
                "llm_semaphore_active": 4,
                "llm_semaphore_max": 200,
            },
        )
        q = RememberTaskQueue(
            shared,
            tmp_path,
            processor_factory=lambda: _DummyProcessor(load_cache_memory=False),
            max_workers=0,
        )
        q._set_active_processor("task-1", p1)
        q._set_active_processor("task-2", p2)

        stats = q.get_runtime_stats_snapshot()

        assert stats["configured_window_workers"] == 128
        assert stats["active_main_pipeline_windows"] == 5
        assert stats["active_step6"] == 1
        assert stats["active_step7"] == 1
        assert stats["llm_semaphore_active"] == 9
        assert stats["llm_semaphore_max"] == 400

    def test_request_pause_and_resume_running_task(self, tmp_path: Path):
        processor = _DummyProcessor(load_cache_memory=False)
        q = RememberTaskQueue(
            processor,
            tmp_path,
            processor_factory=lambda: _DummyProcessor(load_cache_memory=False),
            max_workers=0,
        )
        task = _make_task(task_id="task-run", status="running")
        q._tasks[task.task_id] = task

        ok_pause, _msg, status = q.request_pause_task(task.task_id)
        assert ok_pause is True
        assert status == "pausing"
        assert task.control_action == "pause"

        task.status = "paused"
        task.phase = "paused"
        task.control_action = None
        ok_resume, _msg, status = q.resume_task(task.task_id)
        assert ok_resume is True
        assert status == "queued"
        assert task.status == "queued"

    def test_request_delete_running_task_marks_cancel(self, tmp_path: Path):
        processor = _DummyProcessor(load_cache_memory=False)
        q = RememberTaskQueue(
            processor,
            tmp_path,
            processor_factory=lambda: _DummyProcessor(load_cache_memory=False),
            max_workers=0,
        )
        task = _make_task(task_id="task-cancel", status="running")
        q._tasks[task.task_id] = task

        ok_delete, _msg, status = q.request_delete_task(task.task_id)
        assert ok_delete is True
        assert status == "cancelling"
        assert task.control_action == "cancel"
