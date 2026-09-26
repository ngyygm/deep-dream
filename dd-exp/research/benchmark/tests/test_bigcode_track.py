"""bigcode_track 单测：id 准备的 stdlib/BLOCKED 判定 + 生成契约（不触网）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from research.benchmark.bigcode_track import (
    BLOCKED_LIBS,
    _build_user_prompt,
    _load_endpoint,
    _select_ids,
    imported_modules,
    is_blocked,
    is_stdlib_only,
    main,
    prepare_ids,
)
from research.benchmark import bigcode_eval_local


def _task(task_id: str, *, imports: str = "", libs: str | list | None = None, test: str = "import unittest\n") -> dict:
    return {
        "task_id": task_id,
        "instruct_prompt": f"Write a function for {task_id}.",
        "complete_prompt": f"{imports}\ndef task_func():\n",
        "canonical_solution": "    return 1\n",
        "test": test,
        "libs": libs if libs is not None else "[]",
    }


def _write_tasks(path: Path, tasks: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(t) for t in tasks) + "\n", encoding="utf-8")
    return path


def test_imported_modules_handles_plain_and_relative_imports():
    code = "import os.path\nfrom json import dumps\nfrom . import sibling\nimport numpy as np\n"
    modules = imported_modules(code)
    assert {"os", "json", "numpy"} <= modules
    assert "sibling" in modules  # 相对导入 level>0 也计入模块面
    assert imported_modules("def broken(:\n") == set()  # 语法错误 -> 空集兜底


def test_stdlib_and_blocked_classification():
    stdlib = set(sys.stdlib_module_names)
    pure = _task("BigCodeBench/1", imports="import re\nfrom collections import defaultdict\n")
    heavy = _task("BigCodeBench/2", imports="import pandas as pd\n")
    blocked = _task("BigCodeBench/3", imports="import tensorflow\n")
    assert is_stdlib_only(pure, stdlib)
    assert not is_stdlib_only(heavy, stdlib)
    assert not is_stdlib_only(blocked, stdlib)
    assert not is_blocked(pure) and not is_blocked(heavy)
    assert is_blocked(blocked)
    # 完全没有 import 面的任务不能算 stdlib-only（空集判 False，防误选）
    assert not is_stdlib_only(_task("BigCodeBench/4", test=""), stdlib)
    # libs 元数据（字符串或列表）也并入判定
    assert is_blocked(_task("BigCodeBench/5", libs="['shapely']"))
    assert is_blocked(_task("BigCodeBench/6", libs=["pytesseract"]))


def test_prepare_ids_writes_smoke_and_full(tmp_path: Path):
    tasks = [
        _task("BigCodeBench/0", imports="import json\n"),        # stdlib -> smoke 首位
        _task("BigCodeBench/1", imports="import pandas\n"),      # 非 stdlib -> 只进 full
        _task("BigCodeBench/2", imports="import keras\n"),       # blocked -> 两边都不进
        _task("BigCodeBench/3", imports="import math\n"),        # stdlib
    ]
    tasks_file = _write_tasks(tmp_path / "tasks.jsonl", tasks)
    stats = prepare_ids(tasks_file, tmp_path, smoke_n=2)
    smoke = (tmp_path / "smoke50.ids").read_text(encoding="utf-8").split()
    full = (tmp_path / "full.ids").read_text(encoding="utf-8").split()
    assert smoke == ["BigCodeBench/0", "BigCodeBench/3"]  # 数据集顺序 + 截断到 smoke_n
    assert full == ["BigCodeBench/0", "BigCodeBench/1", "BigCodeBench/3"]  # 只剔 blocked
    assert stats == {"total": 4, "stdlib_smoke": 2, "full": 3, "blocked": 1}


def test_select_ids_file_overrides_n_and_validates(tmp_path: Path):
    tasks = [_task(f"BigCodeBench/{i}", imports="import os\n") for i in range(5)]
    ids_file = tmp_path / "sel.ids"
    ids_file.write_text("BigCodeBench/3\nBigCodeBench/1\n", encoding="utf-8")
    assert _select_ids(tasks, ids_file, None) == ["BigCodeBench/3", "BigCodeBench/1"]
    assert _select_ids(tasks, None, 2) == ["BigCodeBench/0", "BigCodeBench/1"]
    bad = tmp_path / "bad.ids"
    bad.write_text("BigCodeBench/99\n", encoding="utf-8")
    try:
        _select_ids(tasks, bad, None)
    except ValueError as exc:
        assert "not in tasks file" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown task id")


def test_build_user_prompt_matches_official_chat_protocol():
    prompt = _build_user_prompt(_task("BigCodeBench/7"))
    assert prompt.startswith("Please provide a self-contained Python script")
    assert "Write a function for BigCodeBench/7." in prompt
    assert prompt.rstrip().endswith("passes corresponding tests:")


def test_load_endpoint_defaults_enable_thinking_false(tmp_path: Path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {"llm": {"base_url": "http://x/v1", "api_key": "EMPTY", "model": "kimi-k3", "max_tokens": 512}}
        ),
        encoding="utf-8",
    )
    endpoint = _load_endpoint(cfg)
    assert endpoint["base_url"] == "http://x/v1"
    assert endpoint["max_tokens"] == 512
    chat_kwargs = endpoint["extra_body"]["chat_template_kwargs"]
    # 该端点模板只认 thinking=False（enable_thinking 无效），两个旗标都带上
    assert chat_kwargs["enable_thinking"] is False
    assert chat_kwargs["thinking"] is False


def test_generation_resume_and_record_contract(tmp_path, monkeypatch):
    """端到端（mock 掉 openai）：resume 跳过已完成 + raw/ids 产物契约。"""
    tasks = [_task(f"BigCodeBench/{i}", imports="import os\n") for i in range(3)]
    tasks_file = _write_tasks(tmp_path / "tasks.jsonl", tasks)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps({"llm": {"base_url": "http://x/v1", "api_key": "EMPTY", "model": "kimi-k3", "max_tokens": 64}}),
        encoding="utf-8",
    )

    calls: list[str] = []

    class _FakeCompletions:
        def create(self, **kwargs):  # noqa: ANN001, ANN003
            calls.append(kwargs["messages"][0]["content"][:40])
            assert kwargs["temperature"] == 0
            assert kwargs["extra_body"]["chat_template_kwargs"]["thinking"] is False

            class _Msg:
                content = "```python\ndef task_func():\n    return 1\n```"

            class _Usage:
                completion_tokens = 7

            class _Choice:
                message = _Msg()

            class _Resp:
                choices = [_Choice()]
                usage = _Usage()

            return _Resp()

    class _FakeClient:
        chat = type("Chat", (), {"completions": _FakeCompletions()})()

    fake_openai = type("M", (), {"OpenAI": staticmethod(lambda **kwargs: _FakeClient())})
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    run_dir = tmp_path / "run"
    rc = main(
        [
            str(run_dir), "--config", str(cfg), "--tasks-file", str(tasks_file),
            "--n", "3", "--workers", "2",
        ]
    )
    assert rc == 0 and len(calls) == 3
    raw_path = run_dir / "raw.completion.jsonl"
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    # raw 是完成序（线程池 as_completed），ids 文件才是数据集序的事实来源
    assert {r["task_id"] for r in records} == {f"BigCodeBench/{i}" for i in range(3)}
    assert all(r["solution"].startswith("```python") for r in records)
    assert {r["completion_tokens"] for r in records} == {7}
    assert (run_dir / "ids.completion.txt").read_text(encoding="utf-8").split() == [
        f"BigCodeBench/{i}" for i in range(3)
    ]

    # resume：raw 里已有 3 条，重跑不再调用端点
    rc2 = main(
        [
            str(run_dir), "--config", str(cfg), "--tasks-file", str(tasks_file),
            "--n", "3", "--workers", "2",
        ]
    )
    assert rc2 == 0 and len(calls) == 3


def test_eval_local_helper_rejects_bad_argc_without_importing_bigcodebench(capsys):
    """argc 错误时直接退出并打印用法，不触发 bigcodebench 导入（repo .venv 没装）。"""
    rc = bigcode_eval_local.main(["prog", "only-one-arg"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "macOS" in err  # docstring 用法被打印
    assert "bigcodebench" not in sys.modules
