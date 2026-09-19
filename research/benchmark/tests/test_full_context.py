import json
from pathlib import Path
from types import SimpleNamespace

from research.benchmark.datasets import BenchmarkItem, MemorySession
from research.benchmark.full_context import (
    _prompt_chars,
    _visible_contexts,
    fullctx_evaluate_benchmark,
)
from research.benchmark import full_context as fullctx_module
from research.benchmark.runner import AnswerGenerator


def _item(question_id="q1", scope_id="scope-a", sessions=None, visible=None, question="Where did Alice live?"):
    return BenchmarkItem(
        dataset="locomo", scope_id=scope_id, question_id=question_id,
        question=question, answer="Paris", question_type="span",
        question_date="", sessions=sessions or [], evidence_session_ids=[],
        evidence_turn_ids=[], metadata={}, judge_rubric=[],
        visible_session_ids=visible or [],
    )


def test_visible_contexts_respect_per_question_visibility_and_order():
    sessions = [
        MemorySession("s1", "1 June 2024", "Alice moved to Paris."),
        MemorySession("s2", "8 June 2024", "Bob likes tea."),
        MemorySession("s3", "9 June 2024", "Cara visited Rome."),
    ]
    contexts = _visible_contexts(_item(sessions=sessions, visible=["s3", "s1"]))
    assert [row["session_id"] for row in contexts] == ["s1", "s3"]
    assert contexts[0]["text"] == "Alice moved to Paris."
    # No whitelist: every session of the scope is visible.
    assert len(_visible_contexts(_item(sessions=sessions))) == 3


def test_full_context_prompt_includes_everything_and_never_truncates():
    config = {"llm": {"model": "kimi-k3", "context_window_tokens": 262144}}
    sessions = [
        MemorySession("s1", "1 June 2024", "x" * 200000),
        MemorySession("s2", "2 June 2024", "y" * 200000),
    ]
    contexts = _visible_contexts(_item(sessions=sessions))
    answerer = AnswerGenerator(config, full_context=True)
    prompt = answerer.build_prompt(_item(sessions=sessions), contexts)
    assert "x" * 1000 in prompt and "y" * 1000 in prompt

    # Legacy behaviour (non-full-context) still caps and truncates silently.
    capped = AnswerGenerator(config).build_prompt(_item(sessions=sessions), contexts)
    assert "y" * 1000 not in capped


def test_full_context_prompt_over_budget_raises_instead_of_truncating():
    config = {"llm": {"model": "kimi-k3", "context_window_tokens": 2500}}
    sessions = [MemorySession("s1", "1 June 2024", "z" * 9000)]
    answerer = AnswerGenerator(config, full_context=True)
    try:
        answerer.build_prompt(_item(sessions=sessions), _visible_contexts(_item(sessions=sessions)))
    except ValueError as exc:
        assert "exceeds the prompt budget" in str(exc)
    else:
        raise AssertionError("expected a strict budget violation")


def test_neutral_v1_prompt_is_task_neutral_and_answer_passes_through_verbatim():
    config = {"llm": {"model": "kimi-k3", "context_window_tokens": 262144}}
    sessions = [MemorySession("s1", "1 June 2024", "clinic: 0->a, 1->b, 2->c")]
    item = _item(
        question="Continue the pattern: 3->", sessions=sessions,
        visible=["s1"],
    )
    answerer = AnswerGenerator(config, profile="neutral-v1", full_context=True)
    prompt = answerer.build_prompt(item, _visible_contexts(item))
    # The contract defers to the question's own instructions and keeps the JSON
    # payload to a bare answer — no support/answer_type machinery, no QA heuristics.
    assert "question's own instructions" in prompt
    assert '"answer"' in prompt and "answer_type" not in prompt
    assert "false_premise" not in prompt
    assert "Continue the pattern: 3->" in prompt
    assert "clinic: 0->a, 1->b, 2->c" in prompt

    # Normalization is a non-empty check only: boolean/date/comma heuristics and
    # support downgrades must not rewrite the model's answer.
    answer, payload = answerer._normalize_payload(
        item, _visible_contexts(item), {"answer": "3->d"}
    )
    assert answer == "3->d" and payload == {"answer": "3->d"}
    verbatim, _ = answerer._normalize_payload(
        item, _visible_contexts(item), {"answer": "Yes, since the log says so"}
    )
    assert verbatim == "Yes, since the log says so"
    try:
        answerer._normalize_payload(item, _visible_contexts(item), {"answer": "  "})
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("empty neutral answers must be rejected")


def _write_manifest(run_dir: Path, dataset_path: Path):
    manifest = {
        "dataset": "locomo", "dataset_path": str(dataset_path),
        "dataset_sha256": "fixed-hash", "tracks": [], "scopes": {},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _patch_pipeline(monkeypatch, items, calls=None):
    def _fake_selected(*args, **kwargs):
        if calls is not None:
            calls.append((args, kwargs))
        return items, Path("/tmp/dataset.json")

    monkeypatch.setattr(fullctx_module, "_selected_items", _fake_selected)
    monkeypatch.setattr(fullctx_module, "sha256_file", lambda path: "fixed-hash")
    monkeypatch.setattr(
        fullctx_module, "_load_config",
        lambda path: {"llm": {
            "model": "kimi-k3", "context_window_tokens": 2500,
            "timeout_seconds": 5, "max_tokens": 100,
        }},
    )

    def fake_chat(self, messages):
        return SimpleNamespace(
            content='{"support":"supported","answer_type":"span","answer":"Paris"}',
            model="kimi-k3-test", prompt_eval_count=100, eval_count=10,
        )

    monkeypatch.setattr(AnswerGenerator, "_chat", fake_chat)


def test_fullctx_run_excludes_oversize_scopes_and_writes_records(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("{}", encoding="utf-8")
    _write_manifest(run_dir, dataset_path)

    small = _item(
        question_id="q1", scope_id="scope-a",
        sessions=[MemorySession("s1", "1 June 2024", "Alice moved to Paris.")],
    )
    oversize = _item(
        question_id="q2", scope_id="scope-b",
        sessions=[MemorySession("s2", "2 June 2024", "w" * 20000)],
    )
    _patch_pipeline(monkeypatch, [small, oversize])

    result = fullctx_evaluate_benchmark(
        run_dir, tmp_path / "config.json", result_tag="test",
        answer_profile="normalized-v1",
    )
    assert result["track"] == "full-context-test"
    assert set(result["excluded_scopes"]) == {"scope-b"}
    assert result["processed"] == 1 and result["errors"] == 0

    rows = [
        json.loads(line)
        for line in (run_dir / "results.full-context-test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1 and rows[0]["question_id"] == "q1"
    row = rows[0]
    assert row["status"] == "completed" and row["hypothesis"] == "Paris"
    assert row["full_context"] is True
    assert row["context_session_ids"] == ["s1"]
    assert len(row["prompt_sha256"]) == 64
    assert "prompt" not in row  # the whole-corpus prompt is identified, not embedded

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert "full-context-test" in manifest["tracks"]
    variant = manifest["track_variants"]["full-context-test"]
    assert variant["full_context"]["memory_system_used"] is False
    assert set(variant["full_context"]["excluded_scopes"]) == {"scope-b"}

    # Resume: a second invocation processes nothing new.
    again = fullctx_evaluate_benchmark(
        run_dir, tmp_path / "config.json", result_tag="test", resume=True,
    )
    assert again["processed"] == 0


def test_fullctx_strict_fit_refuses_oversize_scopes(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("{}", encoding="utf-8")
    _write_manifest(run_dir, dataset_path)

    oversize = _item(
        question_id="q2", scope_id="scope-b",
        sessions=[MemorySession("s2", "2 June 2024", "w" * 20000)],
    )
    _patch_pipeline(monkeypatch, [oversize])

    try:
        fullctx_evaluate_benchmark(
            run_dir, tmp_path / "config.json", result_tag="test", strict_fit=True,
        )
    except ValueError as exc:
        assert "scope-b" in str(exc)
    else:
        raise AssertionError("strict_fit should refuse to run with oversize scopes")
    assert not (run_dir / "results.full-context-test.jsonl").exists()


def test_fullctx_walks_up_to_data_root_and_passes_scope_ids(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Registry filename sits at the data root; the manifest points one level below
    # (memoryagentbench-style nested layout) — the data dir must be derived by
    # walking up, not by blindly taking the manifest path's parent.
    (tmp_path / "locomo10.json").write_text("{}", encoding="utf-8")
    nested = tmp_path / "memoryagentbench"
    nested.mkdir()
    _write_manifest(run_dir, nested / "locomo10.json")

    calls = []
    small = _item(
        question_id="q1", scope_id="scope-a",
        sessions=[MemorySession("s1", "1 June 2024", "Alice moved to Paris.")],
    )
    _patch_pipeline(monkeypatch, [small], calls=calls)

    result = fullctx_evaluate_benchmark(
        run_dir, tmp_path / "config.json", result_tag="scopes",
        scope_ids=("scope-a",),
    )
    assert result["processed"] == 1 and result["errors"] == 0

    args, kwargs = calls[0]
    assert args[0] == "locomo"
    assert Path(args[1]) == tmp_path.resolve()
    assert kwargs["scope_ids"] == ("scope-a",)
    assert kwargs["question_ids"] == ()
