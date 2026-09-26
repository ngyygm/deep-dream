from __future__ import annotations

import json
import io
from pathlib import Path

import pytest


def _fake_kimi(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env python3
import json, sys
if '--version' in sys.argv:
    print('kimi-cli 1.49.0')
    raise SystemExit(0)
submit = {'session_ids':['session_1'],'episode_ids':[],'turn_ids':['D1:1']}
print(json.dumps({'role':'assistant','content':'', 'reasoning_content':'secret', 'tool_calls':[{
  'type':'function','id':'tc1','function':{'name':'mcp__deep-dream__search_documents','arguments':json.dumps({'query':'hiking'})}}]}))
print(json.dumps({'role':'tool','tool_call_id':'tc1','content':'source observation'}))
print(json.dumps({'role':'assistant','content':'', 'thinking':'hidden', 'tool_calls':[{
  'type':'function','id':'tc2','function':{'name':'mcp__deep-dream__submit_evidence','arguments':json.dumps({k:json.dumps(v) for k,v in submit.items()})}}]}))
print(json.dumps({'role':'tool','tool_call_id':'tc2','content':json.dumps({'accepted':True, **submit})}))
print(json.dumps({'role':'assistant','content':[{'type':'text','text':json.dumps({'answer':'Hiking','confidence':0.9,'stop_reason':'submit_evidence',**submit})}], 'usage':{'prompt_tokens':10,'completion_tokens':2}}))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _run_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dataset = data_dir / "locomo10.json"
    dataset.write_text(json.dumps([{
        "sample_id": "conv-1",
        "conversation": {
            "session_1": [{"speaker": "A", "text": "I went hiking.", "dia_id": "D1:1"}],
            "session_1_date_time": "1:00 PM on 1 January, 2024",
        },
        "qa": [{
            "question_id": "q/1", "question": "What did A do?", "answer": "Hiking",
            "category": 1, "evidence": ["D1:1"],
        }],
    }]), encoding="utf-8")
    from research.benchmark.datasets import sha256_file
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "schema_version": 4, "dataset": "locomo", "dataset_path": str(dataset),
        "dataset_sha256": sha256_file(dataset), "tracks": [],
        "scopes": {"conv-1": {
            "visible_sessions": ["session_1"], "library_dir": "library",
            "documents": {"session_1": {"document_id": "doc-1", "status": "active"}},
        }},
    }), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "storage_path": str(run_dir / "library"),
        "llm": {"base_url": "https://example.invalid/v1", "model": "qwen3.7-plus",
                "api_key_env": "TEST_QWEN_KEY", "context_window_tokens": 32000},
        "embedding": {"model": "all-MiniLM-L6-v2", "device": "cpu"},
    }), encoding="utf-8")
    return run_dir, config


def test_kimi_runtime_strips_reasoning_and_requires_submitted_ids(tmp_path, monkeypatch):
    from research.benchmark.kimi_runtime import KimiAgentRuntime
    executable = _fake_kimi(tmp_path / "kimi")
    run_dir, config = _run_fixture(tmp_path)
    monkeypatch.setenv("TEST_QWEN_KEY", "not-a-real-key")
    result = KimiAgentRuntime(
        executable=executable, run_dir=run_dir, config_path=config,
        model="qwen3.7-plus", thinking=True,
    ).run(scope_id="conv-1", question_id="q/1", question="What did A do?")
    assert result.final["answer"] == "Hiking"
    assert result.final["turn_ids"] == ["D1:1"]
    assert result.tool_counts["mcp__deep-dream__submit_evidence"] == 1
    serialized = json.dumps(result.events)
    assert "secret" not in serialized
    assert "hidden" not in serialized
    generated = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (run_dir / "kimi_runtime").rglob("*") if path.is_file()
    )
    assert "not-a-real-key" not in generated
    assert "tools: []" in generated


def test_agent_query_writes_direct_evidence_and_v5_manifest(tmp_path, monkeypatch):
    from research.benchmark.kimi_benchmark import agent_query_benchmark
    from research.benchmark.reporting import read_jsonl
    executable = _fake_kimi(tmp_path / "kimi")
    run_dir, config = _run_fixture(tmp_path)
    monkeypatch.setenv("TEST_QWEN_KEY", "not-a-real-key")
    result = agent_query_benchmark(
        run_dir, config, executable=executable, result_tag="kimi-qwen37-thinking-on",
        qa_workers=2,
    )
    assert result["processed"] == 1
    assert result["errors"] == 0
    direct = read_jsonl(run_dir / "results.kimi-agent-direct-qwen37-thinking-on.jsonl")
    evidence = read_jsonl(run_dir / "retrieval.kimi-agent-evidence-qwen37-thinking-on.jsonl")
    assert direct[0]["hypothesis"] == "Hiking"
    assert evidence[0]["ranked_turn_ids"] == ["D1:1"]
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["schema_version"] == 5
    assert manifest["agent_runtimes"]["kimi-qwen37-thinking-on"]["model"] == "qwen3.7-plus"
    assert manifest["agent_runtimes"]["kimi-qwen37-thinking-on"]["temperature"] == "provider-default"
    assert "not-a-real-key" not in json.dumps(manifest)
    assert not any((run_dir / "trajectories").rglob("q/1.json"))


def test_kimi_final_evidence_must_match_submit(tmp_path, monkeypatch):
    from research.benchmark.kimi_runtime import KimiAgentRuntime
    executable = _fake_kimi(tmp_path / "kimi")
    text = executable.read_text().replace(
        "'stop_reason':'submit_evidence',**submit",
        "'stop_reason':'submit_evidence',**submit,'turn_ids':['D1:2']",
    )
    executable.write_text(text)
    executable.chmod(0o755)
    run_dir, config = _run_fixture(tmp_path)
    monkeypatch.setenv("TEST_QWEN_KEY", "not-a-real-key")
    with pytest.raises(ValueError, match="does not match submitted evidence"):
        KimiAgentRuntime(
            executable=executable, run_dir=run_dir, config_path=config,
            model="qwen3.7-plus", thinking=False,
        ).run(scope_id="conv-1", question_id="q/1", question="What did A do?")


def test_mcp_normalizes_json_array_strings():
    from research.benchmark.mcp_server import _string_list
    assert _string_list('["session_1", "session_1"]') == ["session_1"]
    assert _string_list(["D1:1"]) == ["D1:1"]
    assert _string_list(None) == []


def test_persistent_runtime_reuses_process_but_requires_fresh_sessions(tmp_path, monkeypatch):
    from research.benchmark.kimi_runtime import KimiAgentRuntime

    run_dir, config = _run_fixture(tmp_path)
    executable = tmp_path / "runtime" / "bin" / "kimi"
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable = _fake_kimi(executable)
    monkeypatch.setenv("TEST_QWEN_KEY", "not-a-real-key")

    submit = {"session_ids": ["session_1"], "episode_ids": [], "turn_ids": ["D1:1"]}

    def response(session_id, *, plain=False, evidence=True):
        updates = ([
            {
                "sessionUpdate": "tool_call", "toolCallId": "tc",
                "title": "mcp__deep-dream__submit_evidence", "status": "in_progress",
                "content": [{"type": "content", "content": {
                    "type": "text", "text": json.dumps(submit),
                }}],
            },
            {
                "sessionUpdate": "tool_call_update", "toolCallId": "tc",
                "status": "completed", "content": [{"type": "content", "content": {
                    "type": "text", "text": json.dumps({"accepted": True, **submit}),
                }}],
            },
        ] if evidence else []) + [
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Hiking" if plain else json.dumps({
                    "answer": "Hiking", "confidence": 1, "stop_reason": "submit_evidence",
                    **submit,
                })},
            },
        ]
        return json.dumps({
            "ok": True, "request_id": "REPLACE", "session_id": session_id,
            "fresh_context": True, "bridge_pid": 77, "latency_seconds": 1,
            "updates": updates,
        })

    class FakeProcess:
        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(
                response("fresh-a").replace('"REPLACE"', '"q1"') + "\n"
                + response("fresh-b", plain=True).replace('"REPLACE"', '"q2"') + "\n"
                + response("fresh-c", plain=True, evidence=False).replace('"REPLACE"', '"q3"') + "\n"
            )

    runtime = KimiAgentRuntime(
        executable=executable, run_dir=run_dir, config_path=config,
        model="qwen3.7-plus", thinking=False, lifecycle="persistent",
    )
    process = FakeProcess()
    monkeypatch.setattr(runtime, "_bridge", lambda: process)
    first = runtime.run(scope_id="conv-1", question_id="q1", question="First?")
    second = runtime.run(scope_id="conv-1", question_id="q2", question="Second?")
    third = runtime.run(scope_id="conv-1", question_id="q3", question="Third?")
    first_meta = [row for row in first.events if row.get("type") == "runtime_session"][0]
    second_meta = [row for row in second.events if row.get("type") == "runtime_session"][0]
    assert first_meta["bridge_pid"] == second_meta["bridge_pid"] == 77
    assert first_meta["session_id"] != second_meta["session_id"]
    assert first_meta["fresh_context"] and second_meta["fresh_context"]
    assert second.final["answer"] == "Hiking"
    assert second.final["format_fallback"] == "plain-text-after-accepted-evidence"
    assert third.final["answer"] == "Hiking"
    assert third.final["turn_ids"] == []
    assert third.final["format_fallback"] == "plain-text-without-evidence"
