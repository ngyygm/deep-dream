import json
from pathlib import Path

import pytest

from core.benchmark.agentic import (
    AgenticMemoryRunner,
    AgenticMemoryTools,
    _agent_extra_body,
    _parse_json_object,
)
from core.benchmark.datasets import BenchmarkItem, MemorySession, load_locomo, load_longmemeval, parse_timestamp
from core.benchmark.metrics import aggregate_records, locomo_f1, retrieval_at_k, token_f1
from core.benchmark.retrieval import (
    HYBRID_V2_WEIGHTS,
    HybridRetrievalConfig,
    UnifiedRetriever,
    extract_query_terms,
)
from core.benchmark.runner import _document_id
from core.benchmark.scoring import JUDGE_MODEL, longmemeval_judge_prompt, report_run, score_run
from core.benchmark import runner


def test_runtime_policy_is_packaged_and_fingerprinted():
    from core.agent import load_runtime_policy, runtime_policy_metadata
    policy = load_runtime_policy()
    metadata = runtime_policy_metadata()
    assert "submit_evidence" in policy
    assert "home country" in policy
    assert metadata["version"] == "1.0.0"
    assert len(metadata["sha256"]) == 64


def test_quality_v1_filters_noise_before_content_generation():
    from core.remember.quality import cap_relation_pairs, filter_entity_names
    source = "[D1:1] Alice: Hello?\n[D1:2] Bob: Alice moved to Sweden and reads Dr. Seuss."
    names = ["D1:1", "Hello", "Hello?", "Alice moved to Sweden and reads Dr. Seuss?", "Alice", "Sweden", "Dr. Seuss"]
    assert filter_entity_names(names, source, limit=16) == ["Alice", "Sweden", "Dr. Seuss"]
    assert cap_relation_pairs([("Alice", "Sweden"), ("Sweden", "Alice"), ("Alice", "Alice")]) == [
        ("Alice", "Sweden")
    ]


def test_manifest_v4_records_policy_skill_and_remember_profile(tmp_path):
    dataset = _write(tmp_path / "locomo10.json", [])
    manifest = runner.create_manifest(
        "locomo", dataset, {"llm": {"model": "qwen3.6-27b-awq"}}, tmp_path / "run",
        remember_profile="quality-v1",
    )
    assert manifest["schema_version"] == 4
    assert manifest["remember_profile"] == "quality-v1"
    assert len(manifest["runtime_policy"]["sha256"]) == 64
    assert len(manifest["skill"]["sha256"]) == 64


def test_hybrid_v2_query_terms_are_deterministic_and_bounded():
    question = "When did Caroline move from her home country, Sweden, last Friday?"
    first = extract_query_terms(question)
    second = extract_query_terms(question)
    assert first == second
    assert 3 <= len(first) <= 8
    assert first[0] == {"term": question, "source": "original"}
    assert any("Caroline" in row["term"] for row in first)
    assert all("category" not in row["term"].lower() for row in first)


def test_hybrid_v2_profile_is_stable_and_declares_weights():
    profile = HybridRetrievalConfig()
    assert profile.payload()["weights"] == HYBRID_V2_WEIGHTS
    assert len(profile.fingerprint()) == 64
    assert profile.fingerprint() == HybridRetrievalConfig().fingerprint()


def test_hybrid_v2_ranks_turns_expands_neighbors_and_preserves_anchor_metrics(monkeypatch):
    sessions = [
        MemorySession(
            "s1", "1 June 2024",
            "[t1] A: We talked about travel.\n"
            "[t2] A: I moved from Sweden to Canada.\n"
            "[t3] B: That was a big change.",
            ["t1", "t2", "t3"],
        ),
        MemorySession("s2", "2 June 2024", "[u1] B: I enjoy painting.", ["u1"]),
    ]
    retriever = UnifiedRetriever.__new__(UnifiedRetriever)
    retriever.storage = object()
    retriever.sessions = {row.session_id: row for row in sessions}
    retriever._session_turns = {
        "s1": [("t1", "A: We talked about travel."),
               ("t2", "A: I moved from Sweden to Canada."),
               ("t3", "B: That was a big change.")],
        "s2": [("u1", "B: I enjoy painting.")],
    }
    retriever._turn_to_session = {"t1": "s1", "t2": "s1", "t3": "s1", "u1": "s2"}
    retriever._candidate_channels = lambda *_args, **_kwargs: ({
        "raw-document": [{
            "evidence_id": "e1", "session_id": "s1", "turn_id": "t2",
            "raw_text": "A: I moved from Sweden to Canada.", "channel_rank": 1,
            "retrieval_channel": "raw-document",
        }],
        "semantic-provenance": [{
            "evidence_id": "e2", "session_id": "s2", "turn_id": "u1",
            "raw_text": "B: I enjoy painting.", "channel_rank": 1,
            "retrieval_channel": "semantic-provenance",
        }],
        "graph-neighbor": [{
            "evidence_id": "e3", "session_id": "s1", "turn_id": "t2",
            "raw_text": "A: I moved from Sweden to Canada.", "channel_rank": 2,
            "retrieval_channel": "graph-neighbor",
        }],
    }, {"coverage": {"neighbor_evidence": 1}})
    # Sorted turn IDs are t2, u1. Make t2 semantically identical to the query.
    retriever._embeddings = lambda _texts: __import__("numpy").array([
        [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]
    ], dtype="float32")

    result = retriever.explore(
        "Where did A move from?", retrieval_profile="hybrid-v2",
        candidate_k=30, context_k=2, evidence_token_budget=200, neighbor_turns=1,
    )
    assert result["ranked_turn_ids"][0] == "t2"
    assert result["ranked_session_ids"][0] == "s1"
    first = next(row for row in result["contexts"] if row["session_id"] == "s1")
    assert first["matched_turn_ids"] == ["t2"]
    assert first["turn_ids"] == ["t1", "t2", "t3"]
    assert result["turn_scores"][0]["channels"] == ["raw-document", "graph-neighbor"]
    assert result["budget"]["used"] <= result["budget"]["limit"]


def test_hybrid_v2_budget_keeps_one_anchor_per_selected_session():
    sessions = [
        MemorySession("s1", "", "[t1] A: " + "x" * 80, ["t1"]),
        MemorySession("s2", "", "[t2] B: " + "y" * 80, ["t2"]),
    ]
    retriever = UnifiedRetriever.__new__(UnifiedRetriever)
    retriever.storage = object()
    retriever.sessions = {row.session_id: row for row in sessions}
    retriever._session_turns = {
        "s1": [("t1", "A: " + "x" * 80)], "s2": [("t2", "B: " + "y" * 80)],
    }
    retriever._turn_to_session = {"t1": "s1", "t2": "s2"}
    retriever._candidate_channels = lambda *_args, **_kwargs: ({
        "episode-bm25": [
            {"evidence_id": "a", "session_id": "s1", "turn_id": "t1",
             "raw_text": "A: " + "x" * 80, "channel_rank": 1},
            {"evidence_id": "b", "session_id": "s2", "turn_id": "t2",
             "raw_text": "B: " + "y" * 80, "channel_rank": 2},
        ],
    }, {"coverage": {}})
    retriever._embeddings = lambda _texts: __import__("numpy").array([
        [1.0, 0.0], [1.0, 0.0], [0.9, 0.1]
    ], dtype="float32")
    result = retriever.explore(
        "memory", retrieval_profile="hybrid-v2", context_k=2,
        evidence_token_budget=1, neighbor_turns=0,
    )
    assert len(result["contexts"]) == 2
    assert all(len(row["matched_turn_ids"]) == 1 for row in result["contexts"])
    assert result["budget"]["anchor_overflow"] is True


def test_hybrid_v2_resolves_home_country_referential_bridge():
    sessions = [
        MemorySession(
            "s1", "", "[t1] A: I moved from my home country four years ago.", ["t1"],
        ),
        MemorySession(
            "s2", "", "[u1] A: My home country is Sweden.\n[u2] A: Four years is a long time.",
            ["u1", "u2"],
        ),
    ]
    retriever = UnifiedRetriever.__new__(UnifiedRetriever)
    retriever.storage = object()
    retriever.sessions = {row.session_id: row for row in sessions}
    retriever._session_turns = {
        "s1": [("t1", "A: I moved from my home country four years ago.")],
        "s2": [("u1", "A: My home country is Sweden."),
               ("u2", "A: Four years is a long time.")],
    }
    retriever._turn_to_session = {"t1": "s1", "u1": "s2", "u2": "s2"}
    retriever._candidate_channels = lambda *_args, **_kwargs: ({
        "episode-bm25": [
            {"evidence_id": "a", "session_id": "s1", "turn_id": "t1",
             "raw_text": "A: I moved from my home country four years ago.", "channel_rank": 1},
            {"evidence_id": "b", "session_id": "s2", "turn_id": "u2",
             "raw_text": "A: Four years is a long time.", "channel_rank": 2},
        ],
    }, {"coverage": {}})
    retriever._embeddings = lambda _texts: __import__("numpy").array([
        [1.0, 0.0], [1.0, 0.0], [0.8, 0.2]
    ], dtype="float32")

    result = retriever.explore(
        "Where did A move from four years ago?", retrieval_profile="hybrid-v2",
        context_k=2, evidence_token_budget=200, neighbor_turns=0,
    )
    assert result["ranked_turn_ids"][:2] == ["t1", "u1"]
    assert result["referential_bridges"]["turn_ids"] == ["t1", "u1"]
    second = next(row for row in result["contexts"] if row["session_id"] == "s2")
    assert "u1" in second["matched_turn_ids"]


def test_benchmark_cli_rejects_top_k_with_context_k_and_keeps_alias(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from core.cli.cmd_benchmark import benchmark

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cli = CliRunner()
    ambiguous = cli.invoke(benchmark, [
        "retrieve", str(run_dir), "--context-k", "5", "--top-k", "5",
    ])
    assert ambiguous.exit_code == 2
    assert "not both" in ambiguous.output

    monkeypatch.setattr(runner, "retrieve_benchmark", lambda *_args, **kwargs: {
        "context_k": kwargs["context_k"], "processed": 0,
    })
    alias = cli.invoke(benchmark, ["retrieve", str(run_dir), "--top-k", "7"])
    assert alias.exit_code == 0
    assert "context_k: 7" in alias.output


def _write(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_longmemeval_adapter_preserves_roles_timestamps_and_evidence(tmp_path):
    path = _write(tmp_path / "lme.json", [{
        "question_id": "q1_abs", "question_type": "multi-session", "question": "Where?",
        "question_date": "2023/05/30 (Tue) 23:40", "answer": "unknown",
        "answer_session_ids": ["s2"], "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2023/05/01 (Mon) 10:00", "2023/05/02 (Tue) 10:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "hello", "has_answer": False}],
            [{"role": "assistant", "content": "Paris", "has_answer": True}],
        ],
    }])
    item = load_longmemeval(path)[0]
    assert item.metadata["abstention"] is True
    assert item.evidence_session_ids == ["s2"]
    assert item.evidence_turn_ids == ["s2:1"]
    assert "[s2:1] assistant: Paris" in item.sessions[1].text
    assert parse_timestamp(item.question_date).year == 2023


def test_locomo_adapter_shares_scope_and_maps_dialog_evidence(tmp_path):
    path = _write(tmp_path / "locomo.json", [{
        "sample_id": "conv1",
        "conversation": {
            "speaker_a": "A", "speaker_b": "B",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [{"speaker": "A", "dia_id": "D1:3", "text": "I went today."}],
        },
        "qa": [{"question": "When?", "answer": "8 May", "evidence": ["D1:3"], "category": 2}],
    }])
    item = load_locomo(path)[0]
    assert item.scope_id == "conv1"
    assert item.evidence_session_ids == ["session_1"]
    assert item.evidence_turn_ids == ["D1:3"]
    assert parse_timestamp(item.sessions[0].timestamp).month == 5


def test_locomo_adapter_repairs_known_malformed_dialog_ids(tmp_path):
    path = _write(tmp_path / "locomo.json", [{
        "sample_id": "conv1",
        "conversation": {
            "session_11_date_time": "1:56 pm on 8 May, 2023",
            "session_11": [{"speaker": "A", "dia_id": "D11:26", "text": "A book"}],
        },
        "qa": [{"question": "Which?", "answer": "A book", "evidence": ["D:11:26", "D"], "category": 1}],
    }])
    item = load_locomo(path)[0]
    assert item.evidence_turn_ids == ["D11:26"]
    assert item.evidence_session_ids == ["session_11"]


def test_retrieval_metrics_and_locomo_scoring():
    values = retrieval_at_k(["s0", "s2", "s1"], ["s1", "s2"], 3)
    assert values["recall_any@3"] == 1
    assert values["recall_all@3"] == 1
    assert 0 < values["ndcg_any@3"] < 1
    partial = retrieval_at_k(["s2"], ["s1", "s2"], 1)
    assert partial["recall_any@1"] == 1
    assert partial["recall_all@1"] == 0
    assert partial["evidence_recall@1"] == 0.5
    assert token_f1("the Business Administration", "Business Administration") == 1
    assert locomo_f1("blue, hiking", "hiking, blue", 1) == 1
    assert locomo_f1("This was not mentioned.", "no event", 5) == 1


def test_isolation_document_ids_include_scope():
    first = _document_id("locomo", "conversation-a", "session_1")
    second = _document_id("locomo", "conversation-b", "session_1")
    assert first != second
    assert first == _document_id("locomo", "conversation-a", "session_1")


def _concurrent_ingest_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    turns = [
        ("session_1", "Alice went hiking with Bob. They talked about the new telescope."),
        ("session_2", "Bob met Carol at the observatory. Carol studies dark matter."),
        ("session_3", "Alice and Carol reviewed the research notes on quantum physics."),
        ("session_4", "Bob told Alice that the telescope arrived at the lab."),
    ]
    dataset = data_dir / "locomo10.json"
    dataset.write_text(json.dumps([{
        "sample_id": "conv-cc",
        "conversation": {
            **{sid: [{"speaker": "S", "text": text, "dia_id": f"{sid}:1"}]
               for sid, text in turns},
            **{f"{sid}_date_time": f"{i + 1}:00 PM on 1 January, 2024"
               for i, (sid, _) in enumerate(turns)},
        },
        "qa": [{
            "question_id": "q/1", "question": "Who went hiking?",
            "answer": "Alice", "category": 1, "evidence": ["session_1:1"],
        }],
    }]), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "storage_path": str(tmp_path / "lib"),
        # llm.mock=true → LLMClient 模拟响应模式（离线测试）
        "llm": {"model": "mock", "mock": True, "context_window_tokens": 32000},
        "embedding": {"model": "/nonexistent-mock-model"},
        "pipeline": {"remember": {
            "profile": "strong-v1",
            "window_size_chars": 6000, "overlap_chars": 300,
        }},
    }), encoding="utf-8")
    return data_dir, config


def test_concurrent_ingest_two_workers(tmp_path):
    data_dir, config = _concurrent_ingest_fixture(tmp_path)
    run_dir = tmp_path / "run-cc"
    result = runner.ingest_benchmark(
        "locomo", data_dir, run_dir, config,
        remember_profile="strong-v1", ingest_workers=2,
    )
    assert result["ingested"] == 4
    assert result["failed"] == 0
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    scope = manifest["scopes"]["conv-cc"]
    assert len(scope["documents"]) == 4
    assert set(scope["visible_sessions"]) == {"session_1", "session_2", "session_3", "session_4"}
    # 并发下不产生重复 family / 不丢文档
    # FamilyWriteGate（registry 默认开启）把并发同名创建收敛到单一 family → 重复率应为 0
    dup = scope.get("duplicate_family_report") or {}
    assert dup.get("duplicate_family_rate", 0) == 0
    assert manifest["library_integrity"]["visible_session_count"] == 4


def test_concurrent_ingest_fail_fast(tmp_path):
    data_dir, config = _concurrent_ingest_fixture(tmp_path)
    from core.remember.orchestrator import TemporalMemoryGraphProcessor
    original = TemporalMemoryGraphProcessor.remember_text
    calls = {"n": 0}

    def flaky(self, text, *a, **k):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("simulated window failure")
        return original(self, text, *a, **k)

    TemporalMemoryGraphProcessor.remember_text = flaky
    try:
        with pytest.raises(RuntimeError):
            runner.ingest_benchmark(
                "locomo", data_dir, tmp_path / "run-fail", config,
                remember_profile="strong-v1", ingest_workers=2,
            )
    finally:
        TemporalMemoryGraphProcessor.remember_text = original
    # fail-fast 与串行路径语义一致：异常抛出，但已处理文档的 failed 状态落库
    import sqlite3
    db = next((tmp_path / "run-fail" / "libraries").glob("conv-cc*/library.db"))
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT state FROM document_ingestion_state").fetchall()
    finally:
        conn.close()
    states = {row[0] for row in rows}
    assert "failed" in states or "processing" in states


def test_longmemeval_prompt_policies():
    temporal = longmemeval_judge_prompt("temporal-reasoning", "q", "a", "r")
    abstention = longmemeval_judge_prompt("multi-session", "q", "a", "r", True)
    assert "off-by-one" in temporal
    assert "unanswerable" in abstention
    assert JUDGE_MODEL == "gpt-4o-2024-08-06"


def test_agentic_loop_can_search_jump_read_and_answer():
    session = MemorySession("s1", "2024-01-01", "[t1] user: Alice lives in Paris", ["t1"])
    item = BenchmarkItem("locomo", "c1", "q1", "Where does Alice live?", "Paris", "1", "", [session], ["s1"], ["t1"])

    class FakeTools:
        sessions = {"s1": session}

        def execute(self, tool, arguments):
            if tool == "search_concepts":
                return [{"family_id": "ent_alice", "name": "Alice"}]
            if tool == "trace_concept":
                return [{"episode_id": "ep1", "session_id": "s1", "turn_ids": ["t1"], "source_text": session.text}]
            if tool == "read_episode":
                return {"episode_id": "ep1", "session_id": "s1", "turn_ids": ["t1"], "source_text": session.text}
            raise AssertionError((tool, arguments))

        def contexts_for_submission(self, session_ids, episode_ids, turn_ids, *, limit):
            assert session_ids == ["s1"]
            assert episode_ids == ["ep1"]
            return ([{"session_id": "s1", "timestamp": session.timestamp, "text": session.text,
                      "turn_ids": ["t1"], "matched_turn_ids": ["t1"], "evidence": []}], ["t1"])

    class ScriptedPolicy:
        actions = [
            {"tool": "search_concepts", "arguments": {"query": "Alice"}},
            {"tool": "trace_concept", "arguments": {"family_id": "ent_alice"}},
            {"tool": "read_episode", "arguments": {"episode_id": "ep1"}},
            {"tool": "submit_evidence", "arguments": {
                "session_ids": ["s1"], "episode_ids": ["ep1"], "turn_ids": ["t1"],
            }},
        ]

        def decide(self, _item, _trajectory, step):
            return {"action": self.actions[step - 1], "prompt": f"step {step}", "prompt_tokens": 2,
                    "completion_tokens": 1, "model": "fake"}

    class SharedAnswerer:
        def answer(self, _item, contexts):
            assert contexts[0]["session_id"] == "s1"
            return {"hypothesis": "Paris", "prompt": "answer", "prompt_tokens": 1,
                    "completion_tokens": 1, "model": "fake"}

    result = AgenticMemoryRunner(FakeTools(), ScriptedPolicy(), SharedAnswerer(), max_steps=6).run(item)
    assert result["hypothesis"] == "Paris"
    assert result["agent_steps"] == 4
    assert result["agent_stop_reason"] == "submit_evidence"
    assert result["retrieved"][0]["session_id"] == "s1"
    assert result["retrieved_turn_ids"] == ["t1"]
    assert [row["tool"] for row in result["trajectory"]] == [
        "search_concepts", "trace_concept", "read_episode", "submit_evidence",
    ]


def test_agentic_loop_bounds_invalid_actions_and_falls_back():
    item = BenchmarkItem("locomo", "c1", "q1", "Unknown?", "none", "5", "", [], [], [])

    class EmptyTools:
        sessions = {}

        def execute(self, *_args):
            raise AssertionError("invalid tools must not execute")

        def contexts_for_submission(self, *_args, **_kwargs):
            return [], []

    class InvalidPolicy:
        def decide(self, *_args):
            return {"tool": "delete_library", "arguments": {}}

    class Fallback:
        def answer(self, _item, contexts):
            assert contexts == []
            return {"hypothesis": "not available", "prompt": "fallback", "prompt_tokens": 1,
                    "completion_tokens": 1, "model": "fake"}

    result = AgenticMemoryRunner(EmptyTools(), InvalidPolicy(), Fallback(), max_steps=2).run(item)
    assert result["agent_steps"] == 2
    assert result["agent_stop_reason"] == "max_steps"
    assert all(row["error"]["type"] == "ValueError" for row in result["trajectory"])
    assert result["hypothesis"] == "not available"


def test_agentic_loop_rejects_direct_final_answer_and_uses_answerer():
    item = BenchmarkItem("locomo", "c1", "q1", "Where?", "Paris", "1", "", [], [], [])

    class EmptyTools:
        sessions = {}

        def contexts_for_submission(self, *_args, **_kwargs):
            return [], []

    class FlatPolicy:
        def decide(self, *_args):
            return {"tool": "final_answer", "answer": "Paris"}

    class SharedAnswerer:
        def answer(self, *_args):
            return {"hypothesis": "No information available.", "prompt": "answer",
                    "prompt_tokens": 1, "completion_tokens": 1, "model": "fake"}

    result = AgenticMemoryRunner(EmptyTools(), FlatPolicy(), SharedAnswerer(), max_steps=1).run(item)
    assert result["hypothesis"] == "No information available."
    assert result["agent_stop_reason"] == "max_steps"
    assert result["trajectory"][0]["error"]["type"] == "ValueError"
    assert result["trajectory"][0].get("tool") is None


def test_agent_action_parser_accepts_plain_and_fenced_json():
    expected = {"tool": "search_memory", "arguments": {"query": "Paris"}}
    assert _parse_json_object(json.dumps(expected)) == expected
    assert _parse_json_object(f"```json\n{json.dumps(expected)}\n```") == expected


def test_agent_collects_singular_ranked_turn_evidence():
    sessions, episodes, turns = [], [], []
    AgenticMemoryRunner._collect_ids(
        {"session_id": "s1", "episode_id": "ep1", "turn_id": "D1:3"},
        sessions, episodes, turns,
    )
    assert (sessions, episodes, turns) == (["s1"], ["ep1"], ["D1:3"])


def test_locomo_answer_prompt_uses_score_compatible_format():
    multi = BenchmarkItem("locomo", "c", "q1", "Name both", "A, B", "1", "", [], [], [])
    adversarial = BenchmarkItem("locomo", "c", "q2", "Unknown?", "none", "5", "", [], [], [])
    temporal = BenchmarkItem("locomo", "c", "q3", "When?", "7 May", "2", "", [], [], [])
    answerer = runner.AnswerGenerator({"llm": {"context_window_tokens": 8000}})
    assert "multiple requested facts separated by commas" in answerer.build_prompt(multi, [])
    assert "No information available." in answerer.build_prompt(adversarial, [])
    assert "false premises" in answerer.build_prompt(adversarial, [])
    assert "resolve relative dates" in answerer.build_prompt(temporal, [])
    assert "full conventional label" in answerer.build_prompt(multi, [])
    assert "question type" not in answerer.build_prompt(adversarial, []).lower()


def test_openai_compatible_chat_forwards_provider_extra_body(monkeypatch):
    from core.llm import chat_api

    captured = {}

    class Completions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return {
                "model": "fake",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    class Client:
        class Chat:
            completions = Completions()

        chat = Chat()

    monkeypatch.setattr(chat_api, "_openai_shared_client", lambda *_args: Client())
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    response = chat_api.openai_compatible_chat(
        [{"role": "user", "content": "hello"}], model="fake", base_url="http://fake/v1",
        api_key="EMPTY", extra_body=extra, temperature=0,
    )
    assert response.content == "ok"
    assert captured["extra_body"] == extra
    assert captured["temperature"] == 0


def test_answerer_reads_named_api_key_env_and_dashscope_thinking_flag(monkeypatch):
    captured = {}

    def fake_chat(_messages, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("TEST_PROVIDER_API_KEY", "private-test-key")
    monkeypatch.setattr("core.llm.chat_api.openai_compatible_chat", fake_chat)
    answerer = runner.AnswerGenerator({"llm": {
        "api_key_env": "TEST_PROVIDER_API_KEY",
        "model": "qwen3.7-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "answer_extra_body": {"enable_thinking": False},
    }})
    answerer._chat([{"role": "user", "content": "ping"}])
    assert captured["api_key"] == "private-test-key"
    assert captured["extra_body"] == {"enable_thinking": False}


def test_answerer_rejects_missing_named_api_key_env(monkeypatch):
    monkeypatch.delenv("MISSING_PROVIDER_API_KEY", raising=False)
    answerer = runner.AnswerGenerator({"llm": {
        "api_key_env": "MISSING_PROVIDER_API_KEY",
        "base_url": "https://example.invalid/v1",
    }})
    with pytest.raises(RuntimeError, match="MISSING_PROVIDER_API_KEY"):
        answerer._chat([{"role": "user", "content": "ping"}])


def test_locomo_semantic_judge_parses_json_and_resumes_1540_protocol(tmp_path, monkeypatch):
    from core.benchmark import judging

    assert judging._parse_label('{"reasoning":"same fact","label":"CORRECT"}') == (
        True, "same fact",
    )
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps({"dataset": "locomo", "schema_version": 4}), encoding="utf-8",
    )
    rows = [
        {"question_id": "q1", "question_type": "4", "question": "Where?",
         "answer": "Paris", "hypothesis": "Paris"},
        {"question_id": "q2", "question_type": "5", "question": "False?",
         "answer": "No", "hypothesis": "No information available."},
    ]
    for track in ("baseline", "candidate"):
        (run / f"results.{track}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
        )
    config = tmp_path / "judge.json"
    config.write_text(json.dumps({"llm": {
        "api_key_env": "TEST_JUDGE_KEY", "model": "judge-model", "base_url": "http://fake/v1",
    }}), encoding="utf-8")
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret")
    calls = []

    def fake_judge(record, _llm, _key):
        calls.append(record["question_id"])
        return {"question_id": record["question_id"], "question_type": record["question_type"],
                "label": "CORRECT", "score": 1.0, "reasoning": "ok", "raw": "{}",
                "latency_seconds": 0.1, "status": "completed"}

    monkeypatch.setattr(judging, "_judge_one", fake_judge)
    first = judging.judge_run(
        run, config, tracks=["baseline", "candidate"], protocol="locomo-1540",
        judge_tag="test", max_workers=2, resume=True,
    )
    second = judging.judge_run(
        run, config, tracks=["baseline", "candidate"], protocol="locomo-1540",
        judge_tag="test", max_workers=2, resume=True,
    )
    assert first["tracks"]["baseline"]["total"] == 1
    assert first["tracks"]["candidate"]["overall"] == 1.0
    assert first["delta"] == 0.0
    assert len(calls) == 2
    assert second["tracks"]["baseline"]["completed"] == 1


def test_locomo_batch_judge_submits_collects_and_resumes(tmp_path, monkeypatch):
    from core.benchmark import judging

    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps({"dataset": "locomo", "schema_version": 4}), encoding="utf-8",
    )
    row = {"question_id": "q1", "question_type": "4", "question": "Where?",
           "answer": "Paris", "hypothesis": "Paris"}
    (run / "results.baseline.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = tmp_path / "judge.json"
    config.write_text(json.dumps({"llm": {
        "api_key_env": "TEST_JUDGE_KEY", "model": "judge-model", "base_url": "http://fake/v1",
        "answer_extra_body": {"enable_thinking": False},
    }}), encoding="utf-8")
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret")

    class Obj:
        def __init__(self, **values):
            self.__dict__.update(values)

    class Counts(Obj):
        def model_dump(self):
            return self.__dict__

    class FakeFiles:
        def create(self, **_kwargs): return Obj(id="file-in")
        def content(self, _file_id):
            payload = {"custom_id": "t0-q0", "response": {"status_code": 200,
                "request_id": "req-1", "body": {"choices": [{"message": {"content":
                    '{"reasoning":"same","label":"CORRECT"}'}}], "usage": {"total_tokens": 9}}}}
            return Obj(content=(json.dumps(payload) + "\n").encode())

    class FakeBatches:
        creates = 0
        def create(self, **_kwargs):
            self.creates += 1
            return Obj(id="batch-1", status="validating")
        def retrieve(self, _batch_id):
            return Obj(status="completed", output_file_id="file-out", error_file_id=None,
                       request_counts=Counts(total=1, completed=1, failed=0), completed_at=1)

    fake = Obj(files=FakeFiles(), batches=FakeBatches())
    monkeypatch.setattr("openai.OpenAI", lambda **_kwargs: fake)
    submitted = judging.batch_judge(
        run, config, action="submit", tracks=["baseline"], judge_tag="test",
    )
    collected = judging.batch_judge(
        run, config, action="collect", tracks=["baseline"], judge_tag="test",
    )
    repeated = judging.batch_judge(
        run, config, action="collect", tracks=["baseline"], judge_tag="test",
    )
    assert submitted["batch_status"]["baseline"]["batch_id"] == "batch-1"
    assert collected["tracks"]["baseline"]["overall"] == 1.0
    assert repeated["tracks"]["baseline"]["completed"] == 1
    assert fake.batches.creates == 1


def test_locomo_score_and_report_are_recomputable(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"dataset": "locomo"}), encoding="utf-8")
    record = {
        "question_id": "q1", "question_type": "2", "question": "Where?", "answer": "Paris",
        "hypothesis": "Paris", "retrieval_metrics": {"session_recall_any@5": 1.0},
    }
    (run / "results.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    summary = score_run(run)
    assert summary["overall"] == 1
    assert (run / "summary.json").exists()
    assert (run / "scored_results.jsonl").exists()
    assert "Overall: 1.0000" in (run / "report.md").read_text()
    assert report_run(run) == summary


def test_schema_v3_scores_tracks_and_writes_comparison(tmp_path):
    run = tmp_path / "run-v3"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({
        "schema_version": 3, "dataset": "locomo", "tracks": ["baseline", "skill-agent"],
    }), encoding="utf-8")
    base = {"question_id": "q1", "question_type": "4", "question": "Where?",
            "answer": "Paris", "hypothesis": "London", "status": "completed",
            "retrieval_metrics": {"session_recall_any@5": 1.0}, "track": "baseline"}
    skill = {**base, "hypothesis": "Paris", "track": "skill-agent"}
    (run / "results.baseline.jsonl").write_text(json.dumps(base) + "\n", encoding="utf-8")
    (run / "results.skill-agent.jsonl").write_text(json.dumps(skill) + "\n", encoding="utf-8")
    result = score_run(run)
    assert result["skill_agent_minus_baseline"]["overall"] == 1.0
    assert (run / "summary.baseline.json").exists()
    assert (run / "summary.skill-agent.json").exists()
    assert (run / "comparison.md").exists()


def test_aggregate_handles_retrieval_only_records():
    summary = aggregate_records([
        {"score": None, "question_type": "x", "retrieval_metrics": {"recall_any@5": 1}},
        {"score": None, "question_type": "x", "retrieval_metrics": {"recall_any@5": 0}},
    ], "longmemeval-s")
    assert summary["overall"] is None
    assert summary["retrieval"]["recall_any@5"] == 0.5


def test_mock_end_to_end_run_is_resumable(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "locomo10.json", [{
        "sample_id": "conv1",
        "conversation": {
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [{"speaker": "A", "dia_id": "D1:3", "text": "Paris"}],
        },
        "qa": [{"question": "Where?", "answer": "Paris", "evidence": ["D1:3"], "category": 2}],
    }])
    config = tmp_path / "config.json"
    _write(config, {"llm": {"model": "fake", "base_url": "http://localhost:1/v1"}, "embedding": {}, "chunking": {}})

    class FakeStorage:
        def close(self):
            pass

    class FakeProcessor:
        def __init__(self):
            self.storage = FakeStorage()
            self.calls = []

        def remember_text(self, text, **kwargs):
            self.calls.append((text, kwargs))
            return {"episode_id": "ep1"}

    class FakeRegistry:
        def __init__(self, *_args, **_kwargs):
            self.processor = FakeProcessor()

        def get_processor(self, _graph):
            return self.processor

    class FakeRetriever:
        def __init__(self, _storage, _mapping, sessions):
            self.session = sessions[0]

        def search(self, _question, **_kwargs):
            return [{
                "session_id": self.session.session_id, "timestamp": self.session.timestamp,
                "text": self.session.text, "turn_ids": self.session.turn_ids,
                "matched_turn_ids": self.session.turn_ids, "score": 1.0, "evidence": [],
            }]

    import core.server.registry as registry_module
    monkeypatch.setattr(registry_module, "GraphRegistry", FakeRegistry)
    monkeypatch.setattr(runner, "DeepDreamRetriever", FakeRetriever)
    monkeypatch.setattr(runner.AnswerGenerator, "answer", lambda self, item, contexts: {
        "hypothesis": "Paris", "prompt": "p", "latency_seconds": 0,
        "prompt_tokens": 1, "completion_tokens": 1, "model": "fake",
    })
    run_dir = tmp_path / "run"
    first = runner.run_benchmark("locomo", data_dir, run_dir, config, limit=1, retrieval_mode="single-pass")
    second = runner.run_benchmark(
        "locomo", data_dir, run_dir, config, limit=1, resume=True, retrieval_mode="single-pass"
    )
    assert first["completed"] == 1
    assert second["processed"] == 0
    records = (run_dir / "results.jsonl").read_text().splitlines()
    assert len(records) == 1
    result = json.loads(records[0])
    assert result["retrieval_metrics"]["session_recall_any@5"] == 1
    assert result["retrieval_metrics"]["turn_recall_any@5"] == 1

    class FakePolicy:
        def decide(self, _item, _trajectory, step):
            if step == 1:
                action = {"tool": "explore_memory", "arguments": {"query": "Paris", "limit": 5}}
            else:
                action = {"tool": "submit_evidence", "arguments": {
                    "session_ids": ["session_1"], "episode_ids": ["ep1"], "turn_ids": ["D1:3"],
                }}
            return {"action": action, "prompt": f"step {step}", "model": "fake"}

    class FakeAgentTools:
        def __init__(self, _storage, _retriever, _mapping, sessions):
            self.sessions = {session.session_id: session for session in sessions}

        def execute(self, tool, _arguments):
            assert tool == "explore_memory"
            return {"session_id": "session_1", "episode_id": "ep1", "turn_ids": ["D1:3"]}

        def contexts_for_submission(self, _sessions, _episodes, _turns, *, limit):
            session = self.sessions["session_1"]
            return ([{"session_id": session.session_id, "timestamp": session.timestamp,
                      "text": session.text, "turn_ids": session.turn_ids,
                      "matched_turn_ids": session.turn_ids, "evidence": []}], session.turn_ids)

    import core.benchmark.agentic as agentic_module
    monkeypatch.setattr(agentic_module, "AgentDecisionModel", lambda _config: FakePolicy())
    monkeypatch.setattr(agentic_module, "AgenticMemoryTools", FakeAgentTools)
    agentic_dir = tmp_path / "agentic-run"
    agentic_result = runner.run_benchmark(
        "locomo", data_dir, agentic_dir, config, limit=1, retrieval_mode="agentic"
    )
    assert agentic_result["completed"] == 1
    agentic_record = json.loads((agentic_dir / "results.jsonl").read_text().splitlines()[0])
    assert agentic_record["track"] == "skill-agent"
    assert agentic_record["hypothesis"] == "Paris"
    assert [step["tool"] for step in agentic_record["trajectory"]] == ["explore_memory", "submit_evidence"]
    assert agentic_record["retrieval_metrics"]["turn_recall_any@5"] == 1


def test_normalized_answer_profile_serializes_dates_booleans_and_false_premises():
    answerer = runner.AnswerGenerator({"llm": {}}, profile="normalized-v1")
    contexts = [{
        "session_id": "session_3",
        "timestamp": "7:55 pm on 9 June, 2023",
        "text": "[D3:1] Caroline: I gave my school speech last week.",
    }]
    temporal = BenchmarkItem(
        "locomo", "c", "q1", "When did Caroline give the speech?", "", "2", "", [], [], [],
    )
    hypothesis, payload = answerer._normalize_payload(temporal, contexts, {
        "support": "supported", "answer_type": "date", "answer": "last week",
    })
    assert hypothesis == "The week before 9 June 2023"
    assert payload["answer"] == hypothesis

    yes_item = BenchmarkItem(
        "locomo", "c", "q2", "Did Melanie make the bowl?", "", "4", "", [], [], [],
    )
    assert answerer._normalize_payload(yes_item, contexts, {
        "support": "supported", "answer_type": "boolean", "answer": "Yes, she did.",
    })[0] == "Yes"
    assert answerer._normalize_payload(yes_item, contexts, {
        "support": "supported", "answer_type": "boolean", "answer": "No, Caroline did.",
    })[0] == "No information available."


def test_submission_validation_only_checks_selected_turns_for_indirect_references():
    tools = object.__new__(AgenticMemoryTools)
    tools.sessions = {"session_3": object()}
    tools.contexts_for_submission = lambda *_args, **_kwargs: ([{
        "session_id": "session_3",
        "text": "[D3:1] Caroline: I gave a school speech last week.\n"
                "[D3:13] Caroline: I moved from my home country.",
    }], [])
    tools.validate_submission(["session_3"], [], ["D3:1"])
    with pytest.raises(ValueError, match="home country"):
        tools.validate_submission(["session_3"], [], ["D3:13"])


def test_turn_only_submission_infers_its_session_context():
    session = MemorySession("session_1", "2024-01-01", "[D1:3] Alice: Paris", ["D1:3"])
    tools = AgenticMemoryTools.__new__(AgenticMemoryTools)
    tools.sessions = {session.session_id: session}
    tools.turn_to_session = {"D1:3": "session_1"}
    tools._episode_payload = lambda _episode_id: (_ for _ in ()).throw(AssertionError())
    contexts, turns = tools.contexts_for_submission([], [], ["D1:3"], limit=5)
    assert turns == ["D1:3"]
    assert contexts[0]["session_id"] == "session_1"
    assert contexts[0]["matched_turn_ids"] == ["D1:3"]


def test_agent_thinking_override_is_isolated_in_extra_body():
    config = {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "agent_think": True,
    }
    assert _agent_extra_body(config, True)["chat_template_kwargs"]["enable_thinking"] is True
    assert config["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_answer_replay_materializes_retrieval_cache_without_rerunning_tools(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dataset = _write(data_dir / "locomo10.json", [{
        "sample_id": "conv1",
        "conversation": {
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [{"speaker": "A", "dia_id": "D1:3", "text": "Paris"}],
        },
        "qa": [{"question": "Where?", "answer": "Paris", "evidence": ["D1:3"], "category": 2}],
    }])
    config = _write(tmp_path / "config.json", {
        "llm": {"model": "fake", "base_url": "http://localhost:1/v1"},
        "embedding": {}, "chunking": {},
    })
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "run_manifest.json", {
        "schema_version": 3,
        "dataset": "locomo",
        "dataset_path": str(dataset),
        "dataset_sha256": runner.sha256_file(dataset),
        "tracks": ["skill-agent"],
        "answer_top_k": 5,
        "config": {"llm": {"think": False}},
        "scopes": {"conv1": {"visible_sessions": ["session_1"]}},
    })
    old = {
        "dataset": "locomo", "scope_id": "conv1", "question_id": "conv1:0",
        "question_type": "2", "question": "Where?", "answer": "Paris", "question_date": "",
        "evidence_session_ids": ["session_1"], "evidence_turn_ids": ["D1:3"],
        "track": "skill-agent", "status": "completed", "hypothesis": "old answer",
        "prompt": "old prompt", "answer_latency_seconds": 2.0, "total_latency_seconds": 5.0,
        "retrieved": [{"session_id": "session_1", "timestamp": "1:56 pm on 8 May, 2023",
                       "text": "[D1:3] A: Paris"}],
        "ranked_session_ids": ["session_1"], "ranked_turn_ids": ["D1:3"],
        "retrieval_metrics": {"session_recall_any@5": 1.0},
        "trajectory": [{"step": 1, "tool": "submit_evidence", "prompt_tokens": 3,
                        "completion_tokens": 1}],
        "agent_steps": 1, "agent_stop_reason": "submit_evidence",
    }
    (run_dir / "results.skill-agent.jsonl").write_text(json.dumps(old) + "\n", encoding="utf-8")
    monkeypatch.setattr(runner.AnswerGenerator, "answer", lambda self, item, contexts: {
        "hypothesis": "Paris", "prompt": "new prompt", "answer_latency_seconds": 0.1,
        "prompt_tokens": 4, "completion_tokens": 1, "model": "fake",
        "answer_profile": "normalized-v1",
    })
    result = runner.replay_answers(
        run_dir, config, tracks=["skill-agent"], result_tag="answer-v1",
    )
    assert result["retrieval_reused"] is True
    cache = json.loads((run_dir / "retrieval.skill-agent.jsonl").read_text())
    assert "hypothesis" not in cache
    assert cache["retrieved"][0]["session_id"] == "session_1"
    replayed = json.loads((run_dir / "results.skill-agent-answer-v1.jsonl").read_text())
    assert replayed["hypothesis"] == "Paris"
    assert replayed["trajectory"] == old["trajectory"]
    assert replayed["answer_replay"]["retrieval_reused"] is True


def test_retrieve_then_answer_v4_is_resumable_and_keeps_old_tracks(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dataset = _write(data_dir / "locomo10.json", [{
        "sample_id": "conv1",
        "conversation": {
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [{"speaker": "A", "dia_id": "D1:3", "text": "Paris"}],
        },
        "qa": [{"question": "Where?", "answer": "Paris", "evidence": ["D1:3"], "category": 2}],
    }])
    config = _write(tmp_path / "config.json", {
        "llm": {"model": "fake", "base_url": "http://localhost:1/v1"},
        "embedding": {}, "chunking": {},
    })
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    library_dir = run_dir / "library"
    library_dir.mkdir()
    _write(run_dir / "run_manifest.json", {
        "schema_version": 3, "dataset": "locomo", "dataset_path": str(dataset),
        "dataset_sha256": runner.sha256_file(dataset), "tracks": ["baseline"],
        "answer_top_k": 5, "config": {"llm": {"think": False}},
        "scopes": {"conv1": {
            "library_dir": "library", "visible_sessions": ["session_1"],
            "documents": {"session_1": {"document_id": "doc1", "status": "active"}},
        }},
    })

    class FakeStorage:
        def close(self):
            pass

    class FakeProcessor:
        storage = FakeStorage()

    class FakeRegistry:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_processor(self, _graph):
            return FakeProcessor()

    class FakeRetriever:
        calls = 0

        def __init__(self, _storage, _mapping, sessions, **_kwargs):
            self.session = sessions[0]

        def explore(self, _question, **_kwargs):
            FakeRetriever.calls += 1
            context = {
                "session_id": "session_1", "timestamp": self.session.timestamp,
                "text": self.session.text, "turn_ids": ["D1:3"],
                "matched_turn_ids": ["D1:3"], "score": 1.0, "evidence": [],
            }
            return {
                "contexts": [context], "ranked_session_ids": ["session_1"],
                "ranked_turn_ids": ["D1:3"], "retrieval_profile": "hybrid-v2",
                "query_terms": [{"term": "Where?", "source": "original"}],
                "turn_scores": [{"turn_id": "D1:3", "score": 1.0}],
                "budget": {"limit": 1600, "used": 2},
                "profile": {"profile": "hybrid-v2", "sha256": "a" * 64},
                "explore": {"coverage": {"episode_hits": 1}},
            }

    import core.server.registry as registry_module
    monkeypatch.setattr(registry_module, "GraphRegistry", FakeRegistry)
    monkeypatch.setattr(runner, "DeepDreamRetriever", FakeRetriever)
    first = runner.retrieve_benchmark(
        run_dir, config, tracks=["baseline"], retrieval_profile="hybrid-v2",
        result_tag="hybrid-v2", resume=True,
    )
    second = runner.retrieve_benchmark(
        run_dir, config, tracks=["baseline"], retrieval_profile="hybrid-v2",
        result_tag="hybrid-v2", resume=True,
    )
    assert first["processed"] == 1
    assert second["processed"] == 0
    assert FakeRetriever.calls == 1
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["schema_version"] == 4
    assert "baseline" in manifest["tracks"]
    assert "baseline-hybrid-v2" in manifest["tracks"]
    assert manifest["retrieval_profiles"]["baseline-hybrid-v2"]["candidate_k"] == 30

    monkeypatch.setattr(runner.AnswerGenerator, "answer", lambda self, item, contexts: {
        "hypothesis": "Paris", "prompt": "new", "answer_latency_seconds": 0.1,
        "prompt_tokens": 1, "completion_tokens": 1, "model": "fake",
        "answer_profile": "normalized-v1",
    })
    answered = runner.replay_answers(
        run_dir, config, tracks=["baseline-hybrid-v2"], result_tag="answer-v1",
    )
    assert answered["tracks"] == ["baseline-hybrid-v2-answer-v1"]
    record = json.loads(
        (run_dir / "results.baseline-hybrid-v2-answer-v1.jsonl").read_text()
    )
    assert record["hypothesis"] == "Paris"
    assert record["retrieval_audit"]["budget"]["limit"] == 1600


def test_cached_turn_only_submission_is_reassembled_without_retrieval():
    session = MemorySession(
        "session_3", "1:00 pm on 8 May, 2023",
        "[D3:13] A: I grew up in Sweden.", ["D3:13"],
    )
    item = BenchmarkItem(
        "locomo", "conv1", "conv1:0", "Where did A grow up?", "Sweden", "1", "",
        [session], ["session_3"], ["D3:13"],
    )
    repaired = runner._reassemble_turn_only_contexts(item, {
        "submitted_evidence": {"session_ids": [], "turn_ids": ["D3:13"]},
        "retrieved": [],
        "ranked_session_ids": [],
        "ranked_turn_ids": ["D3:13"],
    })
    assert repaired["evidence_reassembled_from_cache"] is True
    assert repaired["ranked_session_ids"] == ["session_3"]
    assert repaired["retrieved"][0]["matched_turn_ids"] == ["D3:13"]
    assert repaired["retrieval_metrics"]["session_recall_any@5"] == 1
