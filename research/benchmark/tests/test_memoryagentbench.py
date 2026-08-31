"""MemoryAgentBench 数据加载、官方 scorer 与分层抽样的单元测试。"""
import hashlib
import json
from pathlib import Path

import pytest

from research.benchmark.datasets import (
    BenchmarkItem,
    _partition_exact_text,
    load_benchmark,
    load_memoryagentbench,
)
from research.benchmark.mab_sample import build_scope_records, select_scopes, source_family
from research.benchmark.score_memoryagentbench_official import (
    deterministic_score,
    macro_summary,
    normalize_answer,
    parse_answer,
    parse_json,
)


# ---------------------------------------------------------------- partition


@pytest.mark.parametrize(
    "text",
    [
        "",
        "short text",
        "no newline at all " * 9000,
        "line\n" * 30000,
        "para\n\npara\n\n" * 8000,
        "x" * 119_999 + "\n" + "y" * 5000,
    ],
)
def test_partition_exact_text_preserves_source(text):
    chunks = _partition_exact_text(text, max_chars=120_000)
    assert "".join(chunks) == text
    assert all(len(chunk) <= 120_000 for chunk in chunks)
    assert (chunks == []) if not text else (chunks and all(chunks))


def test_partition_snaps_to_newline_boundary():
    text = "".join(f"line {i:05d}\n" for i in range(40000))
    chunks = _partition_exact_text(text, max_chars=1000)
    assert "".join(chunks) == text
    # 非末块应在换行边界收尾，且保留至少 75% 的块长下限。
    for chunk in chunks[:-1]:
        assert chunk.endswith("\n")
        assert len(chunk) >= 750


# ------------------------------------------------------------------- loader


def _write_shard(directory: Path, stem: str, rows: list[dict]) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    directory.mkdir(parents=True, exist_ok=True)
    name = f"{stem}-00000-of-00001.parquet"
    path = directory / name
    # 类型推断即可：两个分片分别是嵌套/扁平两种 keypoints 编码。
    table = pa.Table.from_pylist(rows)
    parquet.write_table(table, path)
    return {
        "filename": name,
        # 故意记录一个失效的绝对路径：loader 必须回退到 manifest 同目录。
        "path": "/definitely/stale/" + name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _make_dataset(root: Path) -> Path:
    ar_rows = [{
        "context": "alpha session\n\nbeta session\ngamma line",
        "questions": ["Q1?", "Q2?"],
        "answers": [["A1"], ["A2a", "A2b"]],
        "metadata": {
            "qa_pair_ids": ["ar_no0", "ar_no1"],
            "question_types": ["qa", "qa"],
            "question_dates": ["", "2024-01-01"],
            "keypoints": [["k1"], ["k2a", "k2b"]],
            "source": "ruler_qa1_197K",
        },
    }]
    ttl_rows = [{
        # InfBench 式编码：单题 + 扁平 keypoints 列表。
        "context": "\n".join(f"story paragraph {i}" for i in range(200)),
        "questions": ["Summarize."],
        "answers": [["reference summary"]],
        "metadata": {
            "qa_pair_ids": ["sum_no0"],
            "question_types": ["summarization"],
            "question_dates": [""],
            "keypoints": ["point one", "point two", "point three"],
            "source": "infbench_sum_eng_shots2",
        },
    }]
    manifest = {
        "dataset": "memoryagentbench",
        "revision": "test",
        "files": [
            _write_shard(root / "data", "Accurate_Retrieval", ar_rows),
            _write_shard(root / "data", "Test_Time_Learning", ttl_rows),
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_load_memoryagentbench_synthetic(tmp_path: Path):
    manifest_path = _make_dataset(tmp_path)
    items = load_memoryagentbench(manifest_path)
    assert [i.question_id for i in items] == [
        "mab-ar-000:ar_no0", "mab-ar-000:ar_no1", "mab-ttl-000:sum_no0",
    ]
    first, second, summary = items
    assert first.scope_id == "mab-ar-000"
    assert first.answer == "A1"
    assert first.metadata["answer_aliases"] == ["A1"]
    assert first.judge_rubric == ["k1"]
    assert first.visible_session_ids == ["context_0000"]
    assert first.metadata["competency"] == "AR"
    assert first.metadata["official_source"] == "ruler_qa1_197K"
    assert second.metadata["answer_aliases"] == ["A2a", "A2b"]
    assert second.judge_rubric == ["k2a", "k2b"]
    assert second.question_date == "2024-01-01"
    # 扁平 keypoints（InfBench 编码）不得截断成第一个点。
    assert summary.judge_rubric == ["point one", "point two", "point three"]
    context = "alpha session\n\nbeta session\ngamma line"
    assert "".join(s.text for s in first.sessions) == context
    assert first.metadata["context_sha256"] == hashlib.sha256(context.encode()).hexdigest()
    assert first.metadata["context_documents"] == 1


def test_load_memoryagentbench_rejects_hash_drift(tmp_path: Path):
    manifest_path = _make_dataset(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        load_memoryagentbench(manifest_path)


def test_load_benchmark_dispatches_memoryagentbench(tmp_path: Path):
    # 与真实布局一致：<data_dir>/memoryagentbench/{manifest.json,data/*.parquet}
    _make_dataset(tmp_path / "memoryagentbench")
    items, resolved = load_benchmark("memoryagentbench", tmp_path)
    assert len(items) == 3
    assert resolved.name == "manifest.json"


def test_benchmark_item_new_fields_default_empty():
    item = BenchmarkItem(
        dataset="d", scope_id="s", question_id="q", question="Q", answer="A",
        question_type="t", question_date="", sessions=[], evidence_session_ids=[],
    )
    assert item.judge_rubric == []
    assert item.visible_session_ids == []


# ------------------------------------------------------------------- scorer


def test_scorer_text_helpers():
    assert normalize_answer("The Answer!") == "answer"
    assert parse_answer("blah\nAnswer: Foo Bar\nnext") == "Foo Bar"
    assert parse_answer("only line") == "only line"


def test_scorer_deterministic_score_modes():
    def item(source, aliases, answer="gold"):
        return BenchmarkItem(
            dataset="memoryagentbench", scope_id="s", question_id="q",
            question="Q", answer=answer, question_type="t", question_date="",
            sessions=[], evidence_session_ids=[],
            metadata={"official_source": source, "answer_aliases": aliases},
        )

    exact = item("icl_banking77_x", ["Class A"])
    assert deterministic_score(exact, "Answer: class a") == 1.0
    assert deterministic_score(exact, "Answer: totally wrong") == 0.0
    # icl/detective 走整串精确匹配：包含不够。
    assert deterministic_score(exact, "Answer: it is Class A indeed") == 0.0
    sub = item("ruler_qa1_197K", ["France"])
    assert deterministic_score(sub, "Answer: France") == 1.0
    assert deterministic_score(sub, "the answer is France.") == 1.0
    assert parse_json('noise {"recall": 3} trailing') == {"recall": 3}


def _row(source, score, scope="s0"):
    return {"source": source, "score": score, "scope_id": scope, "question_id": f"{source}-{score}"}


def test_scorer_macro_summary_matches_official_formula():
    rows = [
        _row("ruler_qa1_197K", 1.0), _row("ruler_qa1_197K", 0.0),           # 0.5
        _row("ruler_qa2_421K", 0.25),                                        # 0.25
        _row("longmemeval_s*", 0.5),                                         # 0.5
        _row("eventqa_65536", 0.0), _row("eventqa_131072", 0.5),             # 0.25
        _row("icl_a", 0.5), _row("icl_b", 1.0),                              # 0.75
        _row("recsys_redial_full", 0.5),                                     # 0.5
        _row("infbench_sum_eng_shots2", 0.5),                                # 0.5
        _row("detective_qa", 1.0),                                           # 1.0
        _row("factconsolidation_sh_6k", 1.0),                                # 1.0
        _row("factconsolidation_mh_6k", 0.5),                                # 0.5
    ]
    table = macro_summary(rows)["table3"]
    assert table["AR"]["SH-QA"] == pytest.approx(0.5)
    assert table["AR"]["MH-QA"] == pytest.approx(0.25)
    assert table["AR"]["EventQA"] == pytest.approx(0.25)
    assert table["AR"]["Avg"] == pytest.approx((0.5 + 0.25 + 0.5 + 0.25) / 4)
    assert table["TTL"]["MCC"] == pytest.approx(0.75)
    assert table["TTL"]["Avg"] == pytest.approx((0.75 + 0.5) / 2)
    assert table["LRU"]["Avg"] == pytest.approx((0.5 + 1.0) / 2)
    assert table["SF"]["Avg"] == pytest.approx((1.0 + 0.5) / 2)
    assert table["Overall"] == pytest.approx(((0.375) + (0.625) + (0.75) + (0.75)) / 4)


def test_scorer_macro_summary_marks_sampled_sources_missing():
    summary = macro_summary([_row("ruler_qa1_197K", 1.0), _row("detective_qa", 0.0)])
    table = summary["table3"]
    assert table["AR"]["MH-QA"] is None
    assert table["AR"]["Avg"] == 1.0
    assert table["TTL"]["Avg"] is None
    assert summary["source_scores"]["ruler_qa1_197K"]["questions"] == 1
    assert summary["source_scores"]["ruler_qa1_197K"]["scopes"] == 1
    assert table["Overall"] == pytest.approx((1.0 + 0.0) / 2)


# ---------------------------------------------------------------- sampling


def test_source_family_grouping():
    assert source_family("icl_banking77_5900shot_balance") == "icl"
    assert source_family("factconsolidation_sh_6k") == "factconsolidation_sh"
    assert source_family("factconsolidation_mh_262k") == "factconsolidation_mh"
    assert source_family("longmemeval_s*") == "longmemeval_s"
    assert source_family("eventqa_65536") == "eventqa_65536"
    assert source_family("detective_qa") == "detective_qa"


def _record(scope, comp, source, chars, questions=1):
    return {
        "scope_id": scope, "competency": comp, "official_source": source,
        "questions": questions, "context_chars": chars,
        "context_documents": 1, "context_sha256": "",
    }


def test_select_scopes_keeps_families_and_respects_budget():
    records = [
        _record("ttl-0", "TTL", "recsys_redial_full", 500),
        _record("ttl-1", "TTL", "icl_a", 100),
        _record("ttl-2", "TTL", "icl_b", 90),
        _record("cr-0", "CR", "factconsolidation_sh_6k", 30),
        _record("cr-1", "CR", "factconsolidation_sh_32k", 60),
        _record("ar-0", "AR", "ruler_qa1_197K", 200),
        _record("ar-1", "AR", "longmemeval_s*", 150),
        _record("ar-2", "AR", "longmemeval_s*", 140),
        _record("lru-0", "LRU", "infbench_sum_eng_shots2", 40),
        _record("lru-1", "LRU", "infbench_sum_eng_shots2", 50),
    ]
    selected, dropped = select_scopes(records, budget_chars=10_000)
    ids = {r["scope_id"] for r in selected}
    # TTL/CR 全收，longmemeval 取最小 2 个，其余 family 各 1 个最小代表。
    assert ids == {"ttl-0", "ttl-1", "ttl-2", "cr-0", "cr-1", "ar-0", "ar-1", "ar-2", "lru-0", "lru-1"}
    assert not dropped

    tight_selected, tight_dropped = select_scopes(records, budget_chars=1000)
    tight_ids = {r["scope_id"] for r in tight_selected}
    assert sum(r["context_chars"] for r in tight_selected) <= 1000
    # 预算内先丢"多代表 family"里最大的（ar-1），family 代表仍在（ar-2）。
    assert "ar-1" in {r["scope_id"] for r in tight_dropped}
    assert "ar-2" in tight_ids
    # 每 family 至少留一个代表，除非总预算装不下最小代表集。
    for family in ("icl", "factconsolidation_sh", "ruler_qa1_197K", "infbench_sum_eng_shots2"):
        assert any(
            source_family(r["official_source"]) == family for r in tight_selected
        )


def test_build_scope_records_counts_questions_once():
    from research.benchmark.datasets import MemorySession
    session = MemorySession("context_0000", "", "x" * 10)
    items = [
        BenchmarkItem(dataset="memoryagentbench", scope_id="mab-ar-000",
                      question_id="q1", question="Q", answer="A", question_type="t",
                      question_date="", sessions=[session], evidence_session_ids=[],
                      metadata={"competency": "AR", "official_source": "s",
                                "context_documents": 1, "context_sha256": "h"}),
        BenchmarkItem(dataset="memoryagentbench", scope_id="mab-ar-000",
                      question_id="q2", question="Q", answer="A", question_type="t",
                      question_date="", sessions=[session], evidence_session_ids=[],
                      metadata={"competency": "AR", "official_source": "s",
                                "context_documents": 1, "context_sha256": "h"}),
    ]
    records = build_scope_records(items)
    assert records == [{
        "scope_id": "mab-ar-000", "competency": "AR", "official_source": "s",
        "questions": 2, "context_chars": 10, "context_documents": 1,
        "context_sha256": "h",
    }]
