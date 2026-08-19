"""strong-v1 单遍抽取（extract_window_structured）单测：全 mock LLM，无真实调用。

覆盖：
- mock 分支命中（<窗口文本> + 请一次性抽取）
- 干净 fence 解析 / 去重 / 端点过滤 / 硬截断
- 脏响应一次重试后成功
- 两次均失败 → ValueError
- known_entity_names 注入 prompt
"""
import pytest

from core.llm.client import LLMClient
from core.llm.mock_response import mock_llm_response

WINDOW = "Alice met Bob at the cafe. Alice and Bob discussed quantum physics. Carol joined later."


def _mock_client(**kwargs) -> LLMClient:
    return LLMClient(api_key="k", model_name="mock", context_window_tokens=8192, **kwargs)


@pytest.fixture
def client():
    c = _mock_client()
    # mock 模式：_call_llm 走 mock_llm_response
    return c


class TestMockBranch:
    def test_mock_response_matches_structured_prompt(self):
        prompt = (
            "<窗口文本>\n" + WINDOW + "\n</窗口文本>\n\n"
            "请一次性抽取上述窗口文本中的概念实体与实体间关系。"
        )
        out = mock_llm_response(prompt)
        assert "```json" in out
        import json
        data = json.loads(out.split("```json", 1)[1].split("```", 1)[0])
        assert "entities" in data and "relations" in data
        names = [e["name"] for e in data["entities"]]
        assert "Alice" in names and "Bob" in names
        # 关系端点全部来自实体列表
        for rel in data["relations"]:
            assert rel["entity1_name"] in names and rel["entity2_name"] in names

    def test_mock_branch_not_shadowed_by_legacy_branches(self):
        # 不含 <窗口文本> 的旧 prompt 不应命中新分支
        prompt = "给定概念列表：Alice、Bob\n请输出关系对数组"
        out = mock_llm_response(prompt)
        assert "一次性抽取" not in out


class TestExtractWindowStructured:
    def test_clean_parse_with_mock_llm(self, client, monkeypatch):
        monkeypatch.setattr(client, "_call_llm", lambda *a, **k: mock_llm_response(
            "<窗口文本>\n" + WINDOW + "\n</窗口文本>\n请一次性抽取"
        ))
        result = client.extract_window_structured(WINDOW)
        assert isinstance(result, dict)
        assert result["entities"] and result["relations"]
        names = {e["name"] for e in result["entities"]}
        assert "Alice" in names
        for rel in result["relations"]:
            assert rel["entity1_name"] in names
            assert rel["entity2_name"] in names
            assert rel["content"]

    def test_dedup_and_endpoint_filter(self, client, monkeypatch):
        dirty = """```json
{"entities": [
    {"name": "Alice", "content": "c1"},
    {"name": "alice", "content": "dup-casefold"},
    {"name": "", "content": "no name"},
    {"name": "Ghost", "content": "未在关系里出现没关系，实体保留"}
],
 "relations": [
    {"entity1_name": "Alice", "entity2_name": "Bob", "content": "端点不存在"},
    {"entity1_name": "Alice", "entity2_name": "alice", "content": "自环"},
    {"entity1_name": "Alice", "entity2_name": "Ghost", "content": "有效"}
]}
```"""
        monkeypatch.setattr(client, "_call_llm", lambda *a, **k: dirty)
        result = client.extract_window_structured(WINDOW, max_entities=10, max_relations=10)
        names = [e["name"] for e in result["entities"]]
        assert names.count("Alice") == 1
        assert "alice" not in names
        rels = result["relations"]
        assert len(rels) == 1
        assert {rels[0]["entity1_name"], rels[0]["entity2_name"]} == {"Alice", "Ghost"}

    def test_hard_caps(self, client, monkeypatch):
        entities = [{"name": f"E{i}", "content": "c"} for i in range(30)]
        relations = [
            {"entity1_name": "E0", "entity2_name": f"E{i}", "content": "c"}
            for i in range(1, 25)
        ]
        payload = ("```json\n" + _dumps({"entities": entities, "relations": relations}) + "\n```")
        monkeypatch.setattr(client, "_call_llm", lambda *a, **k: payload)
        result = client.extract_window_structured(WINDOW, max_entities=5, max_relations=7)
        assert len(result["entities"]) == 5
        assert len(result["relations"]) == 7

    def test_dirty_then_retry_succeeds(self, client, monkeypatch):
        responses = iter([
            "前置思考：这窗口讲的是 Alice。\n```json\n{\"entities\": [{\"name\": \"Alice\"",
            "```json\n" + _dumps({
                "entities": [{"name": "Alice", "content": "人物。"}],
                "relations": [{"entity1_name": "Alice", "entity2_name": "Bob", "content": "无效端点"}],
            }) + "\n```",
        ])
        seen_prompts = []
        calls = {"n": 0}

        def fake_call(task, **kw):
            calls["n"] += 1
            msg = kw.get("messages") or (kw.get("messages_list") or [])
            if msg:
                seen_prompts.append(msg[-1]["content"])
            return next(responses)

        monkeypatch.setattr(client, "_call_llm", fake_call)
        result = client.extract_window_structured(WINDOW)
        assert calls["n"] == 2
        assert result["entities"][0]["name"] == "Alice"
        assert result["relations"] == []  # Bob 不是实体，关系被端点过滤

    def test_both_attempts_fail_raises(self, client, monkeypatch):
        monkeypatch.setattr(client, "_call_llm", lambda *a, **k: "完全不是 JSON")
        with pytest.raises(ValueError):
            client.extract_window_structured(WINDOW)

    def test_empty_result_raises_and_retries(self, client, monkeypatch):
        # 空 entities+空 relations 视为解析失败 → 重试一次 → 仍失败抛 ValueError
        payload = "```json\n" + _dumps({"entities": [], "relations": []}) + "\n```"
        calls = {"n": 0}

        def fake_call(*a, **k):
            calls["n"] += 1
            return payload

        monkeypatch.setattr(client, "_call_llm", fake_call)
        with pytest.raises(ValueError):
            client.extract_window_structured(WINDOW)
        assert calls["n"] == 2

    def test_known_entity_names_in_prompt(self, client, monkeypatch):
        captured = {}

        def fake_call(task, **kw):
            msg = kw.get("messages") or []
            captured["prompt"] = msg[-1]["content"] if msg else ""
            return "```json\n" + _dumps({
                "entities": [{"name": "Alice", "content": "人物。"}],
                "relations": [],
            }) + "\n```"

        monkeypatch.setattr(client, "_call_llm", fake_call)
        client.extract_window_structured(WINDOW, known_entity_names=["Alice", "Bob"])
        assert "库中已有实体名" in captured["prompt"]
        assert "Alice" in captured["prompt"]
        assert "Bob" in captured["prompt"]

    def test_name_grounding_constraint_in_prompt(self, client, monkeypatch):
        """矩阵 C 教训：prompt 必须硬约束名称逐字取自原文（禁翻译/改写）。"""
        captured = {}

        def fake_call(task, **kw):
            msg = kw.get("messages") or []
            captured["prompt"] = msg[-1]["content"] if msg else ""
            return "```json\n" + _dumps({
                "entities": [{"name": "Alice", "content": "人物。"}],
                "relations": [],
            }) + "\n```"

        monkeypatch.setattr(client, "_call_llm", fake_call)
        client.extract_window_structured(WINDOW)
        assert "逐字取自" in captured["prompt"]
        assert "禁止翻译" in captured["prompt"]

    def test_filter_entity_names_grounding_toggle(self):
        """strong-v1 关闭落地硬杀：非逐字名保留；噪声/问句过滤仍生效。"""
        from core.remember.quality import filter_entity_names
        src = "Alice met Bob at the cafe."
        names = ["Alice", "quantum physics research", "hi", "what is this?"]
        strict = filter_entity_names(names, src)
        assert "Alice" in strict and "quantum physics research" not in strict
        loose = filter_entity_names(names, src, require_grounding=False)
        assert "Alice" in loose and "quantum physics research" in loose
        assert "hi" not in loose and "what is this?" not in loose

    def test_distill_step_set_during_call(self, client, monkeypatch):
        seen_steps = []

        def fake_call(*a, **k):
            seen_steps.append(getattr(client, "_current_distill_step", None))
            return "```json\n" + _dumps({
                "entities": [{"name": "Alice", "content": "人物。"}],
                "relations": [],
            }) + "\n```"

        monkeypatch.setattr(client, "_call_llm", fake_call)
        client.extract_window_structured(WINDOW)
        assert seen_steps == ["02s_onepass_extract"]
        assert getattr(client, "_current_distill_step", None) != "02s_onepass_extract"


def _dumps(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)


# ----------------------------------------------------------------------
# strong-v1 端到端（mock LLM：LLMClient 无 key/端点 → mock_llm_response）
# ----------------------------------------------------------------------

STRONG_TEXT = (
    "Alice met Bob at the cafe. Alice and Bob discussed quantum physics for hours. "
    "Carol joined later and told Alice about the new telescope at the observatory. "
    "Bob said the telescope could help their research on dark matter."
)


class TestStrongV1PipelineEndToEnd:
    def _processor(self, tmp_path, remember_config):
        from core.remember.orchestrator import TemporalMemoryGraphProcessor
        from core.storage.embedding import EmbeddingClient
        emb = EmbeddingClient(model_path="/nonexistent/mock-model", use_local=True)
        emb.model = None  # 强制文本相似度回退，不加载真实模型
        return TemporalMemoryGraphProcessor(
            storage_path=str(tmp_path / "lib"),
            embedding_client=emb,
            remember_config=remember_config,
        )

    def test_strong_v1_ingests_and_writes_graph(self, tmp_path):
        proc = self._processor(tmp_path, {
            "profile": "strong-v1",
            "mode": "strong_one_pass",
            "window_size_chars": 6000,
            "overlap_chars": 300,
        })
        assert proc.remember_mode == "strong_one_pass"
        assert proc.remember_profile == "strong-v1"
        assert proc.document_processor.window_size == 6000

        result = proc.remember_text(STRONG_TEXT, doc_name="strong_doc.md", verbose=False)
        assert result.get("chunks_processed", 0) >= 1
        assert result.get("entities", 0) >= 1
        assert result.get("relations", 0) >= 1
        assert result.get("llm_call_stats")

        # 单遍抽取生效：一个小文档的调用数远低于弱模型阶梯（40~80/窗口）
        stats = result["llm_call_stats"]
        assert stats.get("calls", 0) <= 20

        # 库里确实有实体与关系
        entities = proc.storage.get_all_entities()
        names = {e.get("name") if isinstance(e, dict) else getattr(e, "name", "")
                 for e in (entities or [])}
        assert any("Alice" in str(n) for n in names)

    def test_strong_v1_call_breakdown_has_no_legacy_recall_rounds(self, tmp_path):
        proc = self._processor(tmp_path, {
            "profile": "strong-v1",
            "mode": "strong_one_pass",
            "window_size_chars": 6000,
            "overlap_chars": 300,
        })
        result = proc.remember_text(STRONG_TEXT, doc_name="strong_doc2.md", verbose=False)
        by_step = (result.get("llm_call_stats") or {}).get("by_step") or {}
        # 单遍抽取步骤出现，弱模型多轮召回步骤不存在
        onepass = by_step.get("02s_onepass_extract") or {}
        assert int(onepass.get("calls", 0)) >= 1
        for legacy in ("03_anchor_recall", "03_recall", "05_content_write_rounds"):
            assert legacy not in by_step

    def test_invalid_mode_falls_back_to_dual_model(self, tmp_path):
        proc = self._processor(tmp_path, {"profile": "strong-v1", "mode": "bogus_mode"})
        assert proc.remember_mode == "dual_model"

    def test_strong_v1_uses_window_batch_alignment(self, tmp_path):
        proc = self._processor(tmp_path, {
            "profile": "strong-v1",
            "mode": "strong_one_pass",
            "window_size_chars": 6000,
            "overlap_chars": 300,
        })
        # 开关随 profile 默认开启，并传递给步骤9/10处理器
        assert proc.window_batch_alignment_enabled is True
        assert proc.entity_processor.window_batch_alignment_enabled is True
        assert proc.relation_processor.window_batch_alignment_enabled is True

        # 第二次入库：库中已有关系 → 关系侧窗口批量裁决应被调用（distill 标签可观测）。
        # 实体侧此场景全部命中精确同名快路径（by-design 不进批量），预筛逻辑由下方单测覆盖。
        proc.remember_text(STRONG_TEXT, doc_name="wb1.md", verbose=False)
        result = proc.remember_text(STRONG_TEXT, doc_name="wb2.md", verbose=False)
        by_step = (result.get("llm_call_stats") or {}).get("by_step") or {}
        assert int((by_step.get("10s_window_batch_relations") or {}).get("calls", 0)) >= 1

    def test_window_batch_alignment_can_be_disabled(self, tmp_path):
        proc = self._processor(tmp_path, {
            "profile": "strong-v1",
            "mode": "strong_one_pass",
            "window_batch_alignment": False,
        })
        assert proc.window_batch_alignment_enabled is False


class TestWindowBatchResolve:
    """resolve_entities_window_batch / resolve_relation_pairs_window_batch 单测。"""

    def test_entities_window_batch_schema_and_chunking(self, monkeypatch):
        client = _mock_client()
        calls = {"n": 0}

        def fake_call(*a, **k):
            calls["n"] += 1
            return "```json\n" + _dumps({"results": [
                {"name": "Alice", "match_existing_id": "ent_1", "update_mode": "reuse_existing"},
                {"name": "Unknown", "match_existing_id": "", "update_mode": "create_new"},
                {"name": "Ghost", "update_mode": "create_new"},  # 不在请求里 → 丢弃
            ]}) + "\n```"

        monkeypatch.setattr(client, "_call_llm", fake_call)
        entities = [{"name": "Alice", "content": "a"}, {"name": "Bob", "content": "b"},
                    {"name": "Unknown", "content": "u"}]
        cands = {"Alice": [{"family_id": "ent_1", "name": "Alice"}], "Bob": [], "Unknown": []}
        out = client.resolve_entities_window_batch(entities, cands, max_entities_per_call=2)
        # 3 实体 / 上限2 → 2 次调用
        assert calls["n"] == 2
        assert out["Alice"]["match_existing_id"] == "ent_1"
        assert out["Alice"]["update_mode"] == "reuse_existing"
        assert out["Unknown"]["update_mode"] == "create_new"
        assert "Ghost" not in out
        # verdict 补齐默认键
        assert out["Unknown"]["relations_to_create"] == []
        assert "confidence" in out["Unknown"]

    def test_entities_window_batch_error_returns_empty(self, monkeypatch):
        client = _mock_client()
        monkeypatch.setattr(client, "_call_llm", lambda *a, **k: "not json at all")
        out = client.resolve_entities_window_batch(
            [{"name": "A", "content": "x"}], {"A": []})
        assert out == {}

    def test_relation_pairs_window_batch_schema(self, monkeypatch):
        client = _mock_client()
        seen = {}

        def fake_call(*a, **k):
            msgs = k.get("messages") or []
            seen["prompt"] = msgs[-1]["content"] if msgs else (a[0] if a else "")
            return "```json\n" + _dumps({"results": [
                {"entity1_name": "Alice", "entity2_name": "Bob",
                 "action": "match_existing", "matched_relation_id": "rel_9"},
                {"entity1_name": "X", "entity2_name": "Y", "action": "create_new"},
            ]}) + "\n```"

        monkeypatch.setattr(client, "_call_llm", fake_call)
        pairs = [{
            "entity1_name": "Alice", "entity2_name": "Bob",
            "new_relation_contents": ["Alice 与 Bob 一起工作"],
            "existing_relations": [{"family_id": "rel_9", "content": "同事"}],
        }]
        out = client.resolve_relation_pairs_window_batch(pairs)
        key = client._pair_batch_key("Alice", "Bob")
        assert out[key]["action"] == "match_existing"
        assert out[key]["matched_relation_id"] == "rel_9"
        # 不在请求中的 (X, Y) 被丢弃
        assert len(out) == 1
        assert "待裁决关系对列表" in seen["prompt"]

    def test_entity_needs_batch_llm_predicates(self):
        from core.remember.entity_batch import entity_needs_batch_llm
        assert entity_needs_batch_llm("A", []) is False
        assert entity_needs_batch_llm("A", [{"name": "A", "combined_score": 0.9, "merge_safe": True}]) is False
        assert entity_needs_batch_llm("A", [{"name": "B", "combined_score": 0.1}]) is False
        assert entity_needs_batch_llm("A", [{"name": "B", "combined_score": 0.5}]) is True
        # 精确同名但 merge_safe=False → 需要 LLM
        assert entity_needs_batch_llm("A", [{"name": "A", "combined_score": 0.9, "merge_safe": False}]) is True

    def test_build_window_batch_verdicts_prefilters(self):
        from core.remember.entity_parallel import _build_window_batch_verdicts

        class FakeClient:
            def __init__(self):
                self.calls = []

            def resolve_entities_window_batch(self, entities, cands, context_text=None):
                self.calls.append((entities, cands))
                return {e["name"]: {"update_mode": "create_new"} for e in entities}

        ents = [
            {"name": "Fast", "content": "x"},   # 精确同名高分 → 快路径，不进批量
            {"name": "New", "content": "y"},    # 无候选 → 直建，不进批量
            {"name": "Judge", "content": "z"},  # 中间分 → 需要裁决
        ]
        table = {
            0: [{"name": "Fast", "combined_score": 0.9, "merge_safe": True}],
            1: [],
            2: [{"name": "Other", "combined_score": 0.5}],
        }
        fake = FakeClient()
        out = _build_window_batch_verdicts(fake, ents, table, "ctx")
        assert len(fake.calls) == 1
        sent_names = [e["name"] for e in fake.calls[0][0]]
        assert sent_names == ["Judge"]
        assert out == {"Judge": {"update_mode": "create_new"}}

        # 全部快路径 → 不发起调用
        fake2 = FakeClient()
        out2 = _build_window_batch_verdicts(fake2, ents[:2], {0: table[0], 1: table[1]}, None)
        assert fake2.calls == [] and out2 == {}

        # LLM 侧异常 → 空 dict 回退逐实体
        class BoomClient:
            def resolve_entities_window_batch(self, *a, **k):
                raise RuntimeError("boom")
        out3 = _build_window_batch_verdicts(BoomClient(), ents, table, None)
        assert out3 == {}

    def test_mock_branches_for_window_batch_prompts(self):
        ent_prompt = (
            '<待对齐实体列表>\n<待对齐实体 name="Alice">\n- content: x\n'
            '  候选1: family_id=ent_1 | name=Alice | content=y\n</待对齐实体>\n'
            "</待对齐实体列表>\n请逐实体判断"
        )
        out = mock_llm_response(ent_prompt)
        assert '"results"' in out and "Alice" in out

        rel_prompt = (
            '<待裁决关系对列表>\n<待裁决关系对 entity1="A" entity2="B">\n新关系描述:\n  - r\n'
            "已有关系:\n  - family_id=rel_1 | old\n</待裁决关系对>\n</待裁决关系对列表>\n请逐对判断"
        )
        out = mock_llm_response(rel_prompt)
        assert '"results"' in out and '"entity1_name": "A"' in out
