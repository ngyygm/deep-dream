"""
Tests for mock mode (_mock_llm_response), _call_llm in mock mode,
source document prompt context, and multi-round accepted assistant history.
"""

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

import processor.llm.client as client_module
from processor.llm.client import LLMClient
from processor.llm.errors import LLMContextBudgetExceeded
from processor.models import Episode

# 与 service_config.llm.context_window_tokens / server.config.DEFAULTS 对齐
_TEST_CONTEXT_WINDOW_TOKENS = 8000


def _llm_client(**kwargs):
    kwargs.setdefault("context_window_tokens", _TEST_CONTEXT_WINDOW_TOKENS)
    return LLMClient(**kwargs)


# ---------------------------------------------------------------------------
# Mock mode (_mock_llm_response)
# ---------------------------------------------------------------------------


class TestMockLlmResponse:
    """Tests for _mock_llm_response matching various prompt patterns."""

    def setup_method(self):
        self.client = _llm_client()

    def test_update_memory_cache_prompt(self):
        result = self.client._mock_llm_response("请更新记忆缓存，根据新内容调整")
        assert "当前摘要" in result
        assert "自我思考" in result

    def test_extract_entity_prompt(self):
        result = self.client._mock_llm_response("请抽取实体：张三是一名工程师")
        parsed = self.client._parse_json_response(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert "name" in parsed[0]

    def test_extract_relation_prompt(self):
        result = self.client._mock_llm_response("请抽取关系：张三和李四是同事")
        parsed = self.client._parse_json_response(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert "entity1_name" in parsed[0]
        assert "entity2_name" in parsed[0]

    def test_entity_enhance_prompt(self):
        result = self.client._mock_llm_response("对该实体的content进行更细致的补全和挖掘")
        parsed = self.client._parse_json_response(result)
        assert "content" in parsed
        assert "增强信息" in parsed["content"]

    def test_unknown_prompt_returns_default(self):
        result = self.client._mock_llm_response("这是一条完全无关的提示语")
        assert result == "默认响应"

    def test_entity_extraction_by_name_keyword(self):
        result = self.client._mock_llm_response("实体抽取任务开始")
        parsed = self.client._parse_json_response(result)
        assert isinstance(parsed, list)

    def test_relation_extraction_with_empty_entities(self):
        prompt = "抽取关系\n已抽取的实体：\n</已抽取实体>"
        result = self.client._mock_llm_response(prompt)
        parsed = self.client._parse_json_response(result)
        assert isinstance(parsed, list)

    def test_memory_cache_keyword(self):
        result = self.client._mock_llm_response("请更新memory_cache内容")
        assert "当前摘要" in result


    def test_extract_entities_by_names_prompt_returns_requested_names(self):
        prompt = """<指定实体名称>
- 张三
- 项目A
</指定实体名称>"""
        result = self.client._mock_llm_response(prompt)
        parsed = self.client._parse_json_response(result)
        assert [item["name"] for item in parsed] == ["张三", "项目A"]

    def test_relation_candidate_prompt_uses_stable_names(self):
        prompt = """<稳定概念实体列表>
- 张三 | 人物
- 项目A | 项目
- 设计原则 | 抽象概念
</稳定概念实体列表>

请先发现值得建立关系的概念对。"""
        result = self.client._mock_llm_response(prompt)
        parsed = self.client._parse_json_response(result)
        assert parsed[0]["entity1_name"] in {"张三", "项目A"}
        assert parsed[0]["entity2_name"] in {"张三", "项目A", "设计原则"}

    def test_anchor_recall_prompt_returns_structural_anchors(self):
        result = self.client._mock_llm_response("请召回所有结构性文本锚点概念候选。")
        parsed = self.client._parse_json_response(result)
        assert isinstance(parsed, list)
        assert parsed[0]["name"] == "第一章"


# ---------------------------------------------------------------------------
# _call_llm in mock mode
# ---------------------------------------------------------------------------


class TestCallLlmMockMode:
    """Tests for _call_llm when the client is in mock mode."""

    def setup_method(self):
        self.client = _llm_client()

    def test_returns_nonempty_string(self):
        result = self.client._call_llm("请抽取实体：一些内容")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_with_messages_parameter(self):
        messages = [
            {"role": "system", "content": "你是一个助手"},
            {"role": "user", "content": "请更新记忆缓存"},
        ]
        result = self.client._call_llm("ignored", messages=messages)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_utf8_validation_valid(self):
        assert self.client._is_valid_utf8("hello world") is True
        assert self.client._is_valid_utf8("你好世界") is True
        assert self.client._is_valid_utf8("") is True

    def test_utf8_validation_replacement_char(self):
        assert self.client._is_valid_utf8("bad \ufffd char") is False

    def test_parse_json_response_valid(self):
        result = self.client._parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_response_code_block(self):
        raw = '```json\n{"key": "value"}\n```'
        result = self.client._parse_json_response(raw)
        assert result == {"key": "value"}

    def test_parse_json_response_salvages_truncated_array_tail(self):
        raw = """```json
[
  {"name": "水浒传", "content": "中国古典小说"},
  {"name": "洪太尉", "content": "朝廷命官"},
  {"name": "龙虎山", "content": "道教名山
```"""
        result = self.client._parse_json_response(raw)
        assert result == [
            {"name": "水浒传", "content": "中国古典小说"},
            {"name": "洪太尉", "content": "朝廷命官"},
        ]

    def test_parse_json_response_repairs_truncated_object(self):
        """Truncated JSON object: last key-value pair is incomplete."""
        raw = '{"action": "match_existing", "matched_relation_id": "rel_abc123", "merged_content": "贾宝玉与薛宝钗的关系非常密切'
        result = self.client._parse_json_response(raw)
        assert result == {
            "action": "match_existing",
            "matched_relation_id": "rel_abc123",
        }

    def test_parse_json_response_repairs_truncated_object_with_nested(self):
        """Truncated JSON object with nested array value."""
        raw = '{"status": "ok", "items": [{"id": 1}, {"id": 2}, {"id": 3'
        result = self.client._parse_json_response(raw)
        assert result == {"status": "ok"}

    def test_parse_json_response_repairs_truncated_object_only_key(self):
        """Truncated JSON object where only the first key-value is complete."""
        raw = '{"action": "create_new", "content": "some long text that got trun'
        result = self.client._parse_json_response(raw)
        assert result == {"action": "create_new"}

    def test_parse_json_response_no_repair_for_valid_object(self):
        """Valid objects should parse normally, not trigger repair."""
        raw = '{"action": "match", "id": "rel_123"}'
        result = self.client._parse_json_response(raw)
        assert result == {"action": "match", "id": "rel_123"}

    def test_try_repair_truncated_json_object_returns_none_for_empty(self):
        assert self.client._try_repair_truncated_json_object("") is None
        assert self.client._try_repair_truncated_json_object("{}") is None
        assert self.client._try_repair_truncated_json_object("[]") is None

    def test_try_repair_truncated_json_object_unclosed_string(self):
        """Object with unclosed string value."""
        raw = '{"name": "test", "value": "unclosed string'
        result = self.client._try_repair_truncated_json_object(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == {"name": "test"}

    def test_resolve_request_max_tokens_keeps_desired_when_prompt_within_budget(self):
        messages = [{"role": "user", "content": "你" * 7900}]
        resolved = self.client._resolve_request_max_tokens(messages, desired_max_tokens=6000)
        assert resolved == 6000

    def test_resolve_request_max_tokens_raises_when_prompt_exceeds_budget(self):
        messages = [{"role": "user", "content": "你" * 8100}]
        with pytest.raises(LLMContextBudgetExceeded, match="输入上下文超限"):
            self.client._resolve_request_max_tokens(messages, desired_max_tokens=100)

    def test_call_llm_does_not_retry_when_finish_reason_is_length(self, monkeypatch):
        client = _llm_client(
            api_key="test-key",
            model_name="test-model",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            max_tokens=6000,
        )
        calls = []

        def fake_chat(messages, model, base_url, api_key, timeout=300, max_tokens=None):
            calls.append({
                "messages": messages,
                "model": model,
                "base_url": base_url,
                "api_key": api_key,
                "max_tokens": max_tokens,
            })
            return SimpleNamespace(
                content='{"ok": true}',
                done_reason="length",
                raw={"choices": [{"finish_reason": "length"}]},
            )

        monkeypatch.setattr(client_module, "openai_compatible_chat", fake_chat)

        result = client._call_llm(
            "ignored",
            messages=[{"role": "user", "content": "请只输出一个很短的 JSON"}],
            allow_mock_fallback=False,
        )

        assert result == '{"ok": true}'
        assert len(calls) == 1


class TestSourceDocumentPromptContext:
    def setup_method(self):
        self.client = _llm_client()

    def test_judge_content_need_update_includes_source_document(self):
        captured = {}

        def fake_call_llm(prompt, system_prompt=None, *args, **kwargs):
            captured["prompt"] = prompt
            captured["system_prompt"] = system_prompt
            return "false"

        self.client._call_llm = fake_call_llm

        result = self.client.judge_content_need_update(
            "旧实体内容",
            "新实体内容",
            old_source_document="doc_old.txt",
            new_source_document="doc_new.txt",
            old_name="宝玉",
            new_name="贾宝玉",
            object_type="实体",
        )

        assert result is False
        assert "source_document: doc_old.txt" in captured["prompt"]
        assert "source_document: doc_new.txt" in captured["prompt"]
        assert "<对象类型>\n实体\n</对象类型>" in captured["prompt"]

    def test_resolve_relation_pair_batch_includes_source_documents(self):
        captured = {}

        def fake_call_llm(prompt, system_prompt=None, *args, **kwargs):
            captured["prompt"] = prompt
            return json.dumps({
                "action": "create_new",
                "matched_family_id": "",
                "need_update": True,
                "merged_content": "合并后的关系",
                "confidence": 0.9,
            }, ensure_ascii=False)

        self.client._call_llm = fake_call_llm

        result = self.client.resolve_relation_pair_batch(
            entity1_name="贾宝玉",
            entity2_name="林黛玉",
            new_relation_contents=["二人发生互动"],
            existing_relations=[{
                "family_id": "rel_1",
                "content": "二人曾经见面",
                "source_document": "old_rel.txt",
            }],
            new_source_document="new_rel.txt",
        )

        assert result["action"] == "create_new"
        assert "source_document=new_rel.txt" in captured["prompt"]
        assert "source_document=old_rel.txt" in captured["prompt"]


class TestMultiRoundAcceptedAssistantHistory:
    def test_extract_entities_and_relations_uses_accepted_assistant_history_next_round(self):
        client = _llm_client()
        seen_messages = []
        responses = [
            (
                (
                    [
                        {"name": "A", "content": "alpha"},
                        {"name": "A", "content": "alpha duplicate"},
                        {"name": "B", "content": "beta"},
                    ],
                    [
                        {"entity1_name": "A", "entity2_name": "B", "content": "r1"},
                        {"entity1_name": "A", "entity2_name": "B", "content": "r2 should be dropped by pair-only dedupe"},
                    ],
                ),
                '```json\n{"entities":[{"name":"A","content":"alpha"},{"name":"A","content":"alpha duplicate"},{"name":"B","content":"beta"}],"relations":[{"entity1_name":"A","entity2_name":"B","content":"r1"},{"entity1_name":"A","entity2_name":"B","content":"r2 should be dropped by pair-only dedupe"}]}\n```',
            ),
            (
                (
                    [
                        {"name": "B", "content": "beta again"},
                        {"name": "C", "content": "gamma"},
                    ],
                    [
                        {"entity1_name": "B", "entity2_name": "C", "content": "r3"},
                    ],
                ),
                '```json\n{"entities":[{"name":"B","content":"beta again"},{"name":"C","content":"gamma"}],"relations":[{"entity1_name":"B","entity2_name":"C","content":"r3"}]}\n```',
            ),
        ]

        def fake_call_llm_until_json_parses(messages, parse_fn=None, json_parse_retries=None):
            seen_messages.append([dict(m) for m in messages])
            return responses[len(seen_messages) - 1]

        client.call_llm_until_json_parses = fake_call_llm_until_json_parses

        cache = Episode(
            absolute_id="cache_test",
            content="memory",
            event_time=datetime(2025, 1, 1),
            source_document="doc.txt",
            activity_type="文档处理",
        )
        entities, relations = client.extract_entities_and_relations(cache, "body", rounds=2, verbose=False)

        assert [e["name"] for e in entities] == ["A", "B", "C"]
        assert relations == [
            {"entity1_name": "A", "entity2_name": "B", "content": "r1"},
            {"entity1_name": "B", "entity2_name": "C", "content": "r3"},
        ]
        assert len(seen_messages) == 2
        assistant_payload = client._parse_json_response(seen_messages[1][2]["content"])
        assert assistant_payload == {
            "entities": [
                {"name": "A", "content": "alpha duplicate"},
                {"name": "B", "content": "beta"},
            ],
            "relations": [
                {"entity1_name": "A", "entity2_name": "B", "content": "r1"},
            ],
        }

    def test_extract_entities_uses_deduped_assistant_history_next_round(self):
        client = _llm_client()
        seen_messages = []
        responses = [
            (
                [
                    {"name": "A", "content": "alpha"},
                    {"name": "A", "content": "alpha duplicate with longer content"},
                    {"name": "B", "content": "beta"},
                ],
                '```json\n[{"name":"A","content":"alpha"},{"name":"A","content":"alpha duplicate with longer content"},{"name":"B","content":"beta"}]\n```',
            ),
            (
                [
                    {"name": "B", "content": "beta again"},
                    {"name": "C", "content": "gamma"},
                ],
                '```json\n[{"name":"B","content":"beta again"},{"name":"C","content":"gamma"}]\n```',
            ),
        ]

        def fake_call_llm_until_json_parses(messages, parse_fn=None, json_parse_retries=None):
            seen_messages.append([dict(m) for m in messages])
            return responses[len(seen_messages) - 1]

        client.call_llm_until_json_parses = fake_call_llm_until_json_parses

        cache = Episode(
            absolute_id="cache_test",
            content="memory",
            event_time=datetime(2025, 1, 1),
            source_document="doc.txt",
            activity_type="文档处理",
        )
        out = client.extract_entities(cache, "body", rounds=2, verbose=False, compress_multi_round=False)

        assert out == [
            {"name": "A", "content": "alpha duplicate with longer content"},
            {"name": "B", "content": "beta again"},
            {"name": "C", "content": "gamma"},
        ]
        assert len(seen_messages) == 2
        assistant_payload = client._parse_json_response(seen_messages[1][2]["content"])
        assert assistant_payload == [
            {"name": "A", "content": "alpha duplicate with longer content"},
            {"name": "B", "content": "beta"},
        ]

    def test_extract_relations_uses_accepted_assistant_history_next_round(self):
        client = _llm_client()
        seen_messages = []
        responses = [
            (
                [
                    {"entity1_name": "B", "entity2_name": "A", "content": "r1"},
                    {"entity1_name": "A", "entity2_name": "B", "content": "r1"},
                ],
                '```json\n[{"entity1_name":"B","entity2_name":"A","content":"r1"},{"entity1_name":"A","entity2_name":"B","content":"r1"}]\n```',
            ),
            (
                [
                    {"entity1_name": "A", "entity2_name": "B", "content": "r1"},
                    {"entity1_name": "A", "entity2_name": "B", "content": "r2"},
                ],
                '```json\n[{"entity1_name":"A","entity2_name":"B","content":"r1"},{"entity1_name":"A","entity2_name":"B","content":"r2"}]\n```',
            ),
        ]

        def fake_call_llm_until_json_parses(messages, parse_fn=None, json_parse_retries=None):
            seen_messages.append([dict(m) for m in messages])
            return responses[len(seen_messages) - 1]

        client.call_llm_until_json_parses = fake_call_llm_until_json_parses

        cache = Episode(
            absolute_id="cache_test",
            content="memory",
            event_time=datetime(2025, 1, 1),
            source_document="doc.txt",
            activity_type="文档处理",
        )
        out = client.extract_relations(
            cache,
            "body",
            entities=[
                {"name": "A", "content": "alpha"},
                {"name": "B", "content": "beta"},
            ],
            rounds=2,
            verbose=False,
            compress_multi_round=False,
        )

        assert out == [
            {"entity1_name": "A", "entity2_name": "B", "content": "r1"},
            {"entity1_name": "A", "entity2_name": "B", "content": "r2"},
        ]
        assert len(seen_messages) == 2
        assistant_payload = client._parse_json_response(seen_messages[1][2]["content"])
        assert assistant_payload == [
            {"entity1_name": "A", "entity2_name": "B", "content": "r1"},
        ]

    def test_extract_entities_partial_on_context_budget_second_round(self):
        client = _llm_client()
        calls: list[int] = []

        def fake_call_llm_until_json_parses(messages, parse_fn=None, json_parse_retries=None):
            calls.append(len(messages))
            if len(calls) == 1:
                raw = '```json\n[{"name":"A","content":"x"}]\n```'
                return parse_fn(raw), raw
            raise LLMContextBudgetExceeded(
                "LLM 上下文预算超限：估算输入约 99999 tokens，已达到或超过模型总上限 8000。"
            )

        client.call_llm_until_json_parses = fake_call_llm_until_json_parses
        cache = Episode(
            absolute_id="cache_test",
            content="memory",
            event_time=datetime(2025, 1, 1),
            source_document="doc.txt",
            activity_type="文档处理",
        )
        out = client.extract_entities(cache, "body", rounds=2, verbose=False)
        assert out == [{"name": "A", "content": "x"}]
        assert len(calls) == 2

    def test_extract_relations_partial_on_context_budget_second_round(self):
        client = _llm_client()
        calls: list[int] = []

        def fake_call_llm_until_json_parses(messages, parse_fn=None, json_parse_retries=None):
            calls.append(len(messages))
            if len(calls) == 1:
                raw = '```json\n[{"entity1_name":"A","entity2_name":"B","content":"r1"}]\n```'
                return parse_fn(raw), raw
            raise LLMContextBudgetExceeded(
                "LLM 上下文预算超限：估算输入约 99999 tokens，已达到或超过模型总上限 8000。"
            )

        client.call_llm_until_json_parses = fake_call_llm_until_json_parses
        cache = Episode(
            absolute_id="cache_test",
            content="memory",
            event_time=datetime(2025, 1, 1),
            source_document="doc.txt",
            activity_type="文档处理",
        )
        out = client.extract_relations(
            cache,
            "body",
            entities=[{"name": "A", "content": "a"}, {"name": "B", "content": "b"}],
            rounds=2,
            verbose=False,
        )
        assert out == [{"entity1_name": "A", "entity2_name": "B", "content": "r1"}]
        assert len(calls) == 2

    def test_extract_entities_stops_before_second_round_when_precheck_fails(self):
        client = _llm_client()
        calls: list[int] = []

        def fake_resolve_request_max_tokens(messages, desired_max_tokens):
            # 当消息以"继续"开头时触发预算超限（匹配多轮续抽 prompt）
            last_content = messages[-1].get("content", "") if messages else ""
            if last_content.startswith("继续"):
                raise LLMContextBudgetExceeded(
                    "LLM 上下文预算超限：估算输入约 99999 tokens，已达到或超过模型总上限 8000。"
                )
            return desired_max_tokens

        def fake_call_llm_until_json_parses(messages, parse_fn=None, json_parse_retries=None):
            calls.append(len(messages))
            raw = '```json\n[{"name":"A","content":"x"}]\n```'
            return parse_fn(raw), raw

        client._resolve_request_max_tokens = fake_resolve_request_max_tokens
        client.call_llm_until_json_parses = fake_call_llm_until_json_parses
        cache = Episode(
            absolute_id="cache_test",
            content="memory",
            event_time=datetime(2025, 1, 1),
            source_document="doc.txt",
            activity_type="文档处理",
        )

        out = client.extract_entities(cache, "body", rounds=2, verbose=False)

        assert out == [{"name": "A", "content": "x"}]
        assert len(calls) == 1

    def test_extract_relations_stops_before_second_round_when_precheck_fails(self):
        client = _llm_client()
        calls: list[int] = []

        def fake_resolve_request_max_tokens(messages, desired_max_tokens):
            last_content = messages[-1].get("content", "") if messages else ""
            if last_content.startswith("继续从文本中抽取概念关系"):
                raise LLMContextBudgetExceeded(
                    "LLM 上下文预算超限：估算输入约 99999 tokens，已达到或超过模型总上限 8000。"
                )
            return desired_max_tokens

        def fake_call_llm_until_json_parses(messages, parse_fn=None, json_parse_retries=None):
            calls.append(len(messages))
            raw = '```json\n[{"entity1_name":"A","entity2_name":"B","content":"r1"}]\n```'
            return parse_fn(raw), raw

        client._resolve_request_max_tokens = fake_resolve_request_max_tokens
        client.call_llm_until_json_parses = fake_call_llm_until_json_parses
        cache = Episode(
            absolute_id="cache_test",
            content="memory",
            event_time=datetime(2025, 1, 1),
            source_document="doc.txt",
            activity_type="文档处理",
        )

        out = client.extract_relations(
            cache,
            "body",
            entities=[{"name": "A", "content": "a"}, {"name": "B", "content": "b"}],
            rounds=2,
            verbose=False,
        )

        assert out == [{"entity1_name": "A", "entity2_name": "B", "content": "r1"}]
        assert len(calls) == 1

    def test_extract_relations_falls_back_to_single_pass_for_uncovered_entities(self):
        client = _llm_client()
        seen_messages = []

        def fake_resolve_request_max_tokens(messages, desired_max_tokens):
            last_content = messages[-1].get("content", "") if messages else ""
            if last_content.startswith("继续从文本中抽取概念关系"):
                raise LLMContextBudgetExceeded(
                    "LLM 上下文预算超限：估算输入约 99999 tokens，已达到或超过模型总上限 8000。"
                )
            return desired_max_tokens

        def fake_call_llm_until_json_parses(messages, parse_fn=None, json_parse_retries=None):
            seen_messages.append([dict(m) for m in messages])
            user_prompt = messages[-1]["content"]
            if "<未覆盖实体>" in user_prompt:
                raw = '```json\n[{"entity1_name":"A","entity2_name":"C","content":"r2"}]\n```'
            else:
                raw = '```json\n[{"entity1_name":"A","entity2_name":"B","content":"r1"}]\n```'
            return parse_fn(raw), raw

        client._resolve_request_max_tokens = fake_resolve_request_max_tokens
        client.call_llm_until_json_parses = fake_call_llm_until_json_parses
        cache = Episode(
            absolute_id="cache_test",
            content="memory",
            event_time=datetime(2025, 1, 1),
            source_document="doc.txt",
            activity_type="文档处理",
        )

        out = client.extract_relations(
            cache,
            "body",
            entities=[
                {"name": "A", "content": "a"},
                {"name": "B", "content": "b"},
                {"name": "C", "content": "c"},
            ],
            rounds=2,
            verbose=False,
        )

        assert out == [
            {"entity1_name": "A", "entity2_name": "B", "content": "r1"},
            {"entity1_name": "A", "entity2_name": "C", "content": "r2"},
        ]
        assert len(seen_messages) == 2
        assert seen_messages[1][-1]["role"] == "user"
        assert "<未覆盖实体>" in seen_messages[1][-1]["content"]
        assert "- C" in seen_messages[1][-1]["content"]
        assert "继续生成" not in seen_messages[1][-1]["content"]

    def test_prepare_memory_cache_for_prompt_truncates_long_cache(self):
        client = _llm_client(prompt_episode_max_chars=80)
        cache = Episode(
            absolute_id="cache_test",
            content=("A" * 60) + ("B" * 60) + ("C" * 60),
            event_time=datetime(2025, 1, 1),
            source_document="doc.txt",
            activity_type="文档处理",
        )

        prepared = client._prepare_episode_for_prompt(cache)

        assert len(prepared) <= 80
        assert "A" * 20 in prepared
        assert prepared.endswith("C" * 18)
        assert "记忆缓存过长，已截断" in prepared


# NOTE: TestUpdateEpisode and TestCreateDocumentOverallMemory removed —
# update_memory_cache and create_document_overall_memory were removed from LLMClient.
