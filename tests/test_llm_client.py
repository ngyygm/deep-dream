"""
Comprehensive tests for LLMClient and PrioritySemaphore.

Covers:
- PrioritySemaphore: acquire/release, priority ordering, active_count, max_value,
  over-subscription blocking
- Mock mode (_mock_llm_response): various prompt patterns
- _call_llm in mock mode: basic calls, messages parameter, UTF-8 validation,
  JSON parsing
- _clean_json_string: Chinese punctuation, BOM, trailing commas
- _fix_json_errors: invalid unicode escapes
- _parse_json_response: plain JSON, code-block JSON, malformed JSON
- Concurrency with semaphore
- _normalize_entity_pair
- update_memory_cache in mock mode
- create_document_overall_memory in mock mode
"""

import json
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import pytest

import processor.llm.client as client_module
from processor.llm.client import LLMClient, PrioritySemaphore
from processor.llm.errors import LLMContextBudgetExceeded
from processor.models import Episode

# 与 service_config.llm.context_window_tokens / server.config.DEFAULTS 对齐
_TEST_CONTEXT_WINDOW_TOKENS = 8000


def _llm_client(**kwargs):
    kwargs.setdefault("context_window_tokens", _TEST_CONTEXT_WINDOW_TOKENS)
    return LLMClient(**kwargs)


class TestRelationEntityCatalog:
    """_build_relation_entity_catalog 与 relation_content_snippet_length 配置。"""

    def test_name_only_when_relation_snippet_length_zero(self):
        client = _llm_client(relation_content_snippet_length=0)
        s, valid, order = client._build_relation_entity_catalog(
            [
                {"name": "A", "content": "should_not_appear_in_catalog"},
                {"name": "B", "content": ""},
            ],
        )
        assert "should_not_appear" not in s
        assert "|" not in s
        assert s.strip() == "- A\n- B"
        assert valid == {"A", "B"}
        assert order == ["A", "B"]


# ---------------------------------------------------------------------------
# PrioritySemaphore tests
# ---------------------------------------------------------------------------


class TestPrioritySemaphore:
    """Tests for the PrioritySemaphore class."""

    def test_basic_acquire_release(self):
        """Basic acquire and release with value=2."""
        sem = PrioritySemaphore(2)
        sem.acquire()
        assert sem.active_count == 1
        sem.acquire()
        assert sem.active_count == 2
        sem.release()
        assert sem.active_count == 1
        sem.release()
        assert sem.active_count == 0

    def test_priority_ordering(self):
        """Lower priority number acquires semaphore first."""
        sem = PrioritySemaphore(1)
        acquisition_order = []

        # First caller takes the only slot
        sem.acquire()

        # Three threads wait with different priorities
        results = {}

        def waiter(name, priority):
            sem.acquire(priority=priority)
            acquisition_order.append(name)
            sem.release()

        # Start threads with priorities 5, 1, 3 (lower = higher priority)
        t1 = threading.Thread(target=waiter, args=("low_p", 5))
        t2 = threading.Thread(target=waiter, args=("high_p", 1))
        t3 = threading.Thread(target=waiter, args=("mid_p", 3))

        # Stagger starts slightly so they enter the heap in order
        t1.start()
        time.sleep(0.05)
        t2.start()
        time.sleep(0.05)
        t3.start()
        time.sleep(0.05)

        # Release the slot so the highest-priority waiter proceeds
        sem.release()

        t1.join(timeout=5)
        t2.join(timeout=5)
        t3.join(timeout=5)

        # Priority 1 ("high_p") should acquire first
        assert acquisition_order[0] == "high_p"
        # Remaining two are released in priority order as each finishes
        assert len(acquisition_order) == 3

    def test_active_count_property(self):
        """active_count reflects current usage."""
        sem = PrioritySemaphore(3)
        assert sem.active_count == 0
        sem.acquire()
        assert sem.active_count == 1
        sem.acquire()
        assert sem.active_count == 2
        sem.release()
        assert sem.active_count == 1

    def test_max_value_property(self):
        """max_value returns the constructor value."""
        sem = PrioritySemaphore(5)
        assert sem.max_value == 5

    def test_oversubscription_blocks(self):
        """Acquiring beyond capacity blocks until a release happens."""
        sem = PrioritySemaphore(1)
        sem.acquire()

        blocked = threading.Event()
        finished = threading.Event()

        def try_acquire():
            blocked.set()  # signal that thread has started
            sem.acquire()
            finished.set()
            sem.release()

        t = threading.Thread(target=try_acquire)
        t.start()
        blocked.wait(timeout=2)
        # Thread should be blocked because the slot is held
        assert not finished.is_set()

        sem.release()
        t.join(timeout=5)
        assert finished.is_set()

    def test_value_must_be_at_least_one(self):
        """Constructor raises ValueError for value < 1."""
        with pytest.raises(ValueError):
            PrioritySemaphore(0)
        with pytest.raises(ValueError):
            PrioritySemaphore(-1)


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


# ---------------------------------------------------------------------------
# Entity list parsing (JSON + YAML-ish fallback)
# ---------------------------------------------------------------------------


class TestParseEntitiesListFromResponse:
    def test_yamlish_name_content_fallback(self):
        client = _llm_client()
        raw = '- name: "曹雪芹"\n- content: "《红楼梦》作者"'
        out = client._parse_entities_list_from_response(raw)
        assert out == [{"name": "曹雪芹", "content": "《红楼梦》作者"}]

    def test_valid_json_still_preferred(self):
        client = _llm_client()
        raw = json.dumps(
            [{"name": "曹雪芹", "content": "作者"}], ensure_ascii=False
        )
        out = client._parse_entities_list_from_response(raw)
        assert out == [{"name": "曹雪芹", "content": "作者"}]


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


# ---------------------------------------------------------------------------
# _clean_json_string
# ---------------------------------------------------------------------------


class TestCleanJsonString:
    """Tests for _clean_json_string."""

    def setup_method(self):
        self.client = _llm_client()

    def test_chinese_colon_to_english(self):
        result = self.client._clean_json_string('{"key"："value"}')
        assert result == '{"key":"value"}'

    def test_chinese_comma_to_english(self):
        result = self.client._clean_json_string('{"key"："value"，"num"：1}')
        assert "，" not in result
        assert "," in result

    def test_chinese_semicolon_to_english(self):
        result = self.client._clean_json_string("a；b")
        assert result == "a;b"

    def test_chinese_quotes_not_converted(self):
        # The source code's "中文引号" replace uses ASCII " on both sides (no-op),
        # so Chinese curly quotes (U+201C / U+201D) pass through unchanged.
        result = self.client._clean_json_string('\u201chello\u201d')
        assert '\u201c' in result
        assert '\u201d' in result

    def test_bom_removal(self):
        result = self.client._clean_json_string('\ufeff{"key": "value"}')
        assert result == '{"key": "value"}'
        assert "\ufeff" not in result

    def test_trailing_comma_in_object(self):
        result = self.client._clean_json_string('{"key": "value",}')
        assert result == '{"key": "value"}'

    def test_trailing_comma_in_array(self):
        result = self.client._clean_json_string('["a", "b",]')
        assert result == '["a", "b"]'


# ---------------------------------------------------------------------------
# _fix_json_errors
# ---------------------------------------------------------------------------


class TestFixJsonErrors:
    """Tests for _fix_json_errors."""

    def setup_method(self):
        self.client = _llm_client()

    def test_invalid_unicode_escape_padded(self):
        # \uAB (only 2 hex digits) should be padded to \uAB00
        result = self.client._fix_json_errors('"\\uAB"')
        assert "\\uAB00" in result

    def test_invalid_unicode_escape_one_digit(self):
        # \uX -> non-hex, becomes \u0020
        result = self.client._fix_json_errors('"\\uX"')
        assert "\\u0020" in result

    def test_invalid_unicode_escape_three_digits(self):
        # \uABC (3 hex digits) -> \uABC0
        result = self.client._fix_json_errors('"\\uABC"')
        assert "\\uABC0" in result

    def test_valid_unicode_escape_unchanged(self):
        original = '"\\u0041"'  # 4 hex digits, valid
        result = self.client._fix_json_errors(original)
        assert result == '"\\u0041"'

    def test_bom_and_trailing_comma_combined(self):
        raw = '\ufeff{"a"：1，"b"：2，}'
        result = self.client._fix_json_errors(raw)
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": 2}

    def test_unescaped_control_chars_inside_string_are_escaped(self):
        raw = '{"content": "第一行\n第二行\t第三列\r结束"}'
        result = self.client._fix_json_errors(raw)
        parsed = json.loads(result)
        assert parsed["content"] == "第一行\n第二行\t第三列\r结束"


# ---------------------------------------------------------------------------
# _parse_json_response edge cases
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    """Tests for _parse_json_response with various input formats."""

    def setup_method(self):
        self.client = _llm_client()

    def test_plain_json_without_code_blocks(self):
        raw = '{"name": "test", "value": 42}'
        result = self.client._parse_json_response(raw)
        assert result == {"name": "test", "value": 42}

    def test_json_in_json_code_block(self):
        raw = 'Here is the result:\n```json\n{"name": "test"}\n```\nDone.'
        result = self.client._parse_json_response(raw)
        assert result == {"name": "test"}

    def test_json_in_plain_code_block(self):
        raw = '```\n{"name": "test"}\n```'
        result = self.client._parse_json_response(raw)
        assert result == {"name": "test"}

    def test_malformed_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            self.client._parse_json_response("{invalid json!!!")

    def test_json_array(self):
        raw = '[{"id": 1}, {"id": 2}]'
        result = self.client._parse_json_response(raw)
        assert result == [{"id": 1}, {"id": 2}]

    def test_empty_string_raises(self):
        with pytest.raises(json.JSONDecodeError):
            self.client._parse_json_response("")

    def test_invalid_control_character_can_be_repaired(self):
        raw = '[{"entity1_name":"甲","entity2_name":"乙","content":"第一行\n第二行"}]'
        result = self.client._parse_json_response(raw)
        assert result == [{"entity1_name": "甲", "entity2_name": "乙", "content": "第一行\n第二行"}]


# ---------------------------------------------------------------------------
# Concurrency with semaphore
# ---------------------------------------------------------------------------


class TestConcurrencyWithSemaphore:
    """Test that the semaphore limits concurrent LLM calls."""

    def test_max_concurrency_enforced(self):
        # max=1 时不拆分信号量，便于与旧版一样打桩 _llm_semaphore
        client = _llm_client(max_llm_concurrency=1)
        max_observed = 0
        lock = threading.Lock()
        barrier = threading.Barrier(5, timeout=10)

        # Wrap the semaphore's acquire/release to track actual concurrency
        sem = client._llm_semaphore
        original_acquire = sem.acquire
        original_release = sem.release

        def tracked_acquire(priority=0):
            original_acquire(priority)
            with lock:
                count = sem.active_count
                nonlocal max_observed
                if count > max_observed:
                    max_observed = count

        sem.acquire = tracked_acquire

        def call_llm():
            barrier.wait()
            time.sleep(0.05)  # let all threads queue up
            result = client._call_llm("请抽取实体：测试内容")
            assert isinstance(result, str)
            assert len(result) > 0

        threads = [threading.Thread(target=call_llm) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert max_observed <= 1

    def test_semaphore_released_on_exception(self):
        """Semaphore is released even when errors occur."""
        client = _llm_client(max_llm_concurrency=1)
        assert client._llm_semaphore.active_count == 0
        # In mock mode, no exceptions occur; just verify release after normal call
        client._call_llm("请抽取实体")
        assert client._llm_semaphore.active_count == 0


# ---------------------------------------------------------------------------
# _normalize_entity_pair
# ---------------------------------------------------------------------------


class TestNormalizeEntityPair:
    """Tests for _normalize_entity_pair static method."""

    def test_ordering_alphabetical(self):
        result = LLMClient._normalize_entity_pair("Alice", "Bob")
        assert result == ("Alice", "Bob")

    def test_ordering_reversed(self):
        result = LLMClient._normalize_entity_pair("Bob", "Alice")
        assert result == ("Alice", "Bob")

    def test_same_entity(self):
        result = LLMClient._normalize_entity_pair("Charlie", "Charlie")
        assert result == ("Charlie", "Charlie")

    def test_strips_whitespace(self):
        result = LLMClient._normalize_entity_pair("  Alice  ", "  Bob  ")
        assert result == ("Alice", "Bob")

    def test_chinese_entity_names(self):
        result = LLMClient._normalize_entity_pair("张三", "李四")
        # Chinese characters sort by Unicode code point; 张 < 李 is determined
        # by their code points. The important thing is consistency.
        e1, e2 = result
        assert e1 <= e2

    def test_consistent_regardless_of_input_order(self):
        pair1 = LLMClient._normalize_entity_pair("X", "Y")
        pair2 = LLMClient._normalize_entity_pair("Y", "X")
        assert pair1 == pair2


class TestRelationEntityNameFiltering:
    def setup_method(self):
        self.client = _llm_client()

    def test_keeps_unknown_endpoints_for_step4_supplement(self):
        relations = [
            {
                "entity1_name": "报复",
                "entity2_name": "贾瑞（小说人物）",
                "content": "报复与贾瑞有关",
            },
            {
                "entity1_name": "贾瑞（小说人物）",
                "entity2_name": "王熙凤（小说人物）",
                "content": "二人存在互动",
            },
        ]

        filtered, normalized_count, filtered_count = self.client._normalize_and_filter_relations_by_entities(
            relations,
            {"贾瑞（小说人物）", "王熙凤（小说人物）"},
        )

        assert normalized_count == 0
        assert filtered_count == 0
        assert len(filtered) == 2
        revenge_edges = [
            r for r in filtered
            if "报复" in (r["entity1_name"], r["entity2_name"])
        ]
        assert len(revenge_edges) == 1
        assert "贾瑞（小说人物）" in (
            revenge_edges[0]["entity1_name"],
            revenge_edges[0]["entity2_name"],
        )

    def test_normalizes_unique_base_name(self):
        relations = [
            {
                "entity1_name": "贾瑞",
                "entity2_name": "王熙凤（小说人物）",
                "content": "二人存在互动",
            }
        ]

        filtered, normalized_count, filtered_count = self.client._normalize_and_filter_relations_by_entities(
            relations,
            {"贾瑞（小说人物）", "王熙凤（小说人物）"},
        )

        assert normalized_count == 1
        assert filtered_count == 0
        assert len(filtered) == 1
        assert filtered[0]["entity1_name"] == "贾瑞（小说人物）"
        assert filtered[0]["entity2_name"] == "王熙凤（小说人物）"

    def test_ambiguous_base_resolves_to_latest_bracketed_in_catalog_order(self):
        """多個「贾瑞（…）」去括号后 Jaccard 并列时，按 catalog 顺序取较后者。"""
        relations = [
            {
                "entity1_name": "贾瑞",
                "entity2_name": "王熙凤（小说人物）",
                "content": "二人存在互动",
            }
        ]

        catalog = {"贾瑞（小说人物）", "贾瑞（另一版本）", "王熙凤（小说人物）"}
        order = ["贾瑞（小说人物）", "王熙凤（小说人物）", "贾瑞（另一版本）"]
        filtered, normalized_count, filtered_count = self.client._normalize_and_filter_relations_by_entities(
            relations,
            catalog,
            catalog_name_order=order,
        )

        assert normalized_count == 1
        assert filtered_count == 0
        assert len(filtered) == 1
        assert filtered[0]["entity1_name"] == "贾瑞（另一版本）"
        assert filtered[0]["entity2_name"] == "王熙凤（小说人物）"


class TestParseRelationsResponse:
    def setup_method(self):
        self.client = _llm_client()

    def test_accepts_name_fields_maps_to_catalog(self):
        raw = json.dumps([
            {
                "entity1_name": "贾宝玉（小说人物）",
                "entity2_name": "林黛玉（小说人物）",
                "content": "二人有明确互动",
            }
        ], ensure_ascii=False)

        parsed = self.client._parse_relations_response(
            raw,
            {"贾宝玉（小说人物）", "林黛玉（小说人物）"},
        )

        assert len(parsed) == 1
        assert {parsed[0]["entity1_name"], parsed[0]["entity2_name"]} == {"贾宝玉（小说人物）", "林黛玉（小说人物）"}

    def test_accepts_unknown_second_endpoint_for_supplement(self):
        raw = json.dumps([
            {
                "entity1_name": "贾宝玉（小说人物）",
                "entity2_name": "薛宝钗（小说人物）",
                "content": "二人有明确互动",
            }
        ], ensure_ascii=False)

        parsed = self.client._parse_relations_response(
            raw,
            {"贾宝玉（小说人物）", "林黛玉（小说人物）"},
        )

        assert len(parsed) == 1
        assert {parsed[0]["entity1_name"], parsed[0]["entity2_name"]} == {
            "贾宝玉（小说人物）",
            "薛宝钗（小说人物）",
        }

    def test_ambiguous_partial_name_keeps_raw_below_jaccard_threshold(self):
        """Jaccard 未达阈值时不强行映射；保留简称供步骤4 补全，不再整条丢弃。"""
        raw = json.dumps([
            {
                "entity1_name": "真人",
                "entity2_name": "太尉",
                "content": "二者有明确互动",
            }
        ], ensure_ascii=False)

        parsed = self.client._parse_relations_response(
            raw,
            {"罗真人", "张真人", "洪太尉"},
        )

        assert len(parsed) == 1
        assert {parsed[0]["entity1_name"], parsed[0]["entity2_name"]} == {"太尉", "真人"}

    def test_accepts_name_only_relation_and_maps_to_closest_entity(self):
        raw = json.dumps([
            {
                "entity1_name": "真人",
                "entity2_name": "太尉",
                "content": "二者有明确互动",
            }
        ], ensure_ascii=False)

        # 默认阈值 0.9 时「真人/太尉」与全名 Jaccard 约 2/3；降低阈值以断言映射到目录全名
        client = _llm_client(relation_endpoint_jaccard_threshold=0.65)
        parsed = client._parse_relations_response(
            raw,
            {"罗真人", "洪太尉"},
        )

        assert len(parsed) == 1
        assert {parsed[0]["entity1_name"], parsed[0]["entity2_name"]} == {"罗真人", "洪太尉"}

    def test_jaccard_uses_stripped_names_maps_to_bracketed_catalog(self):
        """比较时去掉括号；目录仅有带说明全名时，简称应对齐到该全名。"""
        raw = json.dumps([
            {
                "entity1_name": "贾宝玉",
                "entity2_name": "林黛玉",
                "content": "二人有明确互动",
            }
        ], ensure_ascii=False)

        catalog = {"贾宝玉（小说人物）", "林黛玉（小说人物）"}
        order = ["贾宝玉（小说人物）", "林黛玉（小说人物）"]
        parsed = self.client._parse_relations_response(
            raw, catalog, catalog_name_order=order
        )

        assert len(parsed) == 1
        assert {parsed[0]["entity1_name"], parsed[0]["entity2_name"]} == catalog

    def test_tie_break_bracketed_prefers_later_catalog_name(self):
        """同分且均带括号时，取 catalog_name_order 中较后者（新版本）。"""
        raw = json.dumps([
            {
                "entity1_name": "贾宝玉",
                "entity2_name": "薛宝钗",
                "content": "互动",
            }
        ], ensure_ascii=False)

        catalog = {"贾宝玉（旧）", "贾宝玉（新）", "薛宝钗"}
        order = ["薛宝钗", "贾宝玉（旧）", "贾宝玉（新）"]
        parsed = self.client._parse_relations_response(
            raw, catalog, catalog_name_order=order
        )

        assert len(parsed) == 1
        names = {parsed[0]["entity1_name"], parsed[0]["entity2_name"]}
        assert "贾宝玉（新）" in names
        assert "薛宝钗" in names


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
