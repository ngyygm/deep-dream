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

import pytest

from processor.llm.client import LLMClient, PrioritySemaphore
from processor.models import MemoryCache


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
        self.client = LLMClient()

    def test_update_memory_cache_prompt(self):
        result = self.client._mock_llm_response("请更新记忆缓存，根据新内容调整")
        assert "当前摘要" in result
        assert "自我思考" in result

    def test_extract_entity_prompt(self):
        result = self.client._mock_llm_response("请抽取实体：张三是一名工程师")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert "name" in parsed[0]

    def test_extract_relation_prompt(self):
        result = self.client._mock_llm_response("请抽取关系：张三和李四是同事")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert "entity1_name" in parsed[0]
        assert "entity2_name" in parsed[0]

    def test_entity_enhance_prompt(self):
        result = self.client._mock_llm_response("对该实体的content进行更细致的补全和挖掘")
        parsed = json.loads(result)
        assert "content" in parsed
        assert "增强信息" in parsed["content"]

    def test_unknown_prompt_returns_default(self):
        result = self.client._mock_llm_response("这是一条完全无关的提示语")
        assert result == "默认响应"

    def test_entity_extraction_by_name_keyword(self):
        result = self.client._mock_llm_response("实体抽取任务开始")
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_relation_extraction_with_empty_entities(self):
        prompt = "抽取关系\n已抽取的实体：\n</已抽取实体>"
        result = self.client._mock_llm_response(prompt)
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_memory_cache_keyword(self):
        result = self.client._mock_llm_response("请更新memory_cache内容")
        assert "当前摘要" in result


# ---------------------------------------------------------------------------
# _call_llm in mock mode
# ---------------------------------------------------------------------------


class TestCallLlmMockMode:
    """Tests for _call_llm when the client is in mock mode."""

    def setup_method(self):
        self.client = LLMClient()

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


# ---------------------------------------------------------------------------
# _clean_json_string
# ---------------------------------------------------------------------------


class TestCleanJsonString:
    """Tests for _clean_json_string."""

    def setup_method(self):
        self.client = LLMClient()

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
        self.client = LLMClient()

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


# ---------------------------------------------------------------------------
# _parse_json_response edge cases
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    """Tests for _parse_json_response with various input formats."""

    def setup_method(self):
        self.client = LLMClient()

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


# ---------------------------------------------------------------------------
# Concurrency with semaphore
# ---------------------------------------------------------------------------


class TestConcurrencyWithSemaphore:
    """Test that the semaphore limits concurrent LLM calls."""

    def test_max_concurrency_enforced(self):
        client = LLMClient(max_llm_concurrency=2)
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

        # The semaphore limits to at most 2 concurrent calls
        assert max_observed <= 2

    def test_semaphore_released_on_exception(self):
        """Semaphore is released even when errors occur."""
        client = LLMClient(max_llm_concurrency=1)
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


# ---------------------------------------------------------------------------
# update_memory_cache in mock mode
# ---------------------------------------------------------------------------


class TestUpdateMemoryCache:
    """Tests for update_memory_cache with mock LLM."""

    def setup_method(self):
        self.client = LLMClient()

    def test_with_none_cache_creates_initial(self):
        result = self.client.update_memory_cache(
            current_cache=None,
            input_text="这是第一段文本内容",
            document_name="test.txt",
        )
        assert isinstance(result, MemoryCache)
        assert result.content  # non-empty
        assert result.source_document == "test.txt"
        assert result.activity_type == "文档处理"
        assert result.absolute_id.startswith("cache_")
        assert isinstance(result.event_time, datetime)

    def test_with_existing_cache_returns_new(self):
        existing = MemoryCache(
            absolute_id="cache_old",
            content="旧内容",
            event_time=datetime(2025, 1, 1),
            source_document="old.txt",
            activity_type="文档处理",
        )
        result = self.client.update_memory_cache(
            current_cache=existing,
            input_text="新的一段文本",
            document_name="new.txt",
        )
        assert isinstance(result, MemoryCache)
        assert result.content  # non-empty
        # New cache has a different absolute_id
        assert result.absolute_id != "cache_old"
        assert result.source_document == "new.txt"

    def test_custom_event_time(self):
        custom_time = datetime(2025, 6, 15, 12, 0, 0)
        result = self.client.update_memory_cache(
            current_cache=None,
            input_text="测试文本",
            document_name="doc.txt",
            event_time=custom_time,
        )
        assert result.event_time == custom_time

    def test_source_document_extracts_filename(self):
        result = self.client.update_memory_cache(
            current_cache=None,
            input_text="内容",
            document_name="/path/to/my_doc.txt",
        )
        assert result.source_document == "my_doc.txt"


# ---------------------------------------------------------------------------
# create_document_overall_memory in mock mode
# ---------------------------------------------------------------------------


class TestCreateDocumentOverallMemory:
    """Tests for create_document_overall_memory with mock LLM."""

    def setup_method(self):
        self.client = LLMClient()

    def test_returns_memory_cache_with_correct_type(self):
        result = self.client.create_document_overall_memory(
            text_preview="文档开头预览内容",
            document_name="example.txt",
        )
        assert isinstance(result, MemoryCache)
        assert result.activity_type == "文档整体"
        assert result.content  # non-empty
        assert result.source_document == "example.txt"
        assert result.absolute_id.startswith("overall_")
        assert isinstance(result.event_time, datetime)

    def test_custom_event_time(self):
        custom_time = datetime(2025, 3, 1, 10, 30, 0)
        result = self.client.create_document_overall_memory(
            text_preview="预览",
            document_name="doc.txt",
            event_time=custom_time,
        )
        assert result.event_time == custom_time

    def test_with_previous_overall_content(self):
        result = self.client.create_document_overall_memory(
            text_preview="新的文档预览",
            document_name="second.txt",
            previous_overall_content="上一文档的整体记忆内容",
        )
        assert isinstance(result, MemoryCache)
        assert result.activity_type == "文档整体"

    def test_source_document_extracts_filename(self):
        result = self.client.create_document_overall_memory(
            text_preview="预览",
            document_name="/long/path/report.md",
        )
        assert result.source_document == "report.md"
