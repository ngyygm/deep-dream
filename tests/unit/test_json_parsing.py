"""
Tests for JSON parsing utilities in LLMClient:
_clean_json_string, _fix_json_errors, _parse_json_response,
_parse_entities_list_from_response.
"""

import json

import pytest

from processor.llm.client import LLMClient

# 与 service_config.llm.context_window_tokens / server.config.DEFAULTS 对齐
_TEST_CONTEXT_WINDOW_TOKENS = 8000


def _llm_client(**kwargs):
    kwargs.setdefault("context_window_tokens", _TEST_CONTEXT_WINDOW_TOKENS)
    return LLMClient(**kwargs)


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
