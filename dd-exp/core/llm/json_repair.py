"""
JSON repair utilities extracted from client.py.

Pure functions for cleaning, fixing, and parsing JSON responses from LLM output.
None of these functions depend on LLMClient state.
"""
import json
import re
from typing import Any, List, Optional

from ..utils import wprint_info

# Pre-compiled regex for JSON cleanup
_TRAILING_COMMA_RE = re.compile(r',(\s*[}\]])')
_MD_BULLET_IN_JSON_RE = re.compile(r'^\s*\*\s+(?=[\[{])', re.MULTILINE)
_BARE_IDENTIFIER_RE = re.compile(r',?\s*\b(?:gap|ellipsis|continue|\.\.\.)\b\s*,?', re.IGNORECASE)
_INVALID_UNICODE_ESCAPE_RE = re.compile(r'\\u([0-9a-fA-F]{0,3})(?![0-9a-fA-F])')
_CJK_PUNCT_RE = re.compile(r'[：，；]')  # ：，；
_CJK_PUNCT_MAP = {'：': ':', '，': ',', '；': ';'}
_CURRENT_ENTITY_NAME_RE = re.compile(r"<当前实体>.*?name:\s*(\S+)", re.DOTALL)
_FAMILY_ID_RE = re.compile(r"family_id:\s*(\S+)")
_ENTRY_NAME_RE = re.compile(r"name:\s*(\S+)")
# Single-pass JSON fence extraction — replaces 4 separate find() calls
_JSON_FENCE_RE = re.compile(r'```(?:json)?\s*\n?', re.DOTALL)

# JSON parse failure correction prompts (used with call_llm_until_json_parses)
_JSON_RETRY_USER_MESSAGE = (
    "【输出格式纠错】上一条输出无法被解析为合法 JSON。"
    "请严格只输出一个 markdown `json` 代码块，不要任何解释文字；"
    "若是数组，代码块内部必须是合法 JSON 数组；若是对象，代码块内部必须是合法 JSON 对象。"
)
# Suspected truncation (unclosed string etc.) suffix: guide shorter fields to avoid another truncation
_JSON_RETRY_TRUNCATION_SUFFIX = (
    " 若疑似因输出过长在字符串中间被截断：请缩小每条 content 的篇幅（建议单字段不超过约 200 字），"
    "字符串内的换行必须写成转义 \\n；仍只输出一个合法的 ```json ... ``` 代码块。"
)
# Truncation detection keywords — computed once at import time
_TRUNCATION_KEYWORDS = (
    "Unterminated string",
    "Expecting value",
    "Expecting ',' delimiter",
    "Unterminated",
)


def _fix_unicode_escapes(text: str) -> str:
    """修复无效的 Unicode 转义序列（\\u 后不足 4 位十六进制）。"""
    def _replace_invalid_escape(match):
        hex_part = match.group(1)
        if not hex_part:
            return '\\u0020'
        elif len(hex_part) < 4:
            return '\\u' + hex_part.ljust(4, '0')
        else:
            return match.group(0)
    return _INVALID_UNICODE_ESCAPE_RE.sub(_replace_invalid_escape, text)


def _escape_control_chars_in_json_strings(text: str) -> str:
    """仅在 JSON 字符串内部转义裸控制字符，避免破坏结构字符。"""
    result = []
    in_string = False
    escaped = False

    for ch in text:
        if in_string:
            if escaped:
                result.append(ch)
                escaped = False
                continue
            if ch == '\\':
                result.append(ch)
                escaped = True
                continue
            if ch == '"':
                result.append(ch)
                in_string = False
                continue
            if ch == '\n':
                result.append('\\n')
                continue
            if ch == '\r':
                result.append('\\r')
                continue
            if ch == '\t':
                result.append('\\t')
                continue
            if ord(ch) < 0x20:
                result.append(f'\\u{ord(ch):04x}')
                continue
            result.append(ch)
        else:
            result.append(ch)
            if ch == '"':
                in_string = True
                escaped = False

    return ''.join(result)


def clean_json_string(json_str: str) -> str:
    """
    清理JSON字符串，修复常见错误

    Args:
        json_str: 原始JSON字符串

    Returns:
        清理后的JSON字符串
    """
    # 移除BOM标记
    json_str = json_str.lstrip('﻿')
    # 移除首尾空白
    json_str = json_str.strip()
    # 修复中文标点符号（fast path: skip when no CJK punctuation present）
    if _CJK_PUNCT_RE.search(json_str):
        json_str = _CJK_PUNCT_RE.sub(lambda m: _CJK_PUNCT_MAP[m.group()], json_str)
    # 注意：中文弯引号 “ ” 经常出现在 JSON 字符串值内部（如 "研制"九章"…"）
    # 不能全局替换为 ASCII "，否则会破坏 JSON 结构。
    # 它们是合法 UTF-8 字符，可直接保留。
    # 移除可能的尾随逗号（在数组或对象的最后一个元素后）
    json_str = _TRAILING_COMMA_RE.sub(r'\1', json_str)
    # 移除 markdown 列表标记（弱模型在 JSON 内混入 "*   {" 等）
    json_str = _MD_BULLET_IN_JSON_RE.sub('', json_str)
    # 移除模型在 JSON 对象间插入的占位符（gap, ellipsis, ...）
    json_str = _BARE_IDENTIFIER_RE.sub(',', json_str)
    # 修复连续逗号（前一步可能产生 ,,）
    json_str = re.sub(r',{2,}', ',', json_str)
    return json_str


def fix_json_errors(json_str: str) -> str:
    """
    尝试修复JSON错误

    Args:
        json_str: 有错误的JSON字符串 (assumed already clean_json_string'd by caller)

    Returns:
        修复后的JSON字符串
    """
    # Note: caller (parse_json_response) already applied clean_json_string,
    # so we skip it here to avoid redundant work.

    # Fix invalid Unicode escape sequences
    json_str = _fix_unicode_escapes(json_str)

    # Escape bare control characters inside JSON string values
    json_str = _escape_control_chars_in_json_strings(json_str)

    return json_str


def parse_json_response(response: str) -> Any:
    """从 LLM 响应中提取并解析 JSON。"""
    json_str = response or ""
    # Single-pass fence extraction using regex (replaces 4 find() calls)
    fence_match = _JSON_FENCE_RE.search(json_str)
    if fence_match:
        json_start = fence_match.end()
        json_end = json_str.find("```", json_start)
        if json_end == -1:
            wprint_info("[DeepDream] 警告: LLM 响应的 ```json 块未闭合，JSON 可能被截断")
            json_str = json_str[json_start:].strip()
        else:
            json_str = json_str[json_start:json_end].strip()

    json_str = clean_json_string(json_str)

    # 截断检测：检查 JSON 结构是否完整（cache stripped versions for reuse in except）
    _ls = json_str.lstrip()
    _first_char = _ls[0] if _ls else ''
    if _first_char in ('[', '{'):
        close_char = ']' if _first_char == '[' else '}'
        _rs = json_str.rstrip()
        _last_char = _rs[-1] if _rs else ''
        if _last_char != close_char:
            wprint_info(f"[DeepDream] 警告: LLM 响应 JSON 被截断，以 {_first_char} 开头但不以 {close_char} 结尾。"
                  f"请缩短输入上下文或输出内容。响应前200字符: {json_str[:200]}")

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        fixed = fix_json_errors(json_str)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            # Reuse _ls/_rs from above (avoid recomputing strip)
            _rs_cached = _rs if _first_char in ('[', '{') else json_str.rstrip()
            _is_complete = (
                (_first_char == '[' and _rs_cached.endswith(']')) or
                (_first_char == '{' and _rs_cached.endswith('}'))
            )
            if not _is_complete:
                # Try array truncation repair
                repaired = try_repair_truncated_json_array(json_str)
                if repaired is not None:
                    try:
                        parsed = json.loads(repaired)
                        wprint_info(
                            "[DeepDream] 警告: 检测到数组型 JSON 尾部截断；"
                            "已裁剪不完整尾部并补全 `]`，沿用可恢复部分。"
                        )
                        return parsed
                    except json.JSONDecodeError:
                        pass
                # Try object truncation repair
                repaired_obj = try_repair_truncated_json_object(json_str)
                if repaired_obj is not None:
                    try:
                        parsed = json.loads(repaired_obj)
                        wprint_info(
                            "[DeepDream] 警告: 检测到对象型 JSON 尾部截断；"
                            "已裁剪不完整键值对并补全 `}`，沿用可恢复部分。"
                        )
                        return parsed
                    except json.JSONDecodeError:
                        pass
            wprint_info("[DeepDream] 警告: LLM 响应 JSON 解析失败（可能被截断）。"
                  f"响应: {json_str}")
            raise


def try_repair_truncated_json_array(json_str: str) -> Optional[str]:
    """修复尾部被截断的 JSON 数组：裁掉不完整尾巴并补上 `]`。"""
    stripped = (json_str or "").strip()
    if not stripped.startswith("[") or stripped.endswith("]"):
        return None

    in_string = False
    escaped = False
    stack: List[str] = []
    last_complete_value_end: Optional[int] = None

    for idx, ch in enumerate(stripped):
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch in "[{":
            stack.append(ch)
            continue

        if ch in "]}":
            if not stack:
                break
            opener = stack[-1]
            if (opener == "[" and ch != "]") or (opener == "{" and ch != "}"):
                break
            stack.pop()
            if stack == ["["]:
                last_complete_value_end = idx + 1
            elif not stack and ch == "]":
                last_complete_value_end = idx + 1
                break

    if last_complete_value_end is None:
        return None

    candidate = stripped[:last_complete_value_end].rstrip()
    if not candidate.startswith("["):
        return None
    candidate = candidate.rstrip(", \n\r\t") + "]"
    return candidate if candidate != stripped else None


def try_repair_truncated_json_object(json_str: str) -> Optional[str]:
    """修复尾部被截断的 JSON 对象：裁掉不完整键值对并补上 `}`。

    例如 {"action": "match", "id": "rel_xxx", "content": "很长的内容被截断...
    修复为 {"action": "match", "id": "rel_xxx"}
    """
    stripped = (json_str or "").strip()
    if not stripped.startswith("{") or stripped.endswith("}"):
        return None

    in_string = False
    escaped = False
    stack: List[str] = []
    last_complete_value_end: Optional[int] = None

    for idx, ch in enumerate(stripped):
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch in "[{":
            stack.append(ch)
            continue

        if ch in "]}":
            if not stack:
                break
            opener = stack[-1]
            if (opener == "[" and ch != "]") or (opener == "{" and ch != "}"):
                break
            stack.pop()
            if not stack:
                last_complete_value_end = idx + 1
                break
            if stack == ["{"]:
                last_complete_value_end = idx + 1
            continue

        # Comma at top-level object: the key-value pair before it is complete
        if ch == "," and stack == ["{"]:
            last_complete_value_end = idx

    if last_complete_value_end is None:
        return None

    candidate = stripped[:last_complete_value_end].rstrip()
    if not candidate.startswith("{"):
        return None
    candidate = candidate.rstrip(", \n\r\t") + "}"
    return candidate if candidate != stripped else None
