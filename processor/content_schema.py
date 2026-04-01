"""
Markdown 内容结构化与 Section 级 diff 模块。

将 Entity / Relation 的 content 字段解析为命名 section，
支持 section 级 diff、增量合并，避免全量重写。
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Section Schema 定义
# ---------------------------------------------------------------------------

ENTITY_SECTIONS: List[str] = ["概述", "类型与属性", "详细描述", "关键事实"]
RELATION_SECTIONS: List[str] = ["关系概述", "关系类型", "详细描述", "上下文"]


# ---------------------------------------------------------------------------
# 解析 / 渲染
# ---------------------------------------------------------------------------

def parse_markdown_sections(content: str) -> Dict[str, str]:
    """将 markdown 内容解析为 {section_title: section_body}。

    支持 `## 标题` 和 `# 标题` 两种格式。
    如果内容不含任何 heading，整体作为 "详细描述" section 返回。
    """
    if not content or not content.strip():
        return {}

    sections: Dict[str, str] = {}
    # 匹配 ## 或 # 开头的 heading
    heading_pattern = re.compile(r'^(#{1,2})\s+(.+)$', re.MULTILINE)

    matches = list(heading_pattern.finditer(content))
    if not matches:
        # 无 heading → 整体归入 "详细描述"
        sections["详细描述"] = content.strip()
        return sections

    # heading 前面可能有一段前言文字，归入第一个 section
    first_match = matches[0]
    preamble = content[:first_match.start()].strip()
    if preamble:
        sections["详细描述"] = preamble

    for i, match in enumerate(matches):
        title = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[body_start:body_end].strip()
        sections[title] = body

    return sections


def render_markdown_sections(
    sections: Dict[str, str],
    schema: List[str],
) -> str:
    """按 schema 顺序渲染回 markdown 字符串。"""
    parts: List[str] = []
    for key in schema:
        if key in sections and sections[key]:
            parts.append(f"## {key}\n{sections[key]}")
    # 追加 schema 中未覆盖的 section
    for key, body in sections.items():
        if key not in schema and body:
            parts.append(f"## {key}\n{body}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def compute_section_diff(
    old: Dict[str, str],
    new: Dict[str, str],
) -> Dict[str, Dict[str, object]]:
    """返回 {key: {"old": ..., "new": ..., "changed": bool}}。

    同时检测 added / removed / modified / unchanged。
    """
    all_keys = list(dict.fromkeys(list(old.keys()) + list(new.keys())))
    result: Dict[str, Dict[str, object]] = {}
    for key in all_keys:
        old_val = old.get(key, "")
        new_val = new.get(key, "")
        if key not in old:
            change_type = "added"
        elif key not in new:
            change_type = "removed"
        elif old_val.strip() == new_val.strip():
            change_type = "unchanged"
        else:
            change_type = "modified"
        result[key] = {
            "old": old_val,
            "new": new_val,
            "changed": change_type != "unchanged",
            "change_type": change_type,
        }
    return result


# ---------------------------------------------------------------------------
# 旧格式兼容
# ---------------------------------------------------------------------------

def wrap_plain_as_section(
    plain_text: str,
    section_key: str = "详细描述",
) -> Dict[str, str]:
    """将旧格式纯文本包装为新 schema dict。"""
    if not plain_text or not plain_text.strip():
        return {}
    return {section_key: plain_text.strip()}


def content_to_sections(
    content: str,
    content_format: str,
    schema: List[str],
) -> Dict[str, str]:
    """根据 content_format 将 content 转为 sections dict。

    - "plain" → wrap_plain_as_section
    - "markdown" → parse_markdown_sections
    """
    if content_format == "markdown":
        sections = parse_markdown_sections(content)
        if sections:
            return sections
    # fallback: plain 或解析失败的 markdown
    return wrap_plain_as_section(content)


def section_hash(body: str) -> str:
    """计算 section body 的 SHA-256 hash（前 16 字符）。"""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def has_any_change(diff: Dict[str, Dict[str, object]]) -> bool:
    """判断 diff 中是否有任何 section 发生变更。"""
    return any(v.get("changed", False) for v in diff.values())


def collect_changed_sections(
    diff: Dict[str, Dict[str, object]],
) -> List[Tuple[str, str, str]]:
    """返回 [(key, old_body, new_body), ...] 仅包含变更的 section。"""
    result = []
    for key, info in diff.items():
        if info.get("changed", False):
            result.append((key, info.get("old", ""), info.get("new", "")))
    return result
