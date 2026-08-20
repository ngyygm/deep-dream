"""Markdown-aware text chunking helpers."""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional


def split_markdown_chunks(text: str, *, window_size: int, overlap: int) -> List[Dict[str, object]]:
    """Split Markdown by headings and soft text boundaries.

    The returned offsets are Python string offsets into the original text.
    Oversized heading sections are split near paragraph/sentence boundaries
    before falling back to a hard window cut.
    """
    body = text or ""
    if not body:
        return [{"content": "", "heading_path": "", "start_offset": 0, "end_offset": 0}]

    window_size = max(1, int(window_size or 1))
    overlap = max(0, min(int(overlap or 0), window_size - 1))
    spans = _heading_spans(body)
    chunks: List[Dict[str, object]] = []

    for span in spans:
        span_start = int(span["start"])
        span_end = int(span["end"])
        if span_start >= span_end:
            continue
        if span_end - span_start <= window_size:
            chunks.append(_make_chunk(body, span, span_start, span_end))
            continue

        part = 0
        chunk_start = span_start
        min_chunk = max(1, int(window_size * 0.55))
        while chunk_start < span_end:
            hard_end = min(chunk_start + window_size, span_end)
            if hard_end >= span_end:
                chunk_end = span_end
            else:
                min_end = min(hard_end, chunk_start + min_chunk)
                chunk_end = _best_end_boundary(body, min_end, hard_end)
                if chunk_end <= chunk_start:
                    chunk_end = hard_end

            chunk = _make_chunk(body, span, chunk_start, chunk_end)
            if part:
                heading = str(chunk["heading_path"])
                chunk["heading_path"] = f"{heading} [{part}]".strip()
            chunks.append(chunk)

            if chunk_end >= span_end:
                break
            next_start = max(span_start, chunk_end - overlap)
            next_start = _best_start_boundary(body, next_start, chunk_end)
            if next_start <= chunk_start:
                next_start = chunk_end
            chunk_start = next_start
            part += 1

    return chunks or [{"content": body, "heading_path": "", "start_offset": 0, "end_offset": len(body)}]


def apply_document_metadata_prefix(doc_name: str, chunk: str, window_index: int) -> str:
    """窗口 0 注入「[文档元数据]」前缀（orchestrator 与 task_queue 共用，字节一致）。

    remember 流水线在窗口 0 的 chunk 前拼接文档名元数据标记，task_queue 的
    修复检测 / 完整性检查计算窗口哈希时必须拼出完全相同的字节串，否则
    chunk_hash 对不上、窗口会被误判为缺失。两处统一从这里取，避免漂移
    （不变式 a：前缀内容不得改动）。
    """
    if window_index == 0 and doc_name and not doc_name.startswith(("auto_", "api:")):
        return f"[文档元数据] 文档名：{doc_name} [/文档元数据]\n\n{chunk}"
    return chunk


def sentence_spans(text: str, *, base_offset: int = 0) -> List[Dict[str, object]]:
    """Return sentence-like spans with offsets relative to the source document."""
    body = text or ""
    spans: List[Dict[str, object]] = []
    start = 0
    pattern = re.compile(r"([。！？!?；;]+|[.!?](?=\s|$)|\n\s*\n+)")
    for match in pattern.finditer(body):
        end = match.end()
        _append_sentence_span(spans, body, start, end, base_offset)
        start = end
    _append_sentence_span(spans, body, start, len(body), base_offset)
    return spans or [{"text": body, "start_offset": base_offset, "end_offset": base_offset + len(body)}]


def find_text_evidence(text: str, candidates: Iterable[str], *, base_offset: int = 0, limit: int = 3) -> List[Dict[str, object]]:
    """Find deterministic mention evidence for candidate names in sentence spans."""
    body = text or ""
    if not body:
        return []
    sentences = sentence_spans(body, base_offset=base_offset)
    normalized_body, offset_map = _normalized_with_offsets(body)
    found: List[Dict[str, object]] = []
    seen = set()

    for candidate in candidates:
        needle = str(candidate or "").strip()
        if not needle:
            continue
        for start, end, match_text, match_type, confidence in _candidate_matches(body, normalized_body, offset_map, needle):
            key = (start, end, match_type)
            if key in seen:
                continue
            seen.add(key)
            sentence = _sentence_for_local_span(sentences, base_offset + start, base_offset + end)
            found.append({
                "start_offset": base_offset + start,
                "end_offset": base_offset + end,
                "sentence_start": sentence["start_offset"],
                "sentence_end": sentence["end_offset"],
                "quote": body[start:end],
                "sentence": sentence["text"],
                "match_text": match_text,
                "match_type": match_type,
                "confidence": confidence,
            })
            if len(found) >= limit:
                return found
        for start, end, match_text, match_type, confidence in _similar_substring_matches(body, normalized_body, offset_map, needle):
            key = (start, end, match_type)
            if key in seen:
                continue
            seen.add(key)
            sentence = _sentence_for_local_span(sentences, base_offset + start, base_offset + end)
            found.append({
                "start_offset": base_offset + start,
                "end_offset": base_offset + end,
                "sentence_start": sentence["start_offset"],
                "sentence_end": sentence["end_offset"],
                "quote": body[start:end],
                "sentence": sentence["text"],
                "match_text": match_text,
                "match_type": match_type,
                "confidence": confidence,
            })
            if len(found) >= limit:
                return found
    return found


def _make_chunk(body: str, span: Dict[str, object], start: int, end: int) -> Dict[str, object]:
    return {
        "content": body[start:end],
        "heading_path": str(span.get("heading_path") or ""),
        "start_offset": start,
        "end_offset": end,
    }


def _append_sentence_span(spans: List[Dict[str, object]], body: str, start: int, end: int, base_offset: int) -> None:
    while start < end and body[start].isspace():
        start += 1
    while end > start and body[end - 1].isspace():
        end -= 1
    if start < end:
        spans.append({
            "text": body[start:end],
            "start_offset": base_offset + start,
            "end_offset": base_offset + end,
        })


def _normalized_with_offsets(text: str) -> tuple[str, List[int]]:
    chars: List[str] = []
    offsets: List[int] = []
    for idx, char in enumerate(text or ""):
        normalized = unicodedata.normalize("NFKC", char).casefold()
        if not normalized.strip() or re.match(r"[\W_]+", normalized, flags=re.UNICODE):
            continue
        for out_char in normalized:
            chars.append(out_char)
            offsets.append(idx)
    return "".join(chars), offsets


def _candidate_matches(body: str, normalized_body: str, offset_map: List[int], needle: str):
    flags = 0 if _has_case_sensitive_chars(needle) else re.IGNORECASE
    try:
        for match in re.finditer(re.escape(needle), body, flags=flags):
            yield match.start(), match.end(), needle, "exact", 1.0
    except re.error:
        pass

    normalized_needle, _ = _normalized_with_offsets(needle)
    if not normalized_needle:
        return
    start = 0
    while True:
        idx = normalized_body.find(normalized_needle, start)
        if idx < 0:
            break
        local_start = offset_map[idx]
        local_end = offset_map[idx + len(normalized_needle) - 1] + 1
        yield local_start, local_end, needle, "normalized", 0.92
        start = idx + max(1, len(normalized_needle))


# 模糊子串搜索的复杂度约束（P3.10）：
# 旧实现对每个句子枚举 0.7n..1.25n 的全部长度 × 全部起始位置，逐窗口跑
# SequenceMatcher——长实体名时是 O(n²) 起步（实测 n=74 的单候选单窗口 ~480ms）。
# 约束方式（结果语义近似，取舍见函数 docstring）：
#   1. 长度按 |窗口长度 - n| 升序枚举——相似度理论上限 2·min(n,ℓ)/(n+ℓ) 随偏差
#      增大单调下降，先扫理论上限最高的窗口，截断只损失低潜力尾部；
#   2. 总枚举窗口数受“字符工作量预算”约束（≈ windows × n² 有上限）：小名不受
#      影响（预算高于全枚举量），长名收敛到 O(预算 / n²)；
#   3. 最佳相似度达到早停阈值即停止（后续窗口理论上限 1.0，超越概率极低）。
_SIMILAR_EARLY_STOP_RATIO = 0.95
_SIMILAR_WORK_BUDGET = 4_000_000
_SIMILAR_MIN_WINDOWS = 64


def _similar_substring_matches(body: str, normalized_body: str, offset_map: List[int], needle: str):
    """句子内模糊子串匹配（至多产出一个最佳窗口）。

    在旧版“全枚举取全局最优”的基础上加了排序 + 预算 + 早停：
    - 未触发预算/早停时，最佳相似度与旧版一致（仅同分并列时所选窗口可能不同，
      本版优先长度更接近 n 的窗口）；
    - 早停在首个 ≥0.95 的窗口处全局停止：其后若存在略更优（如 0.975）的窗口
      会被错过——两者都已过 0.78 阈值，置信度损失上界 ~0.05；
    - 触发预算截断后以低潜力尾部被丢弃为代价换取确定性上界。
    """
    normalized_needle, _ = _normalized_with_offsets(needle)
    n = len(normalized_needle)
    if n < 4 or not normalized_body:
        return
    min_len = max(3, int(n * 0.7))
    max_len = max(n + 4, int(n * 1.25))
    step = 1 if n <= 10 else 2
    max_windows = max(_SIMILAR_MIN_WINDOWS, _SIMILAR_WORK_BUDGET // (n * n))
    best = None
    windows = 0
    # seq2 固定为 needle：difflib 会缓存 seq2 的字符索引，逐窗口只换 seq1。
    # autojunk 必须关掉：默认值只作用于 seq2，固定 seq2=needle 后会把“高频字符
    # 超 1%”的判定挪到 needle 头上——n≥200 的名字会被整体 junk 成 0 匹配，
    # 出现旧版能命中、新版返回空的静默回退（旧版 junk 作用在候选窗一侧）。
    matcher = SequenceMatcher(None, "", normalized_needle, autojunk=False)
    for sentence in sentence_spans(body):
        sentence_start = int(sentence["start_offset"])
        sentence_text = str(sentence["text"])
        normalized_sentence, sentence_offsets = _normalized_with_offsets(sentence_text)
        if not normalized_sentence:
            continue
        sentence_max_len = min(len(normalized_sentence), max_len)
        if sentence_max_len < min_len:
            continue
        exhausted = False
        for length in sorted(range(min_len, sentence_max_len + 1), key=lambda v: abs(v - n)):
            for idx in range(0, len(normalized_sentence) - length + 1, step):
                matcher.set_seq1(normalized_sentence[idx:idx + length])
                ratio = matcher.ratio()
                windows += 1
                if best is None or ratio > best[0]:
                    best = (ratio, sentence_start + sentence_offsets[idx], sentence_start + sentence_offsets[idx + length - 1] + 1)
                # 早停 / 预算耗尽都在最内层判断：单个长句子也要受预算约束
                if best[0] >= _SIMILAR_EARLY_STOP_RATIO or windows >= max_windows:
                    exhausted = True
                    break
            if exhausted:
                break
        if exhausted:
            break
    if not best or best[0] < 0.78:
        return
    ratio, local_start, local_end = best
    yield local_start, local_end, needle, "similar_substring", round(float(ratio), 3)


def _has_case_sensitive_chars(text: str) -> bool:
    return any(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in text or "")


def _sentence_for_local_span(sentences: List[Dict[str, object]], abs_start: int, abs_end: int) -> Dict[str, object]:
    best: Optional[Dict[str, object]] = None
    best_overlap = -1
    for sentence in sentences:
        start = int(sentence["start_offset"])
        end = int(sentence["end_offset"])
        overlap = max(0, min(end, abs_end) - max(start, abs_start))
        if overlap > best_overlap:
            best = sentence
            best_overlap = overlap
    return best or {"text": "", "start_offset": abs_start, "end_offset": abs_end}


def _heading_spans(body: str) -> List[Dict[str, object]]:
    matches = list(re.finditer(r"(?m)^(#{1,6})\s+(.+)$", body))
    if not matches:
        return [{"start": 0, "end": len(body), "heading_path": ""}]

    spans: List[Dict[str, object]] = []
    heading_stack: List[str] = []
    if matches[0].start() > 0:
        spans.append({"start": 0, "end": matches[0].start(), "heading_path": ""})

    for idx, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        heading_stack = heading_stack[: level - 1]
        heading_stack.append(heading)
        spans.append({
            "start": match.start(),
            "end": matches[idx + 1].start() if idx + 1 < len(matches) else len(body),
            "heading_path": " / ".join(heading_stack),
        })
    return spans


def _best_end_boundary(body: str, min_end: int, hard_end: int) -> int:
    window = body[min_end:hard_end]
    base = min_end
    for pattern in (
        r"\n\s*\n+",
        r"[。！？!?；;](?:[）)”’\"'\]]*)",
        r"\n",
        r"[，,、：:]",
        r"\s+",
    ):
        best = None
        for match in re.finditer(pattern, window):
            best = base + match.end()
        if best is not None:
            return best
    return hard_end


def _best_start_boundary(body: str, desired_start: int, previous_end: int) -> int:
    """Move overlap start to a nearby readable boundary when possible."""
    if desired_start <= 0:
        return 0
    search_end = min(previous_end, desired_start + 120)
    window = body[desired_start:search_end]
    for pattern in (r"\n\s*\n+", r"[。！？!?；;](?:[）)”’\"'\]]*)", r"\n", r"\s+"):
        match = re.search(pattern, window)
        if match:
            return desired_start + match.end()
    return desired_start
