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


def _similar_substring_matches(body: str, normalized_body: str, offset_map: List[int], needle: str):
    normalized_needle, _ = _normalized_with_offsets(needle)
    n = len(normalized_needle)
    if n < 4 or not normalized_body:
        return
    min_len = max(3, int(n * 0.7))
    max_len = max(n + 4, int(n * 1.25))
    step = 1 if n <= 10 else 2
    best = None
    for sentence in sentence_spans(body):
        sentence_start = int(sentence["start_offset"])
        sentence_text = str(sentence["text"])
        normalized_sentence, sentence_offsets = _normalized_with_offsets(sentence_text)
        if not normalized_sentence:
            continue
        sentence_max_len = min(len(normalized_sentence), max_len)
        for length in range(min_len, sentence_max_len + 1):
            for idx in range(0, len(normalized_sentence) - length + 1, step):
                candidate = normalized_sentence[idx:idx + length]
                ratio = SequenceMatcher(None, normalized_needle, candidate).ratio()
                if best is None or ratio > best[0]:
                    best = (ratio, sentence_start + sentence_offsets[idx], sentence_start + sentence_offsets[idx + length - 1] + 1)
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
