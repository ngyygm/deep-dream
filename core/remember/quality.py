"""Deterministic pre-LLM quality gates for the strong-v1 remember profile."""
from __future__ import annotations

import re
from typing import Iterable


_NOISE = {
    "hi", "hello", "hey", "thanks", "thank you", "good morning", "good night",
    "how are you", "nice", "okay", "ok", "bye",
}
_TURN_ID = re.compile(r"^(?:d\d+\s*:\s*\d+|session[_ -]?\d+|turn[_ -]?\d+)$", re.I)


def filter_entity_names(names: Iterable[str], source_text: str, *, limit: int = 16,
                        require_grounding: bool = True) -> list[str]:
    """Remove IDs, greetings, questions, and sentence-sized pseudo entities.

    require_grounding=True（弱模型防幻觉）时，名称必须逐字出现在 source_text 中；
    strong-v1 传 False——强模型的规范化名称可能非逐字（如词序调整），
    只保留噪声/长度/去重过滤，由 prompt 硬约束负责落地。
    """
    source_lower = source_text.lower()
    candidates: list[tuple[float, int, str]] = []
    seen = set()
    for index, raw in enumerate(names):
        raw_text = str(raw or "")
        name = re.sub(r"\s+", " ", raw_text).strip(" \t\n\r.,;:!?\"'`[]()")
        lowered = name.lower()
        words = re.findall(r"\w+", name, flags=re.UNICODE)
        if not name or lowered in seen or lowered in _NOISE or _TURN_ID.fullmatch(name):
            continue
        if "?" in raw_text or len(name) > 80 or len(words) > 12:
            continue
        if require_grounding and lowered not in source_lower:
            continue
        seen.add(lowered)
        # Favor concrete repeated spans and compact proper-name-like phrases.
        frequency = source_lower.count(lowered)
        compactness = max(0.0, 5.0 - max(0, len(words) - 3))
        capitalization = 2.0 if any(ch.isupper() for ch in name) else 0.0
        candidates.append((frequency * 3.0 + compactness + capitalization, index, name))
    candidates.sort(key=lambda row: (-row[0], row[1]))
    selected = {name for _score, _index, name in candidates[:max(1, limit)]}
    return [name for _score, _index, name in sorted(candidates, key=lambda row: row[1]) if name in selected]


def cap_relation_pairs(pairs: Iterable[tuple[str, str]], *, limit: int = 24) -> list[tuple[str, str]]:
    """Stable endpoint-validated relation cap before relation content calls."""
    result = []
    seen = set()
    for left, right in pairs:
        key = tuple(sorted((str(left), str(right))))
        if not all(key) or key[0] == key[1] or key in seen:
            continue
        seen.add(key)
        result.append((left, right))
        if len(result) >= max(1, limit):
            break
    return result
