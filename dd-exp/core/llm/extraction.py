"""
Extraction Mixin for LLMClient — surviving single-call helpers.

Window extraction itself (entities + relations in one call) lives in
extraction_strong.py; this mixin keeps the content-writing, parsing, and
alignment-judgment methods shared by the alignment pipeline.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from core.utils import (
    capture_log_context as _capture_log_context,
    set_window_label as _set_window_label,
    set_pipeline_role as _set_pipeline_role,
)
from .errors import LLMContextBudgetExceeded
from .prompts import (
    RELATION_CONTENT_WRITE_SYSTEM,
    RELATION_CONTENT_WRITE_USER,
    RELATION_BATCH_CONTENT_WRITE_SYSTEM,
    RELATION_BATCH_CONTENT_WRITE_USER,
    ENTITY_ALIGNMENT_JUDGE_SYSTEM,
    ENTITY_ALIGNMENT_JUDGE_USER,
)


def _restore_log_context(ctx: tuple):
    """在子线程中恢复父线程的日志上下文。"""
    label, role = ctx
    if label is not None:
        _set_window_label(label)
    if role is not None:
        _set_pipeline_role(role)


def _clear_log_context():
    """清除子线程的日志上下文（恢复默认 ---- ----）。"""
    _set_window_label(None)
    _set_pipeline_role(None)

_VALID_VERDICTS = frozenset(("same", "different", "uncertain"))

class _LLMExtractionMixin:
    """Content writing, response parsing, and alignment judgment for LLMClient."""

    def _source_language_system(self, system_prompt: str, source_text: str) -> str:
        """Keep concept text aligned with the source/embedding language."""
        if not getattr(self, "preserve_source_language", False):
            return system_prompt
        return (
            system_prompt
            + "\n\nLanguage rule: preserve the dominant language of the source text in every "
              "entity name and description. If the source is English, output English; do not translate it."
        )

    # ------------------------------------------------------------------
    # Shared pair parser
    # ------------------------------------------------------------------

    def _parse_pair_list(self, response: str) -> List[Tuple[str, str]]:
        """Parse LLM response into a list of (entity1, entity2) tuples.

        Supports array and object formats:
        - [["conceptA", "conceptB"]]
        - [{"subject": "conceptA", "object": "conceptB"}]
        - Also accepts: {"entity1": "A", "entity2": "B"}

        Deduplication is by (entity1, entity2) pair only.
        """
        data = self._parse_json_response(response)
        pairs = []
        seen: set = set()
        if isinstance(data, list):
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    a, b = str(item[0]).strip(), str(item[1]).strip()
                elif isinstance(item, dict):
                    a = str(item.get("subject") or item.get("entity1") or item.get("entity1_name") or "").strip()
                    b = str(item.get("object") or item.get("entity2") or item.get("entity2_name") or "").strip()
                else:
                    continue
                if a and b and a != b:
                    pair_key = (a, b) if a <= b else (b, a)
                    if pair_key not in seen:
                        seen.add(pair_key)
                        pairs.append(pair_key)
        return pairs

    # ------------------------------------------------------------------
    # Shared content parser
    # ------------------------------------------------------------------

    def _parse_content_field(self, response: str) -> str:
        """Parse a 'content' field from JSON response — used by relation writing."""
        data = self._parse_json_response(response)
        if isinstance(data, dict) and "content" in data:
            return str(data["content"]).strip()
        if isinstance(data, str):
            return data.strip()
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and "content" in first:
                return str(first["content"]).strip()
        return ""

    # ------------------------------------------------------------------
    # Batch Relation Content Writing
    # ------------------------------------------------------------------

    def batch_write_relation_content(
        self, pairs: List[Tuple[str, str]], window_text: str,
        chunk_size: int = 35, max_workers: int = 1,
    ) -> Dict[Tuple[str, str], str]:
        """Write relation descriptions in chunked batch LLM calls.

        Splits pairs into chunks to avoid output truncation.

        Args:
            max_workers: Max parallel threads for chunk processing. Default 1 (sequential).

        Returns:
            Dict mapping (entity1, entity2) -> content string.
        """
        if not pairs:
            return {}
        parent_priority = getattr(self._priority_local, "priority", None)
        _parent_log_ctx = _capture_log_context()

        def _single_with_priority(chunk_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], str]:
            _restore_log_context(_parent_log_ctx)
            previous = getattr(self._priority_local, "priority", None)
            if parent_priority is not None:
                self._priority_local.priority = parent_priority
            try:
                return self._batch_write_relation_content_single(chunk_pairs, window_text)
            finally:
                _clear_log_context()
                if previous is None:
                    try:
                        del self._priority_local.priority
                    except AttributeError:
                        pass
                else:
                    self._priority_local.priority = previous

        if len(pairs) <= chunk_size:
            return _single_with_priority(pairs)

        chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
        workers = min(len(chunks), max(1, max_workers))
        if workers <= 1:
            merged: Dict[Tuple[str, str], str] = {}
            for chunk in chunks:
                merged.update(_single_with_priority(chunk))
            return merged

        merged = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="batch-rcontent") as pool:
            futures = {pool.submit(_single_with_priority, c): c for c in chunks}
            for fut in as_completed(futures):
                try:
                    merged.update(fut.result())
                except Exception:
                    pass
        return merged

    def _batch_write_relation_content_single(
        self, pairs: List[Tuple[str, str]], window_text: str,
    ) -> Dict[Tuple[str, str], str]:
        """Single batch LLM call for relation content writing."""
        pair_list_str = "\n".join(f"  - {p[0]} 与 {p[1]}" for p in pairs)
        user_prompt = RELATION_BATCH_CONTENT_WRITE_USER.format(
            pair_list=pair_list_str,
            window_text=window_text,
        )
        messages = [
            {"role": "system", "content": self._source_language_system(RELATION_BATCH_CONTENT_WRITE_SYSTEM, window_text)},
            {"role": "user", "content": user_prompt},
        ]

        try:
            results, _ = self.call_llm_until_json_parses(
                messages, parse_fn=self._parse_batch_relation_content_list, json_parse_retries=2,
            )
            if isinstance(results, dict):
                return results
            return {}
        except (json.JSONDecodeError, LLMContextBudgetExceeded):
            return {}

    def _parse_batch_relation_content_list(self, response: str) -> Dict[Tuple[str, str], str]:
        """Parse batch relation content response:
        [{"entity1": "A", "entity2": "B", "content": "..."}, ...]
        """
        data = self._parse_json_response(response)
        result: Dict[Tuple[str, str], str] = {}
        # Unify list and dict-wrapper branches — extract items list once
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("relations")
            if items is None:
                items = data.get("data")
            if not isinstance(items, list):
                items = []
        else:
            items = []
        for item in items:
            if isinstance(item, dict) and "entity1" in item and "entity2" in item and "content" in item:
                a = item["entity1"].strip()
                b = item["entity2"].strip()
                content = item["content"].strip()
                if a and b and content:
                    key = (a, b) if a <= b else (b, a)
                    if key not in result:
                        result[key] = content
        return result

    # ------------------------------------------------------------------
    # Per-Pair Relation Content Writing
    # ------------------------------------------------------------------

    def write_relation_content(
        self, entity_a: str, entity_b: str, window_text: str,
    ) -> str:
        """Write a short description of the relationship between two entities.

        Returns a string describing the relationship.
        """
        user_prompt = RELATION_CONTENT_WRITE_USER.format(
            entity_a=entity_a, entity_b=entity_b, window_text=window_text,
        )
        messages = [
            {"role": "system", "content": self._source_language_system(RELATION_CONTENT_WRITE_SYSTEM, window_text)},
            {"role": "user", "content": user_prompt},
        ]

        try:
            result, _ = self.call_llm_until_json_parses(
                messages, parse_fn=self._parse_content_field, json_parse_retries=2,
            )
            fallback = f"{entity_a} is related to {entity_b}." if getattr(
                self, "preserve_source_language", False
            ) else f"{entity_a}与{entity_b}存在关联"
            return result if result else fallback
        except Exception:
            return (f"{entity_a} is related to {entity_b}." if getattr(
                self, "preserve_source_language", False
            ) else f"{entity_a}与{entity_b}存在关联")

    # ------------------------------------------------------------------
    # Entity Alignment Judgment — three-way
    # ------------------------------------------------------------------

    def judge_entity_alignment(
        self, name_a: str, content_a: str, name_b: str, content_b: str,
        *, name_match_type: str = "none",
    ) -> Dict[str, Any]:
        """Judge whether two entities describe the same object.

        Returns:
            {"verdict": "same"|"different"|"uncertain",
             "confidence": 0.0-1.0,
             "reason": "..."}
        """
        return self._judge_entity_alignment_llm(
            name_a, content_a, name_b, content_b,
            name_match_type=name_match_type,
        )

    def _judge_entity_alignment_llm(
        self, name_a: str, content_a: str, name_b: str, content_b: str,
        *, name_match_type: str = "none",
    ) -> Dict[str, Any]:
        """judge_entity_alignment 的原始直调实现（真实 LLM 请求）。"""

        snippet_a = content_a[:500] if len(content_a) > 500 else content_a
        snippet_b = content_b[:500] if len(content_b) > 500 else content_b

        # Build name relationship hint for the prompt
        name_relationship = ""
        if name_match_type == "substring":
            name_relationship = f"子串关系：\"{name_a}\" 和 \"{name_b}\" 存在子串包含关系，强烈暗示是同一对象的简称"
        elif name_match_type == "exact":
            name_relationship = f"核心名称完全相同：\"{name_a}\" 和 \"{name_b}\" 去除修饰后一致"

        user_prompt = ENTITY_ALIGNMENT_JUDGE_USER.format(
            name_a=name_a, content_a=snippet_a,
            name_b=name_b, content_b=snippet_b,
            name_relationship=name_relationship,
        )
        messages = [
            {"role": "system", "content": ENTITY_ALIGNMENT_JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        def _parse_alignment(response: str) -> Dict[str, Any]:
            data = self._parse_json_response(response)
            if isinstance(data, dict):
                verdict = str(data.get("verdict", "uncertain")).lower().strip()
                if verdict not in _VALID_VERDICTS:
                    verdict = "uncertain"
                confidence = 0.5
                try:
                    confidence = float(data.get("confidence", 0.5))
                    confidence = max(0.0, min(1.0, confidence))
                except (TypeError, ValueError):
                    pass
                return {"verdict": verdict, "confidence": confidence}
            # Fallback: parse old-style boolean
            if isinstance(data, bool):
                return {
                    "verdict": "same" if data else "different",
                    "confidence": 0.7,
                }
            return {"verdict": "uncertain", "confidence": 0.3}

        try:
            result, _ = self.call_llm_until_json_parses(
                messages, parse_fn=_parse_alignment, json_parse_retries=2,
            )
            return result
        except (json.JSONDecodeError, LLMContextBudgetExceeded):
            return {"verdict": "uncertain", "confidence": 0.0}
