"""
Extraction Mixin for LLMClient.

Designed for capable models (gemma4:26b) with think mode:
- Single comprehensive extraction prompt instead of category decomposition
- Conversational refinement ("find more") instead of separate category rounds
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# Pre-compiled regex patterns for _extract_text_from_raw
_JSON_BLOCK_RE = re.compile(r'```(?:json)?\s*\n?(.*?)\n?\s*```', re.DOTALL)
_CONTENT_VALUE_RE = re.compile(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"')
_OPEN_FENCE_RE = re.compile(r'```(?:json)?\s*')
_CLOSE_FENCE_RE = re.compile(r'```\s*$')
_QUOTE_TRIM_RE = re.compile(r'^["\']|["\']$')
_VALID_VERDICTS = frozenset(("same", "different", "uncertain"))

from .errors import LLMContextBudgetExceeded
from .prompts import (
    ENTITY_EXTRACT_SYSTEM,
    ENTITY_EXTRACT_USER,
    ENTITY_REFINE_USER,
    RELATION_DISCOVER_SYSTEM,
    RELATION_DISCOVER_USER,
    RELATION_REFINE_USER,
    ENTITY_CONTENT_WRITE_SYSTEM,
    ENTITY_CONTENT_WRITE_USER,
    ENTITY_BATCH_CONTENT_WRITE_SYSTEM,
    ENTITY_BATCH_CONTENT_WRITE_USER,
    RELATION_CONTENT_WRITE_SYSTEM,
    RELATION_CONTENT_WRITE_USER,
    RELATION_BATCH_CONTENT_WRITE_SYSTEM,
    RELATION_BATCH_CONTENT_WRITE_USER,
    ENTITY_ALIGNMENT_JUDGE_SYSTEM,
    ENTITY_ALIGNMENT_JUDGE_USER,
)


class _LLMExtractionMixin:
    """Extraction methods for LLMClient — comprehensive prompts for strong models."""

    # ------------------------------------------------------------------
    # Generic extraction with conversational refinement
    # ------------------------------------------------------------------

    def _extract_with_refinement(
        self,
        system_prompt: str,
        user_prompt: str,
        refine_prompt: str,
        parse_fn,
        key_fn,
        max_refine_rounds: int,
        stage_label: str,
    ) -> Tuple[list, Dict[str, int]]:
        """Generic extraction with conversational refinement.

        Args:
            system_prompt: System message.
            user_prompt: Formatted user message.
            refine_prompt: Prompt for refinement rounds.
            parse_fn: Parses LLM response into items.
            key_fn: Extracts dedup key from an item.
            max_refine_rounds: Max refinement rounds.
            stage_label: Label for logging (e.g. "实体").

        Returns:
            (deduplicated_items, refine_stats).
        """
        refine_stats = {"initial": 0, "refine_added": 0, "rounds_run": 0}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Round 1: initial extraction
        try:
            items, response_text = self.call_llm_until_json_parses(
                messages, parse_fn=parse_fn, json_parse_retries=3,
            )
        except (json.JSONDecodeError, LLMContextBudgetExceeded):
            return [], refine_stats

        if not items:
            return [], refine_stats

        # Dedup initial results
        seen: set = set()
        all_items: list = []
        for item in items:
            k = key_fn(item)
            if k not in seen:
                seen.add(k)
                all_items.append(item)

        refine_stats["initial"] = len(all_items)

        if not all_items or max_refine_rounds < 1:
            return all_items, refine_stats

        messages.append({"role": "assistant", "content": response_text})

        # Refinement rounds
        for round_i in range(max_refine_rounds):
            if not self._can_continue_multi_round(
                messages, next_user_content=refine_prompt,
                stage_label=f"{stage_label}精炼",
            ):
                break
            messages.append({"role": "user", "content": refine_prompt})
            try:
                round_items, round_text = self.call_llm_until_json_parses(
                    messages, parse_fn=parse_fn, json_parse_retries=2,
                )
            except (json.JSONDecodeError, LLMContextBudgetExceeded):
                break
            new_items = []
            for item in round_items:
                k = key_fn(item)
                if k not in seen:
                    seen.add(k)
                    new_items.append(item)
            if not new_items:
                break
            all_items.extend(new_items)
            refine_stats["rounds_run"] = round_i + 1
            refine_stats["refine_added"] += len(new_items)
            messages.append({"role": "assistant", "content": round_text})

        return all_items, refine_stats

    # ------------------------------------------------------------------
    # Step 1 + 1b: Entity Extraction with Conversational Refinement
    # ------------------------------------------------------------------

    def extract_entities(
        self, window_text: str, max_refine_rounds: int = 2
    ) -> Tuple[List[str], Dict[str, int]]:
        """Extract all entities using comprehensive prompt + conversational refinement."""
        return self._extract_with_refinement(
            system_prompt=ENTITY_EXTRACT_SYSTEM,
            user_prompt=ENTITY_EXTRACT_USER.format(window_text=window_text),
            refine_prompt=ENTITY_REFINE_USER,
            parse_fn=self._parse_name_list,
            key_fn=lambda n: n.lower(),
            max_refine_rounds=max_refine_rounds,
            stage_label="实体",
        )

    # ------------------------------------------------------------------
    # Step 5 + 5b: Relation Discovery with Conversational Refinement
    # ------------------------------------------------------------------

    def discover_relations(
        self,
        entity_names: List[str],
        window_text: str,
        max_refine_rounds: int = 1,
    ) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
        """Discover all relation pairs using comprehensive prompt + refinement."""
        entity_list_str = "、".join(entity_names)
        return self._extract_with_refinement(
            system_prompt=RELATION_DISCOVER_SYSTEM,
            user_prompt=RELATION_DISCOVER_USER.format(
                entity_names=entity_list_str,
                window_text=window_text,
            ),
            refine_prompt=RELATION_REFINE_USER,
            parse_fn=self._parse_pair_list,
            key_fn=lambda p: p,  # _parse_pair_list already canonicalizes order
            max_refine_rounds=max_refine_rounds,
            stage_label="关系",
        )

    # ------------------------------------------------------------------
    # Shared parser
    # ------------------------------------------------------------------

    def _parse_name_list(self, response: str) -> List[str]:
        """Parse entity name list from LLM response."""
        data = self._parse_json_response(response)
        if isinstance(data, list):
            return [_s for item in data if (_s := str(item).strip())]
        if isinstance(data, dict):
            for key in ("entities", "names", "data"):
                if key in data and isinstance(data[key], list):
                    return [_s for item in data[key] if (_s := str(item).strip())]
            if "name" in data:
                return [str(data["name"]).strip()]
        return []

    # ------------------------------------------------------------------
    # Shared pair parser
    # ------------------------------------------------------------------

    def _parse_pair_list(self, response: str) -> List[Tuple[str, str]]:
        """Parse LLM response into a list of (entity1, entity2) tuples."""
        data = self._parse_json_response(response)
        pairs = []
        seen: set = set()
        if isinstance(data, list):
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    a, b = item[0].strip(), item[1].strip()
                    if a and b and a != b:
                        pair = (a, b) if a <= b else (b, a)
                        if pair not in seen:
                            seen.add(pair)
                            pairs.append(pair)
                elif isinstance(item, dict):
                    a = str(item.get("entity1") or item.get("entity1_name") or "").strip()
                    b = str(item.get("entity2") or item.get("entity2_name") or "").strip()
                    if a and b and a != b:
                        pair = (a, b) if a <= b else (b, a)
                        if pair not in seen:
                            seen.add(pair)
                            pairs.append(pair)
        return pairs

    # ------------------------------------------------------------------
    # Shared content parser
    # ------------------------------------------------------------------

    def _parse_content_field(self, response: str) -> str:
        """Parse a 'content' field from JSON response — used by entity and relation writing."""
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
    # Per-Entity Content Writing
    # ------------------------------------------------------------------

    def write_entity_content(self, entity_name: str, window_text: str) -> str:
        """Write a description for a single entity. One LLM call."""
        user_prompt = ENTITY_CONTENT_WRITE_USER.format(
            entity_name=entity_name,
            window_text=window_text,
        )
        messages = [
            {"role": "system", "content": ENTITY_CONTENT_WRITE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        # Track raw response for fallback extraction
        raw_holder = [""]

        def _parse_with_capture(response: str) -> str:
            raw_holder[0] = response
            return self._parse_content_field(response)

        try:
            content, _ = self.call_llm_until_json_parses(
                messages, parse_fn=_parse_with_capture, json_parse_retries=2,
            )
            return content
        except (json.JSONDecodeError, LLMContextBudgetExceeded):
            pass

        # Last-resort: extract usable text from raw LLM response
        return self._extract_text_from_raw(raw_holder[0])

    @staticmethod
    def _extract_text_from_raw(raw_response: str, min_length: int = 15) -> str:
        """Try to extract meaningful content from a raw LLM response when JSON parsing fails.

        Handles cases where the model outputs valid descriptive text but in a format
        that cannot be parsed as JSON (e.g., plain text, markdown, or malformed JSON).
        """
        _t = raw_response.strip()
        if not _t:
            return ""

        text = _t

        # 1. Try to extract content from inside ```json ... ``` blocks even if malformed
        json_blocks = _JSON_BLOCK_RE.findall(text)
        for block in json_blocks:
            block = block.strip()
            # Try to find a "content" value in the malformed JSON
            content_match = _CONTENT_VALUE_RE.search(block)
            if content_match:
                val = content_match.group(1)
                # Unescape basic JSON escapes (longest escape first to avoid double-unescape)
                val = val.replace('\\\\', '\\').replace('\\"', '"').replace('\\n', '\n')
                if len(val) >= min_length:
                    return val

        # 2. Strip markdown code fences and common prefixes
        cleaned = _OPEN_FENCE_RE.sub('', text)
        cleaned = _CLOSE_FENCE_RE.sub('', cleaned)
        cleaned = cleaned.strip()

        # 3. If the remaining text looks like a description (not JSON), use it
        if cleaned and not cleaned.startswith(('{', '[')):
            # Remove common template phrases
            cleaned = _QUOTE_TRIM_RE.sub('', cleaned)
            if len(cleaned) >= min_length:
                return cleaned

        return ""

    # ------------------------------------------------------------------
    # Batch Entity Content Writing
    # ------------------------------------------------------------------

    def batch_write_entity_content(
        self, entity_names: List[str], window_text: str,
    ) -> Dict[str, str]:
        """Write descriptions for all entities in a single LLM call.

        Returns:
            Dict mapping entity name -> content string.
            Entities not returned by the LLM are absent from the dict.
        """
        entity_list_str = "、".join(entity_names)
        user_prompt = ENTITY_BATCH_CONTENT_WRITE_USER.format(
            entity_names=entity_list_str,
            window_text=window_text,
        )
        messages = [
            {"role": "system", "content": ENTITY_BATCH_CONTENT_WRITE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        try:
            results, _ = self.call_llm_until_json_parses(
                messages, parse_fn=self._parse_batch_content_list, json_parse_retries=2,
            )
            if isinstance(results, dict):
                return results  # _parse_batch_content_list already filters empty values
            return {}
        except (json.JSONDecodeError, LLMContextBudgetExceeded):
            return {}

    def _parse_batch_content_list(self, response: str) -> Dict[str, str]:
        """Parse batch content response: [{"name": "X", "content": "Y"}, ...]"""
        data = self._parse_json_response(response)
        result: Dict[str, str] = {}
        # Unify list and dict-wrapper branches — extract items list once
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("entities") or data.get("data") or []
            if not isinstance(items, list):
                items = []
        else:
            items = []
        for item in items:
            if isinstance(item, dict) and "name" in item and "content" in item:
                name = item["name"].strip()
                content = item["content"].strip()
                if name and content:
                    result[name] = content
        return result

    # ------------------------------------------------------------------
    # Batch Relation Content Writing
    # ------------------------------------------------------------------

    def batch_write_relation_content(
        self, pairs: List[Tuple[str, str]], window_text: str,
    ) -> Dict[Tuple[str, str], str]:
        """Write relation descriptions for all pairs in a single LLM call.

        Returns:
            Dict mapping (entity1, entity2) -> content string.
            Pairs not returned by the LLM are absent from the dict.
        """
        pair_list_str = "\n".join(f"  - {a} 与 {b}" for a, b in pairs)
        user_prompt = RELATION_BATCH_CONTENT_WRITE_USER.format(
            pair_list=pair_list_str,
            window_text=window_text,
        )
        messages = [
            {"role": "system", "content": RELATION_BATCH_CONTENT_WRITE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        try:
            results, _ = self.call_llm_until_json_parses(
                messages, parse_fn=self._parse_batch_relation_content_list, json_parse_retries=2,
            )
            if isinstance(results, dict):
                return results  # _parse_batch_relation_content_list already filters empty values
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
            items = data.get("relations") or data.get("data") or []
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
            {"role": "system", "content": RELATION_CONTENT_WRITE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        try:
            result, _ = self.call_llm_until_json_parses(
                messages, parse_fn=self._parse_content_field, json_parse_retries=2,
            )
            return result if result else f"{entity_a}与{entity_b}存在关联"
        except Exception:
            return f"{entity_a}与{entity_b}存在关联"

    # ------------------------------------------------------------------
    # Entity Alignment Judgment — three-way
    # ------------------------------------------------------------------

    def judge_entity_alignment(
        self, name_a: str, content_a: str, name_b: str, content_b: str,
        *, name_match_type: str = "none",
    ) -> Dict[str, Any]:
        """Judge whether two entities describe the same object.

        Args:
            name_match_type: How the names matched in candidate search.
                "exact" = core names identical, "substring" = one is substring of the other,
                "none" = no special name relationship.

        Returns:
            {"verdict": "same"|"different"|"uncertain",
             "confidence": 0.0-1.0,
             "reason": "..."}
        """
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
