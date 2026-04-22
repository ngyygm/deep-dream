"""
V2 Extraction Mixin for LLMClient.

Provides focused, single-task LLM methods for the redesigned Remember pipeline.
Each method does exactly one thing: extract names, write content, discover pairs, etc.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from .errors import LLMContextBudgetExceeded
from .v2_prompts import (
    ENTITY_NAMES_EXTRACT_NAMED_SYSTEM,
    ENTITY_NAMES_EXTRACT_NAMED_USER,
    ENTITY_NAMES_EXTRACT_NAMED_CONTINUE,
    ENTITY_NAMES_EXTRACT_ABSTRACT_SYSTEM,
    ENTITY_NAMES_EXTRACT_ABSTRACT_USER,
    ENTITY_NAMES_EXTRACT_ABSTRACT_CONTINUE,
    ENTITY_NAMES_EXTRACT_EVENTS_SYSTEM,
    ENTITY_NAMES_EXTRACT_EVENTS_USER,
    ENTITY_NAMES_EXTRACT_EVENTS_CONTINUE,
    ENTITY_NAMES_EXTRACT_NAMED_SYSTEM as ENTITY_NAMES_EXTRACT_SYSTEM,
    ENTITY_NAMES_EXTRACT_NAMED_USER as ENTITY_NAMES_EXTRACT_USER,
    ENTITY_NAMES_SUPPLEMENT_SYSTEM,
    ENTITY_NAMES_SUPPLEMENT_USER,
    ENTITY_NAMES_REFLECTION_SYSTEM,
    ENTITY_NAMES_REFLECTION_USER,
    ENTITY_CONTENT_WRITE_SYSTEM,
    ENTITY_CONTENT_WRITE_USER,
    RELATION_PAIRS_DISCOVER_SYSTEM,
    RELATION_PAIRS_DISCOVER_USER,
    RELATION_PAIRS_EXPAND_SYSTEM,
    RELATION_PAIRS_EXPAND_USER,
    RELATION_ORPHAN_RECOVERY_SYSTEM,
    RELATION_ORPHAN_RECOVERY_USER,
    RELATION_CONTENT_WRITE_SYSTEM,
    RELATION_CONTENT_WRITE_USER,
    ENTITY_ALIGNMENT_JUDGE_V2_SYSTEM,
    ENTITY_ALIGNMENT_JUDGE_V2_USER,
    RELATION_ALIGNMENT_JUDGE_SYSTEM,
    RELATION_ALIGNMENT_JUDGE_USER,
)


def _parse_name_list(self_ref, response: str) -> List[str]:
    """Shared parser for name list responses."""
    data = self_ref._parse_json_response(response)
    if isinstance(data, list):
        return [str(item).strip() for item in data if str(item).strip()]
    if isinstance(data, dict):
        for key in ("entities", "names", "data"):
            if key in data and isinstance(data[key], list):
                return [str(item).strip() for item in data[key] if str(item).strip()]
        if "name" in data:
            return [str(data["name"]).strip()]
    return []


class _V2ExtractionMixin:
    """V2 extraction methods for LLMClient — one task per call."""

    # ------------------------------------------------------------------
    # Step 2: Entity Name Extraction (original, backward-compatible)
    # ------------------------------------------------------------------

    def extract_entity_names_v2(self, window_text: str) -> List[str]:
        """Extract all entity names from a window of text. One LLM call."""
        user_prompt = ENTITY_NAMES_EXTRACT_USER.format(window_text=window_text)
        messages = [
            {"role": "system", "content": ENTITY_NAMES_EXTRACT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        try:
            names, _ = self.call_llm_until_json_parses(
                messages,
                parse_fn=lambda r: _parse_name_list(self, r),
                json_parse_retries=3,
            )
            return names
        except (json.JSONDecodeError, LLMContextBudgetExceeded):
            return []

    # ------------------------------------------------------------------
    # Step 2a: Named Entity Extraction (focused)
    # ------------------------------------------------------------------

    def extract_entity_names_named_v2(self, window_text: str, max_rounds: int = 1) -> List[str]:
        """Extract concrete/named entity names. Multi-round with continuation."""
        user_prompt = ENTITY_NAMES_EXTRACT_NAMED_USER.format(window_text=window_text)
        messages = [
            {"role": "system", "content": ENTITY_NAMES_EXTRACT_NAMED_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        try:
            names, response_text = self.call_llm_until_json_parses(
                messages,
                parse_fn=lambda r: _parse_name_list(self, r),
                json_parse_retries=3,
            )
        except (json.JSONDecodeError, LLMContextBudgetExceeded):
            return []

        # Multi-round continuation
        if max_rounds > 1 and names:
            messages.append({"role": "assistant", "content": response_text})
            all_names = list(names)
            seen_lower = {n.lower() for n in all_names}

            for _ in range(max_rounds - 1):
                continue_prompt = ENTITY_NAMES_EXTRACT_NAMED_CONTINUE
                if not self._can_continue_multi_round(
                    messages, next_user_content=continue_prompt,
                    stage_label="步骤2a(具名实体续轮)",
                ):
                    break
                messages.append({"role": "user", "content": continue_prompt})
                try:
                    round_names, round_text = self.call_llm_until_json_parses(
                        messages,
                        parse_fn=lambda r: _parse_name_list(self, r),
                        json_parse_retries=2,
                    )
                except (json.JSONDecodeError, LLMContextBudgetExceeded):
                    break
                new_names = [n for n in round_names if n.lower() not in seen_lower]
                if not new_names:
                    break
                all_names.extend(new_names)
                seen_lower.update(n.lower() for n in new_names)
                messages.append({"role": "assistant", "content": round_text})

            return all_names
        return names

    # ------------------------------------------------------------------
    # Step 2b: Abstract Concept Extraction (focused)
    # ------------------------------------------------------------------

    def extract_entity_names_abstract_v2(self, window_text: str, max_rounds: int = 1) -> List[str]:
        """Extract abstract concepts, theories, methods, time periods. Multi-round."""
        user_prompt = ENTITY_NAMES_EXTRACT_ABSTRACT_USER.format(window_text=window_text)
        messages = [
            {"role": "system", "content": ENTITY_NAMES_EXTRACT_ABSTRACT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        try:
            names, response_text = self.call_llm_until_json_parses(
                messages,
                parse_fn=lambda r: _parse_name_list(self, r),
                json_parse_retries=3,
            )
        except (json.JSONDecodeError, LLMContextBudgetExceeded):
            return []

        if max_rounds > 1 and names:
            messages.append({"role": "assistant", "content": response_text})
            all_names = list(names)
            seen_lower = {n.lower() for n in all_names}

            for _ in range(max_rounds - 1):
                continue_prompt = ENTITY_NAMES_EXTRACT_ABSTRACT_CONTINUE
                if not self._can_continue_multi_round(
                    messages, next_user_content=continue_prompt,
                    stage_label="步骤2b(抽象概念续轮)",
                ):
                    break
                messages.append({"role": "user", "content": continue_prompt})
                try:
                    round_names, round_text = self.call_llm_until_json_parses(
                        messages,
                        parse_fn=lambda r: _parse_name_list(self, r),
                        json_parse_retries=2,
                    )
                except (json.JSONDecodeError, LLMContextBudgetExceeded):
                    break
                new_names = [n for n in round_names if n.lower() not in seen_lower]
                if not new_names:
                    break
                all_names.extend(new_names)
                seen_lower.update(n.lower() for n in new_names)
                messages.append({"role": "assistant", "content": round_text})

            return all_names
        return names

    # ------------------------------------------------------------------
    # Step 2c: Event / Process Extraction (focused)
    # ------------------------------------------------------------------

    def extract_entity_names_events_v2(self, window_text: str, max_rounds: int = 1) -> List[str]:
        """Extract events, processes, milestones. Multi-round."""
        user_prompt = ENTITY_NAMES_EXTRACT_EVENTS_USER.format(window_text=window_text)
        messages = [
            {"role": "system", "content": ENTITY_NAMES_EXTRACT_EVENTS_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        try:
            names, response_text = self.call_llm_until_json_parses(
                messages,
                parse_fn=lambda r: _parse_name_list(self, r),
                json_parse_retries=3,
            )
        except (json.JSONDecodeError, LLMContextBudgetExceeded):
            return []

        if max_rounds > 1 and names:
            messages.append({"role": "assistant", "content": response_text})
            all_names = list(names)
            seen_lower = {n.lower() for n in all_names}

            for _ in range(max_rounds - 1):
                continue_prompt = ENTITY_NAMES_EXTRACT_EVENTS_CONTINUE
                if not self._can_continue_multi_round(
                    messages, next_user_content=continue_prompt,
                    stage_label="步骤2c(事件续轮)",
                ):
                    break
                messages.append({"role": "user", "content": continue_prompt})
                try:
                    round_names, round_text = self.call_llm_until_json_parses(
                        messages,
                        parse_fn=lambda r: _parse_name_list(self, r),
                        json_parse_retries=2,
                    )
                except (json.JSONDecodeError, LLMContextBudgetExceeded):
                    break
                new_names = [n for n in round_names if n.lower() not in seen_lower]
                if not new_names:
                    break
                all_names.extend(new_names)
                seen_lower.update(n.lower() for n in new_names)
                messages.append({"role": "assistant", "content": round_text})

            return all_names
        return names

    # ------------------------------------------------------------------
    # Step 2d: Missing Entity Supplement (补充提取)
    # ------------------------------------------------------------------

    def extract_entity_names_supplement_v2(
        self, window_text: str, existing_names: List[str]
    ) -> List[str]:
        """Find entity names that were missed in previous extraction rounds."""
        names_str = "、".join(existing_names) if existing_names else "（无）"
        user_prompt = ENTITY_NAMES_SUPPLEMENT_USER.format(
            existing_names=names_str,
            window_text=window_text,
        )
        messages = [
            {"role": "system", "content": ENTITY_NAMES_SUPPLEMENT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        try:
            names, _ = self.call_llm_until_json_parses(
                messages,
                parse_fn=lambda r: _parse_name_list(self, r),
                json_parse_retries=2,
            )
            return names
        except (json.JSONDecodeError, LLMContextBudgetExceeded):
            return []

    # ------------------------------------------------------------------
    # Step 2e: Reflection / Self-Critique (structured diagnostic)
    # ------------------------------------------------------------------

    def extract_entity_names_reflection_v2(
        self, window_text: str, existing_names: List[str]
    ) -> List[str]:
        """Structured self-critique to find missed entities via diagnostic questions."""
        names_str = "、".join(existing_names) if existing_names else "（无）"
        user_prompt = ENTITY_NAMES_REFLECTION_USER.format(
            existing_names=names_str,
            window_text=window_text,
        )
        messages = [
            {"role": "system", "content": ENTITY_NAMES_REFLECTION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        try:
            names, _ = self.call_llm_until_json_parses(
                messages,
                parse_fn=lambda r: _parse_name_list(self, r),
                json_parse_retries=2,
            )
            return names
        except (json.JSONDecodeError, LLMContextBudgetExceeded):
            return []

    # ------------------------------------------------------------------
    # Step 4: Per-Entity Content Writing
    # ------------------------------------------------------------------

    def write_entity_content_v2(self, entity_name: str, window_text: str) -> str:
        """Write a description for a single entity. One LLM call."""
        user_prompt = ENTITY_CONTENT_WRITE_USER.format(
            entity_name=entity_name,
            window_text=window_text,
        )
        messages = [
            {"role": "system", "content": ENTITY_CONTENT_WRITE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        def _parse_content(response: str) -> str:
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

        # Track raw response for fallback extraction
        raw_holder = [""]

        def _parse_with_capture(response: str) -> str:
            raw_holder[0] = response
            return _parse_content(response)

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
        if not raw_response or not raw_response.strip():
            return ""

        text = raw_response.strip()

        # 1. Try to extract content from inside ```json ... ``` blocks even if malformed
        import re
        json_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        for block in json_blocks:
            block = block.strip()
            # Try to find a "content" value in the malformed JSON
            content_match = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', block)
            if content_match:
                val = content_match.group(1)
                # Unescape basic JSON escapes
                val = val.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                if len(val) >= min_length:
                    return val

        # 2. Strip markdown code fences and common prefixes
        cleaned = re.sub(r'```(?:json)?\s*', '', text)
        cleaned = re.sub(r'```\s*$', '', cleaned)
        cleaned = cleaned.strip()

        # 3. If the remaining text looks like a description (not JSON), use it
        if cleaned and not cleaned.startswith(('{', '[')):
            # Remove common template phrases
            cleaned = re.sub(r'^["\']|["\']$', '', cleaned)
            if len(cleaned) >= min_length:
                return cleaned

        return ""

    # ------------------------------------------------------------------
    # Step 6a: Relation Pair Discovery
    # ------------------------------------------------------------------

    def discover_relation_pairs_v2(
        self, entity_names: List[str], window_text: str
    ) -> List[Tuple[str, str]]:
        """Discover entity pairs with meaningful relations. One LLM call."""
        entity_list_str = "、".join(entity_names)
        user_prompt = RELATION_PAIRS_DISCOVER_USER.format(
            entity_names=entity_list_str,
            window_text=window_text,
        )
        messages = [
            {"role": "system", "content": RELATION_PAIRS_DISCOVER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        try:
            pairs, _ = self.call_llm_until_json_parses(
                messages,
                parse_fn=lambda r: self._parse_pair_list(r),
                json_parse_retries=3,
            )
            return pairs
        except (json.JSONDecodeError, LLMContextBudgetExceeded):
            return []

    # ------------------------------------------------------------------
    # Step 6a-chunked: Chunked Relation Pair Discovery (for large entity lists)
    # ------------------------------------------------------------------

    def discover_relation_pairs_chunked_v2(
        self, entity_names: List[str], window_text: str, chunk_size: int = 10
    ) -> List[Tuple[str, str]]:
        """Discover relation pairs using chunked entity lists.

        When entity_names has > chunk_size entries, splits into overlapping chunks
        and makes one LLM call per chunk, then merges all pairs.
        """
        if len(entity_names) <= chunk_size:
            return self.discover_relation_pairs_v2(entity_names, window_text)

        all_pairs = set()
        # Create overlapping chunks with stride = chunk_size - 2
        stride = max(1, chunk_size - 2)
        for start in range(0, len(entity_names), stride):
            chunk = entity_names[start:start + chunk_size]
            if len(chunk) < 2:
                break
            pairs = self.discover_relation_pairs_v2(chunk, window_text)
            for p in pairs:
                all_pairs.add(tuple(sorted(p)))

        return list(all_pairs)

    # ------------------------------------------------------------------
    # Step 6b: Relation Pair Expansion (find missed pairs)
    # ------------------------------------------------------------------

    def discover_relation_pairs_expand_v2(
        self,
        entity_names: List[str],
        existing_pairs: List[Tuple[str, str]],
        window_text: str,
    ) -> List[Tuple[str, str]]:
        """Find relation pairs missed in the first discovery round. One LLM call."""
        pairs_str = "、".join(
            f"({a}, {b})" for a, b in existing_pairs
        ) if existing_pairs else "（无）"
        entity_list_str = "、".join(entity_names)
        user_prompt = RELATION_PAIRS_EXPAND_USER.format(
            existing_pairs=pairs_str,
            entity_names=entity_list_str,
            window_text=window_text,
        )
        messages = [
            {"role": "system", "content": RELATION_PAIRS_EXPAND_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        try:
            pairs, _ = self.call_llm_until_json_parses(
                messages,
                parse_fn=lambda r: self._parse_pair_list(r),
                json_parse_retries=2,
            )
            return pairs
        except (json.JSONDecodeError, LLMContextBudgetExceeded):
            return []

    # ------------------------------------------------------------------
    # Shared pair parser
    # ------------------------------------------------------------------

    def _parse_pair_list(self, response: str) -> List[Tuple[str, str]]:
        """Parse LLM response into a list of (entity1, entity2) tuples."""
        data = self._parse_json_response(response)
        pairs = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    a, b = str(item[0]).strip(), str(item[1]).strip()
                    if a and b and a != b:
                        pair = tuple(sorted([a, b]))
                        if pair not in pairs:
                            pairs.append(pair)
                elif isinstance(item, dict):
                    a = str(item.get("entity1") or item.get("entity1_name") or "").strip()
                    b = str(item.get("entity2") or item.get("entity2_name") or "").strip()
                    if a and b and a != b:
                        pair = tuple(sorted([a, b]))
                        if pair not in pairs:
                            pairs.append(pair)
        return pairs

    # ------------------------------------------------------------------
    # Step 6c: Orphan Entity Relation Recovery
    # ------------------------------------------------------------------

    def discover_relation_pairs_orphan_v2(
        self,
        orphan_names: List[str],
        all_entity_names: List[str],
        window_text: str,
    ) -> List[Tuple[str, str]]:
        """Find relations for entities that have zero relations. One LLM call."""
        orphan_str = "、".join(orphan_names)
        all_str = "、".join(all_entity_names)
        user_prompt = RELATION_ORPHAN_RECOVERY_USER.format(
            orphan_names=orphan_str,
            all_entity_names=all_str,
            window_text=window_text,
        )
        messages = [
            {"role": "system", "content": RELATION_ORPHAN_RECOVERY_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        try:
            pairs, _ = self.call_llm_until_json_parses(
                messages,
                parse_fn=lambda r: self._parse_pair_list(r),
                json_parse_retries=2,
            )
            return pairs
        except (json.JSONDecodeError, LLMContextBudgetExceeded):
            return []

    # ------------------------------------------------------------------
    # Step 7: Relation Content Writing
    # ------------------------------------------------------------------

    def write_relation_content_v2(
        self, entity_a: str, entity_b: str, window_text: str,
    ) -> str:
        """Write a short description of the relationship between two entities.

        Returns a string describing the relationship.
        """
        from .v2_prompts import RELATION_CONTENT_WRITE_SYSTEM, RELATION_CONTENT_WRITE_USER

        user_prompt = RELATION_CONTENT_WRITE_USER.format(
            entity_a=entity_a, entity_b=entity_b, window_text=window_text,
        )
        messages = [
            {"role": "system", "content": RELATION_CONTENT_WRITE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        def _parse_content(response: str) -> str:
            data = self._parse_json_response(response)
            if isinstance(data, dict):
                return str(data.get("content", "")).strip()
            if isinstance(data, str):
                return data.strip()
            return ""

        try:
            result, _ = self.call_llm_until_json_parses(
                messages, parse_fn=_parse_content, json_parse_retries=2,
            )
            return result if result else f"{entity_a}与{entity_b}存在关联"
        except Exception:
            return f"{entity_a}与{entity_b}存在关联"

    # ------------------------------------------------------------------
    # Step 9v2: Entity Alignment Judgment (three-way: same/different/uncertain)
    # ------------------------------------------------------------------

    def judge_entity_alignment_v2(
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
            name_relationship = f"子串关系：\"{name_a}\" 和 \"{name_b}\" 存在子串包含关系，强烈暗示是同一对象的简称（如甄士隐↔士隐、贾宝玉↔宝玉）"
        elif name_match_type == "exact":
            name_relationship = f"核心名称完全相同：\"{name_a}\" 和 \"{name_b}\" 去除修饰后一致"

        user_prompt = ENTITY_ALIGNMENT_JUDGE_V2_USER.format(
            name_a=name_a, content_a=snippet_a,
            name_b=name_b, content_b=snippet_b,
            name_relationship=name_relationship,
        )
        messages = [
            {"role": "system", "content": ENTITY_ALIGNMENT_JUDGE_V2_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        def _parse_alignment(response: str) -> Dict[str, Any]:
            data = self._parse_json_response(response)
            if isinstance(data, dict):
                verdict = str(data.get("verdict", "uncertain")).lower().strip()
                if verdict not in ("same", "different", "uncertain"):
                    verdict = "uncertain"
                confidence = 0.5
                try:
                    confidence = float(data.get("confidence", 0.5))
                    confidence = max(0.0, min(1.0, confidence))
                except (TypeError, ValueError):
                    pass
                reason = str(data.get("reason", "")).strip()
                return {"verdict": verdict, "confidence": confidence, "reason": reason}
            # Fallback: parse old-style boolean
            if isinstance(data, bool):
                return {
                    "verdict": "same" if data else "different",
                    "confidence": 0.7,
                    "reason": "",
                }
            return {"verdict": "uncertain", "confidence": 0.3, "reason": "parse failure"}

        try:
            result, _ = self.call_llm_until_json_parses(
                messages, parse_fn=_parse_alignment, json_parse_retries=2,
            )
            return result
        except (json.JSONDecodeError, LLMContextBudgetExceeded):
            return {"verdict": "uncertain", "confidence": 0.0, "reason": "LLM error"}

    # ------------------------------------------------------------------
    # Step 10: Relation Alignment Judgment
    # ------------------------------------------------------------------

    def judge_relation_match_v2(
        self, entity_a: str, entity_b: str,
        new_content: str, existing_contents: List[str],
    ) -> int:
        """Judge whether a new relation matches an existing one.
        Returns index of match (0-based), or -1 if no match.
        """
        existing_text = "\n".join(
            f"{i + 1}. {c}" for i, c in enumerate(existing_contents)
        )
        user_prompt = RELATION_ALIGNMENT_JUDGE_USER.format(
            entity_a=entity_a, entity_b=entity_b,
            new_content=new_content,
            existing_relations_text=existing_text,
        )
        messages = [
            {"role": "system", "content": RELATION_ALIGNMENT_JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        def _parse_index(response: str) -> int:
            data = self._parse_json_response(response)
            if isinstance(data, dict) and "match_index" in data:
                idx = int(data["match_index"])
                return idx if 0 <= idx < len(existing_contents) else -1
            if isinstance(data, int):
                return data if 0 <= data < len(existing_contents) else -1
            return -1

        try:
            result, _ = self.call_llm_until_json_parses(
                messages, parse_fn=_parse_index, json_parse_retries=2,
            )
            return result
        except (json.JSONDecodeError, LLMContextBudgetExceeded):
            return -1
