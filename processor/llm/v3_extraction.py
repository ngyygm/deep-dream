"""
V3 Extraction Mixin for LLMClient.

Designed for capable models (gemma4:26b) with think mode:
- Single comprehensive extraction prompt instead of category decomposition
- Conversational refinement ("find more") instead of separate category rounds
- Fewer total LLM calls than V2
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from .errors import LLMContextBudgetExceeded
from .v3_prompts import (
    V3_ENTITY_EXTRACT_SYSTEM,
    V3_ENTITY_EXTRACT_USER,
    V3_ENTITY_REFINE_USER,
    V3_RELATION_DISCOVER_SYSTEM,
    V3_RELATION_DISCOVER_USER,
    V3_RELATION_REFINE_USER,
)


class _V3ExtractionMixin:
    """V3 extraction methods for LLMClient — comprehensive prompts for strong models."""

    # ------------------------------------------------------------------
    # V3-1 + V3-1b: Entity Extraction with Conversational Refinement
    # ------------------------------------------------------------------

    def extract_entities_v3(
        self, window_text: str, max_refine_rounds: int = 2
    ) -> Tuple[List[str], Dict[str, int]]:
        """Extract all entities using comprehensive prompt + conversational refinement.

        Args:
            window_text: The text to extract from.
            max_refine_rounds: Max refinement rounds after initial extraction.

        Returns:
            (Deduplicated list of entity names, refine_stats dict).
            refine_stats: {"initial": N, "refine_added": M, "rounds_run": R}
        """
        refine_stats = {"initial": 0, "refine_added": 0, "rounds_run": 0}

        user_prompt = V3_ENTITY_EXTRACT_USER.format(window_text=window_text)
        messages = [
            {"role": "system", "content": V3_ENTITY_EXTRACT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        # Round 1: comprehensive extraction
        try:
            names, response_text = self.call_llm_until_json_parses(
                messages,
                parse_fn=self._parse_name_list_v3,
                json_parse_retries=3,
            )
        except (json.JSONDecodeError, LLMContextBudgetExceeded):
            return [], refine_stats

        if not names:
            return [], refine_stats

        all_names = list(names)
        refine_stats["initial"] = len(all_names)
        seen_lower = {n.lower() for n in all_names}
        messages.append({"role": "assistant", "content": response_text})

        # Refinement rounds
        for round_i in range(max_refine_rounds):
            existing_str = "、".join(all_names)
            refine_prompt = V3_ENTITY_REFINE_USER.format(
                existing_names=existing_str,
                window_text=window_text,
            )
            if not self._can_continue_multi_round(
                messages, next_user_content=refine_prompt,
                stage_label="V3(实体精炼)",
            ):
                break
            messages.append({"role": "user", "content": refine_prompt})
            try:
                round_names, round_text = self.call_llm_until_json_parses(
                    messages,
                    parse_fn=self._parse_name_list_v3,
                    json_parse_retries=2,
                )
            except (json.JSONDecodeError, LLMContextBudgetExceeded):
                break
            new_names = [n for n in round_names if n.lower() not in seen_lower]
            if not new_names:
                break
            all_names.extend(new_names)
            refine_stats["rounds_run"] = round_i + 1
            refine_stats["refine_added"] += len(new_names)
            seen_lower.update(n.lower() for n in new_names)
            messages.append({"role": "assistant", "content": round_text})

        return all_names, refine_stats

    # ------------------------------------------------------------------
    # V3-5 + V3-5b: Relation Discovery with Conversational Refinement
    # ------------------------------------------------------------------

    def discover_relations_v3(
        self,
        entity_names: List[str],
        window_text: str,
        max_refine_rounds: int = 1,
    ) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
        """Discover all relation pairs using comprehensive prompt + refinement.

        Args:
            entity_names: List of entity names to search relations between.
            window_text: The source text.
            max_refine_rounds: Max refinement rounds after initial discovery.

        Returns:
            (Deduplicated list of (entity1, entity2) tuples, refine_stats dict).
            refine_stats: {"initial": N, "refine_added": M, "rounds_run": R}
        """
        refine_stats = {"initial": 0, "refine_added": 0, "rounds_run": 0}

        entity_list_str = "、".join(entity_names)
        user_prompt = V3_RELATION_DISCOVER_USER.format(
            entity_names=entity_list_str,
            window_text=window_text,
        )
        messages = [
            {"role": "system", "content": V3_RELATION_DISCOVER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        # Round 1: comprehensive relation discovery
        try:
            pairs, response_text = self.call_llm_until_json_parses(
                messages,
                parse_fn=self._parse_pair_list,
                json_parse_retries=3,
            )
        except (json.JSONDecodeError, LLMContextBudgetExceeded):
            pairs = []

        all_pairs_set = set()
        all_pairs = []
        for p in pairs:
            key = tuple(sorted(p))
            if key not in all_pairs_set:
                all_pairs_set.add(key)
                all_pairs.append(p)

        refine_stats["initial"] = len(all_pairs)

        if not all_pairs or max_refine_rounds < 1:
            return all_pairs, refine_stats

        messages.append({"role": "assistant", "content": response_text})

        # Refinement rounds
        for round_i in range(max_refine_rounds):
            # Find orphans for the refine prompt
            connected = set()
            for a, b in all_pairs:
                connected.add(a.lower())
                connected.add(b.lower())
            orphans = [n for n in entity_names if n.lower() not in connected]

            pairs_str = "、".join(
                f"({a}, {b})" for a, b in all_pairs
            ) if all_pairs else "（无）"
            orphan_str = "、".join(orphans) if orphans else "（无）"
            entity_str = "、".join(entity_names)

            refine_prompt = V3_RELATION_REFINE_USER.format(
                orphan_names=orphan_str,
                existing_pairs=pairs_str,
                entity_names=entity_str,
                window_text=window_text,
            )
            if not self._can_continue_multi_round(
                messages, next_user_content=refine_prompt,
                stage_label="V3(关系精炼)",
            ):
                break
            messages.append({"role": "user", "content": refine_prompt})
            try:
                round_pairs, round_text = self.call_llm_until_json_parses(
                    messages,
                    parse_fn=self._parse_pair_list,
                    json_parse_retries=2,
                )
            except (json.JSONDecodeError, LLMContextBudgetExceeded):
                break
            round_added = 0
            for p in round_pairs:
                key = tuple(sorted(p))
                if key not in all_pairs_set:
                    all_pairs_set.add(key)
                    all_pairs.append(p)
                    round_added += 1
            if not round_added:
                break
            refine_stats["rounds_run"] = round_i + 1
            refine_stats["refine_added"] += round_added
            messages.append({"role": "assistant", "content": round_text})

        return all_pairs, refine_stats

    # ------------------------------------------------------------------
    # Shared parser (reuse V2's pattern)
    # ------------------------------------------------------------------

    def _parse_name_list_v3(self, response: str) -> List[str]:
        """Parse entity name list from LLM response."""
        from .v2_extraction import _parse_name_list
        return _parse_name_list(self, response)
