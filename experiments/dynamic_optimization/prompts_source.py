"""Import current production prompts from processor/llm/prompts.py.

Each step maps to one system_prompt constant (or function) used in production.
"""

import sys
import os

# Ensure processor is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from processor.llm.prompts import (
    UPDATE_MEMORY_CACHE_SYSTEM_PROMPT,
    EXTRACT_ENTITIES_SINGLE_PASS_SYSTEM_PROMPT,
    EXTRACT_RELATIONS_SINGLE_PASS_SYSTEM_PROMPT,
    EXTRACT_ENTITIES_BY_NAMES_SYSTEM_PROMPT,
    ENHANCE_ENTITY_CONTENT_SYSTEM_PROMPT,
    RESOLVE_ENTITY_CANDIDATES_BATCH_SYSTEM_PROMPT,
    RESOLVE_RELATION_PAIR_BATCH_SYSTEM_PROMPT,
)

# Step -> production system prompt
STEP_PROMPTS = {
    1: UPDATE_MEMORY_CACHE_SYSTEM_PROMPT,
    2: EXTRACT_ENTITIES_SINGLE_PASS_SYSTEM_PROMPT,
    3: EXTRACT_RELATIONS_SINGLE_PASS_SYSTEM_PROMPT,
    4: EXTRACT_ENTITIES_BY_NAMES_SYSTEM_PROMPT,
    5: ENHANCE_ENTITY_CONTENT_SYSTEM_PROMPT,
    6: RESOLVE_ENTITY_CANDIDATES_BATCH_SYSTEM_PROMPT,
    7: RESOLVE_RELATION_PAIR_BATCH_SYSTEM_PROMPT,
}


def get_production_prompt(step: int) -> str:
    """Get the current production system prompt for a step."""
    return STEP_PROMPTS[step]


def list_steps() -> list:
    """List all steps with production prompts."""
    return sorted(STEP_PROMPTS.keys())
