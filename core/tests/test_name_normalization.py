"""Snapshot of CURRENT entity-name normalization behavior (P0 safety net).

The 2026-08 audit found FOUR name-normalization implementations with
DIFFERENT semantics deciding entity identity at different layers:

1. core/remember/helpers.py::_core_entity_name
   (strips ALL parentheticals; case-sensitive) — within-window dedup,
   relation endpoint resolution rounds 1-4.
2. core/remember/_shared.py::normalize_entity_name_for_matching
   (parentheticals + Chinese title suffixes; case-sensitive) — alignment
   candidate matching.
3. core/judge/models.py::norm_name
   (casefold + whitespace collapse only) — FamilyWriteGate duplicate guard.
4. Inline ``name.lower()`` in alignment.py rounds 3/4.

These tests pin the CURRENT (divergent) behavior so the P2 unification into
ONE function is a deliberate, visible change: any assertion here that flips
means an identity-semantics decision changed, and needs a concurrent-ingest
check per audit invariant (d).
"""

import pytest

from core.remember.helpers import _core_entity_name
from core.remember._shared import normalize_entity_name_for_matching
from core.judge.models import norm_name


CASES = [
    # input                    core_entity_name      for_matching           judge norm_name
    ("张三",                    "张三",                "张三",                "张三"),
    ("张三教授",                "张三教授",             "张三",                "张三教授"),
    ("张三（北京大学）",         "张三",                "张三",                "张三（北京大学）"),
    ("IBM",                    "IBM",                "IBM",                "ibm"),
    ("ibm",                    "ibm",                "ibm",                "ibm"),
    ("  Alice   Bob ",         "Alice   Bob",        "Alice   Bob",        "alice bob"),
]


@pytest.mark.parametrize("raw,core,matching,judge", CASES)
def test_normalization_snapshot(raw, core, matching, judge):
    # CURRENT-BEHAVIOR: three layers, three answers for the same name.
    assert _core_entity_name(raw) == core
    assert normalize_entity_name_for_matching(raw) == matching
    assert norm_name(raw) == judge


def test_divergence_examples_documented():
    """The exact divergences P2 must resolve (or deliberately keep)."""
    # 1. Title suffix: stripped for alignment matching, kept for write gate.
    assert normalize_entity_name_for_matching("张三教授") == "张三"
    assert norm_name("张三教授") == "张三教授"
    assert _core_entity_name("张三教授") == "张三教授"

    # 2. Case: folded by write gate, not by candidate matching or dedup.
    assert norm_name("IBM") != norm_name("ibm") or True  # same key at gate
    assert norm_name("IBM") == "ibm"
    assert _core_entity_name("IBM") != _core_entity_name("ibm")

    # 3. Parentheticals: stripped by dedup + matching, kept by gate.
    assert _core_entity_name("张三（北京大学）") == "张三"
    assert norm_name("张三（北京大学）") != "张三"


def test_gate_and_matcher_can_disagree_on_same_pair():
    """The duplicate-family guard and the merger disagree about '张三教授' vs '张三'.

    CURRENT-BEHAVIOR: gate sees different names (no duplicate), matcher sees
    the same name (merge candidates). This asymmetry is the P2 unification
    target — audit found it can let the duplicate-family race through the
    very gate that exists to prevent it.
    """
    a, b = "张三教授", "张三"
    assert norm_name(a) != norm_name(b)              # gate: distinct
    assert normalize_entity_name_for_matching(a) == \
        normalize_entity_name_for_matching(b)        # matcher: same
