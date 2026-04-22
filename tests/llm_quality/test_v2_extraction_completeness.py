"""
V2 extraction completeness tests with real LLM.

Tests the full V2 extraction pipeline (_extract_only_v2):
  Step 2: Entity name extraction
  Step 3: Dedup
  Step 4: Content writing
  Step 5: Quality gate
  Step 6: Relation pair discovery
  Step 7: Relation content writing
  Step 8: Relation quality gate

Quality metrics:
- Entity recall: % of must_find entities extracted (target >= 80%)
- Relation recall: % of must_find relations extracted (target >= 60%)
- No system leaks in extracted names
- Entity density: entities per 100 chars (target >= 0.5 for dense texts)
"""

import pytest
import sys
import os

# Add project root for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# Allow `from conftest import ...` to find tests/llm_quality/conftest.py
sys.path.insert(0, os.path.dirname(__file__))

pytestmark = pytest.mark.real_llm

from tests.fixtures.extraction_gold import EXTRACTION_GOLD
from conftest import _make_v2_processor, _run_v2_extraction, _fuzzy_match, _count_matches

# All case IDs
ALL_CASE_IDS = list(EXTRACTION_GOLD.keys())

# Dense text cases (high entity density expected)
DENSE_CASE_IDS = [
    "tech_go", "history_tang", "science_quantum",
    "literature_honglou", "mixed_internet",
    "english_long", "ambiguous_names",
]

# Cases for relation testing
RELATION_CASE_IDS = [
    "biography_mayun", "tech_go", "history_tang",
    "science_quantum", "literature_honglou",
    "business_tesla", "mixed_internet",
]

# Sample cases for content quality checks
SAMPLE_CASE_IDS = [
    "biography_mayun", "tech_go", "history_tang",
    "english_long", "science_quantum",
]


class TestEntityRecall:
    """Entity extraction recall rate."""

    @pytest.fixture(scope="class")
    def processor(self, tmp_path_factory, real_llm_config, shared_embedding_client):
        if not real_llm_config:
            pytest.skip("No LLM config")
        tmp = tmp_path_factory.mktemp("v2_extraction")
        return _make_v2_processor(tmp, real_llm_config, shared_embedding_client)

    @pytest.mark.parametrize("case_id", ALL_CASE_IDS)
    def test_must_find_recall_ge_80(self, processor, case_id):
        """At least 80% of must_find entities should be extracted."""
        case = EXTRACTION_GOLD[case_id]
        entities, _ = _run_v2_extraction(processor, case["text"])
        extracted_names = {e["name"] for e in entities}

        must_find = case["expected_entities"]["must_find"]
        matched, total, missing = _count_matches(must_find, extracted_names)
        recall = matched / total if total > 0 else 1.0

        assert recall >= 0.8, (
            f"[{case_id}] Entity recall too low: {recall:.0%} ({matched}/{total}). "
            f"Missing: {missing}. Got: {sorted(extracted_names)}"
        )

    @pytest.mark.parametrize("case_id", ALL_CASE_IDS)
    def test_no_system_leaks(self, processor, case_id):
        """No system metadata entities should appear."""
        case = EXTRACTION_GOLD[case_id]
        entities, _ = _run_v2_extraction(processor, case["text"])

        for e in entities:
            for forbidden in case["expected_entities"]["must_not_find"]:
                assert forbidden.lower() not in e["name"].lower(), (
                    f"[{case_id}] System leak: '{e['name']}' contains '{forbidden}'"
                )

    @pytest.mark.parametrize("case_id", DENSE_CASE_IDS)
    def test_entity_density(self, processor, case_id):
        """Dense texts should produce at least 0.5 entities per 100 chars."""
        case = EXTRACTION_GOLD[case_id]
        entities, _ = _run_v2_extraction(processor, case["text"])
        density = len(entities) / max(1, len(case["text"]) / 100)
        assert density >= 0.5, (
            f"[{case_id}] Entity density too low: {density:.2f}/100chars "
            f"({len(entities)} entities from {len(case['text'])} chars)"
        )

    def test_long_text_more_entities(self, processor):
        """Longer texts should produce more entities than short texts."""
        short_case = EXTRACTION_GOLD["tech_k8s"]     # ~200 chars
        long_case = EXTRACTION_GOLD["tech_go"]        # ~2000 chars

        short_ents, _ = _run_v2_extraction(processor, short_case["text"])
        long_ents, _ = _run_v2_extraction(processor, long_case["text"])

        assert len(long_ents) > len(short_ents), (
            f"Long text ({len(long_ents)} ents) should have more than "
            f"short text ({len(short_ents)} ents)"
        )


class TestRelationRecall:
    """Relation extraction recall rate."""

    @pytest.fixture(scope="class")
    def processor(self, tmp_path_factory, real_llm_config, shared_embedding_client):
        if not real_llm_config:
            pytest.skip("No LLM config")
        tmp = tmp_path_factory.mktemp("v2_rel_extraction")
        return _make_v2_processor(tmp, real_llm_config, shared_embedding_client)

    @pytest.mark.parametrize("case_id", RELATION_CASE_IDS)
    def test_must_find_relations_ge_60(self, processor, case_id):
        """At least 60% of must_find relations should be discovered."""
        case = EXTRACTION_GOLD[case_id]
        entities, relations = _run_v2_extraction(processor, case["text"])

        # Build set of (entity1, entity2) pairs (both orders)
        found_pairs = set()
        for r in relations:
            e1 = r.get("entity1_name", "").lower()
            e2 = r.get("entity2_name", "").lower()
            found_pairs.add((e1, e2))
            found_pairs.add((e2, e1))

        must_find = case["expected_relations"]["must_find"]
        matched = 0
        missing = []
        for e1_hint, e2_hint in must_find:
            # Fuzzy match: check if any found pair contains both hints
            found = False
            for f1, f2 in found_pairs:
                if ((e1_hint.lower() in f1 or f1 in e1_hint.lower()) and
                    (e2_hint.lower() in f2 or f2 in e2_hint.lower())):
                    found = True
                    break
                # Also try reverse
                if ((e2_hint.lower() in f1 or f1 in e2_hint.lower()) and
                    (e1_hint.lower() in f2 or f2 in e1_hint.lower())):
                    found = True
                    break
            if found:
                matched += 1
            else:
                missing.append((e1_hint, e2_hint))

        total = len(must_find)
        recall = matched / total if total > 0 else 1.0
        assert recall >= 0.6, (
            f"[{case_id}] Relation recall too low: {recall:.0%} ({matched}/{total}). "
            f"Missing: {missing}"
        )


class TestContentQuality:
    """Entity content quality checks."""

    @pytest.fixture(scope="class")
    def processor(self, tmp_path_factory, real_llm_config, shared_embedding_client):
        if not real_llm_config:
            pytest.skip("No LLM config")
        tmp = tmp_path_factory.mktemp("v2_content_quality")
        return _make_v2_processor(tmp, real_llm_config, shared_embedding_client)

    @pytest.mark.parametrize("case_id", SAMPLE_CASE_IDS)
    def test_content_not_generic(self, processor, case_id):
        """Entity content should not contain template phrases."""
        case = EXTRACTION_GOLD[case_id]
        entities, _ = _run_v2_extraction(processor, case["text"])

        generic_patterns = ["该实体", "这是一个", "这是一"]
        for e in entities:
            content = e.get("content", "")
            for pattern in generic_patterns:
                assert pattern not in content, (
                    f"[{case_id}] Entity '{e['name']}' has generic content: '{content[:80]}'"
                )

    @pytest.mark.parametrize("case_id", SAMPLE_CASE_IDS)
    def test_content_min_length(self, processor, case_id):
        """Entity content should be at least 15 characters."""
        case = EXTRACTION_GOLD[case_id]
        entities, _ = _run_v2_extraction(processor, case["text"])

        for e in entities:
            content = e.get("content", "")
            assert len(content) >= 15, (
                f"[{case_id}] Entity '{e['name']}' content too short ({len(content)} chars): '{content}'"
            )
