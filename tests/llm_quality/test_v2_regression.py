"""
V2 pipeline regression summary test.

Run this FIRST after any prompt/config change to get a quick quality score.
Aggregates key metrics:
- Average entity recall across representative inputs
- Alignment accuracy (TPR + FPR)
- No crashes on any input

Usage:
    USE_REAL_LLM=1 pytest tests/llm_quality/test_v2_regression.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# Allow `from conftest import ...` to find tests/llm_quality/conftest.py
sys.path.insert(0, os.path.dirname(__file__))

pytestmark = pytest.mark.real_llm

from tests.fixtures.extraction_gold import EXTRACTION_GOLD
from tests.fixtures.alignment_gold import SAME_ENTITY_PAIRS, DIFFERENT_ENTITY_PAIRS
from conftest import (
    _create_llm_client, _make_v2_processor,
    _run_v2_extraction, _count_matches,
)


# Representative subset for quick regression
REGRESSION_CASE_IDS = [
    "biography_mayun",
    "tech_go",
    "history_tang",
    "science_quantum",
    "english_long",
]


class TestRegressionSummary:
    """Quick regression check across all critical quality dimensions."""

    @pytest.fixture(scope="class")
    def processor(self, tmp_path_factory, real_llm_config, shared_embedding_client):
        if not real_llm_config:
            pytest.skip("No LLM config")
        tmp = tmp_path_factory.mktemp("regression")
        return _make_v2_processor(tmp, real_llm_config, shared_embedding_client)

    @pytest.fixture(scope="class")
    def llm_client(self, real_llm_config, shared_embedding_client):
        if not real_llm_config:
            pytest.skip("No LLM config")
        return _create_llm_client(real_llm_config, shared_embedding_client)

    def test_extraction_recall_summary(self, processor, request):
        """Run representative extractions and report aggregate recall."""
        results = {}
        for case_id in REGRESSION_CASE_IDS:
            case = EXTRACTION_GOLD[case_id]
            entities, _ = _run_v2_extraction(processor, case["text"])
            names = {e["name"] for e in entities}

            must_find = case["expected_entities"]["must_find"]
            matched, total, missing = _count_matches(must_find, names)
            recall = matched / total if total > 0 else 1.0
            results[case_id] = {
                "recall": recall,
                "matched": matched,
                "total": total,
                "missing": missing,
                "extracted_count": len(entities),
            }

        # Print summary for visibility
        print("\n=== Extraction Recall Summary ===")
        avg_recall = 0
        for cid, r in results.items():
            print(f"  {cid}: {r['recall']:.0%} ({r['matched']}/{r['total']}) "
                  f"[{r['extracted_count']} entities]")
            if r['missing']:
                print(f"    Missing: {r['missing']}")
            avg_recall += r["recall"]
        avg_recall /= len(results)
        print(f"  Average: {avg_recall:.0%}")

        # Individual case minimum: 60% (more lenient than full tests)
        for cid, r in results.items():
            assert r["recall"] >= 0.6, (
                f"[REGRESSION] {cid}: recall={r['recall']:.0%} (minimum 60%)"
            )

    def test_alignment_accuracy_summary(self, llm_client):
        """Run alignment judgment on key pairs and report accuracy."""
        # Test a subset of pairs for speed
        key_same = [p for p in SAME_ENTITY_PAIRS if p["difficulty"] in ("easy", "medium")][:8]
        key_diff = [p for p in DIFFERENT_ENTITY_PAIRS if p["difficulty"] in ("easy", "medium")][:8]

        correct = 0
        total = 0
        errors = []

        for case in key_same + key_diff:
            result = llm_client.judge_entity_same_v2(
                case["name_a"], case["content_a"],
                case["name_b"], case["content_b"],
            )
            expected = case["expected_same"]
            if result == expected:
                correct += 1
            else:
                errors.append(
                    f"  {case['id']}: expected={expected}, got={result} "
                    f"('{case['name_a']}' vs '{case['name_b']}')"
                )
            total += 1

        accuracy = correct / total if total > 0 else 0

        print(f"\n=== Alignment Accuracy Summary ===")
        print(f"  Accuracy: {accuracy:.0%} ({correct}/{total})")
        if errors:
            print(f"  Errors:")
            for e in errors:
                print(e)

        assert accuracy >= 0.85, (
            f"[REGRESSION] Alignment accuracy={accuracy:.0%} (minimum 85%). "
            f"Errors: {errors}"
        )

    def test_no_crash_on_all_inputs(self, processor):
        """All gold-standard inputs should process without crashing."""
        crash_list = []
        for case_id, case in EXTRACTION_GOLD.items():
            try:
                entities, relations = _run_v2_extraction(processor, case["text"])
                assert isinstance(entities, list), f"{case_id}: entities not a list"
                assert isinstance(relations, list), f"{case_id}: relations not a list"
            except Exception as e:
                crash_list.append(f"{case_id}: {e}")

        assert len(crash_list) == 0, (
            f"[REGRESSION] Crashes on: {crash_list}"
        )
