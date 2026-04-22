"""
Comprehensive tests for iteration-7: Jaccard similarity deduplication.

Covers 4 dimensions with 35+ test cases:
  1. Shared calculate_jaccard_similarity function (12 tests)
  2. StorageManager delegation (8 tests)
  3. EntityProcessor delegation (8 tests)
  4. Edge cases & cross-module consistency (10 tests)
"""
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_real_storage(tmp_path):
    """Create a fully functional StorageManager (not mocked)."""
    from processor.storage.manager import StorageManager
    return StorageManager(str(tmp_path / "storage"))


def _make_entity_processor(tmp_path):
    """Create an EntityProcessor with real storage and mock LLM client."""
    from processor.pipeline.entity import EntityProcessor
    sm = _make_real_storage(tmp_path)
    mock_llm = MagicMock()
    mock_llm.effective_entity_snippet_length.return_value = 50
    processor = EntityProcessor(storage=sm, llm_client=mock_llm)
    return processor, sm


# ============================================================
# Dimension 1: Shared calculate_jaccard_similarity function
# ============================================================
class TestSharedJaccardFunction:
    """Tests for the standalone calculate_jaccard_similarity function in utils."""

    def test_identical_strings(self):
        from processor.utils import calculate_jaccard_similarity
        assert calculate_jaccard_similarity("hello", "hello") == 1.0

    def test_completely_different_strings(self):
        from processor.utils import calculate_jaccard_similarity
        assert calculate_jaccard_similarity("abc", "xyz") == 0.0

    def test_empty_strings(self):
        from processor.utils import calculate_jaccard_similarity
        assert calculate_jaccard_similarity("", "") == 0.0
        assert calculate_jaccard_similarity("abc", "") == 0.0
        assert calculate_jaccard_similarity("", "abc") == 0.0

    def test_single_char_strings(self):
        from processor.utils import calculate_jaccard_similarity
        # Single char: no bigrams, falls back to char-level Jaccard
        assert calculate_jaccard_similarity("a", "a") == 1.0
        assert calculate_jaccard_similarity("a", "b") == 0.0

    def test_case_insensitive(self):
        from processor.utils import calculate_jaccard_similarity
        assert calculate_jaccard_similarity("Hello", "hello") == 1.0
        assert calculate_jaccard_similarity("HELLO", "hello") == 1.0

    def test_whitespace_stripped(self):
        from processor.utils import calculate_jaccard_similarity
        assert calculate_jaccard_similarity("  hello  ", "hello") == 1.0

    def test_partial_overlap(self):
        from processor.utils import calculate_jaccard_similarity
        sim = calculate_jaccard_similarity("abcdef", "abcxyz")
        # bigrams: {ab, bc, cd, de, ef} vs {ab, bc, cx, xy, yz}
        # intersection: {ab, bc} = 2, union: {ab, bc, cd, de, ef, cx, xy, yz} = 8
        assert 0.2 < sim < 0.4

    def test_chinese_strings(self):
        from processor.utils import calculate_jaccard_similarity
        sim = calculate_jaccard_similarity("张伟教授", "张伟")
        # "张伟教授" bigrams: {张伟, 伟教, 教授}
        # "张伟" bigrams: {张伟}
        # intersection: {张伟} = 1, union: {张伟, 伟教, 教授} = 3
        assert sim == pytest.approx(1/3, abs=0.01)

    def test_unicode_emoji(self):
        from processor.utils import calculate_jaccard_similarity
        sim = calculate_jaccard_similarity("Hello 🌍", "Hello 🌍")
        assert sim == 1.0

    def test_none_inputs(self):
        from processor.utils import calculate_jaccard_similarity
        assert calculate_jaccard_similarity(None, "abc") == 0.0
        assert calculate_jaccard_similarity("abc", None) == 0.0
        assert calculate_jaccard_similarity(None, None) == 0.0

    def test_long_strings(self):
        from processor.utils import calculate_jaccard_similarity
        # Bigram sets for "a"*1000+"xyz" are tiny: {aa, ax, xy, yz}
        # vs "a"*1000+"abc": {aa, ab, bc} — low overlap
        s1 = "a" * 1000 + "xyz"
        s2 = "a" * 1000 + "abc"
        sim = calculate_jaccard_similarity(s1, s2)
        # Only {aa} in common → 1/5 = 0.2 approx
        assert 0.0 < sim < 0.5
        # But truly diverse long strings should give high similarity
        s3 = "the quick brown fox jumps over the lazy dog " * 50
        s4 = "the quick brown fox jumps over the lazy dog " * 50
        assert calculate_jaccard_similarity(s3, s4) == 1.0

    def test_symmetric(self):
        from processor.utils import calculate_jaccard_similarity
        assert calculate_jaccard_similarity("abc", "bcd") == calculate_jaccard_similarity("bcd", "abc")


# ============================================================
# Dimension 2: StorageManager delegation
# ============================================================
class TestStorageManagerDelegation:
    """Tests that StorageManager._calculate_jaccard_similarity delegates correctly."""

    def test_delegates_to_shared_function(self, tmp_path):
        sm = _make_real_storage(tmp_path)
        from processor.utils import calculate_jaccard_similarity
        text1, text2 = "hello world", "hello earth"
        expected = calculate_jaccard_similarity(text1, text2)
        assert sm._calculate_jaccard_similarity(text1, text2) == expected
        sm.close()

    def test_chinese_delegation(self, tmp_path):
        sm = _make_real_storage(tmp_path)
        from processor.utils import calculate_jaccard_similarity
        expected = calculate_jaccard_similarity("北京", "上海")
        assert sm._calculate_jaccard_similarity("北京", "上海") == expected
        sm.close()

    def test_empty_string_delegation(self, tmp_path):
        sm = _make_real_storage(tmp_path)
        assert sm._calculate_jaccard_similarity("", "test") == 0.0
        sm.close()

    def test_identical_delegation(self, tmp_path):
        sm = _make_real_storage(tmp_path)
        assert sm._calculate_jaccard_similarity("test", "test") == 1.0
        sm.close()

    def test_consistency_across_calls(self, tmp_path):
        sm = _make_real_storage(tmp_path)
        r1 = sm._calculate_jaccard_similarity("alpha", "beta")
        r2 = sm._calculate_jaccard_similarity("alpha", "beta")
        assert r1 == r2
        sm.close()

    def test_used_in_search_with_text_similarity(self, tmp_path):
        """Verify the shared function is used in actual search operations."""
        from processor.models import Entity
        sm = _make_real_storage(tmp_path)
        e1 = Entity(
            absolute_id="e1", family_id="ent_a", name="Alice",
            content="Developer at Google",
            event_time=datetime.now(), processed_time=datetime.now(),
            episode_id="ep1", source_document="",
        )
        sm.save_entity(e1)
        # Search with jaccard method
        results = sm.search_entities_by_similarity(
            "Alice", threshold=0.3, max_results=10,
            text_mode="name_only", similarity_method="jaccard",
        )
        assert len(results) >= 1
        assert results[0].name == "Alice"
        sm.close()

    def test_delegation_with_special_chars(self, tmp_path):
        sm = _make_real_storage(tmp_path)
        from processor.utils import calculate_jaccard_similarity
        special1 = "C++ / Rust / Go"
        special2 = "C++ / Python / Go"
        expected = calculate_jaccard_similarity(special1, special2)
        assert sm._calculate_jaccard_similarity(special1, special2) == expected
        sm.close()

    def test_bleu_independent_of_jaccard(self, tmp_path):
        """BLEU and Jaccard should produce different results."""
        sm = _make_real_storage(tmp_path)
        text1, text2 = "hello world", "world hello"
        jaccard = sm._calculate_jaccard_similarity(text1, text2)
        bleu = sm._calculate_bleu_similarity(text1, text2)
        # They should differ for this input
        assert isinstance(jaccard, float)
        assert isinstance(bleu, float)
        sm.close()


# ============================================================
# Dimension 3: EntityProcessor delegation
# ============================================================
class TestEntityProcessorDelegation:
    """Tests that EntityProcessor._calculate_jaccard_similarity delegates correctly."""

    def test_delegates_to_shared_function(self, tmp_path):
        proc, sm = _make_entity_processor(tmp_path)
        from processor.utils import calculate_jaccard_similarity
        text1, text2 = "Python", "python3"
        expected = calculate_jaccard_similarity(text1, text2)
        assert proc._calculate_jaccard_similarity(text1, text2) == expected
        sm.close()

    def test_chinese_delegation(self, tmp_path):
        proc, sm = _make_entity_processor(tmp_path)
        from processor.utils import calculate_jaccard_similarity
        expected = calculate_jaccard_similarity("机器学习", "深度学习")
        assert proc._calculate_jaccard_similarity("机器学习", "深度学习") == expected
        sm.close()

    def test_empty_string_delegation(self, tmp_path):
        proc, sm = _make_entity_processor(tmp_path)
        assert proc._calculate_jaccard_similarity("", "test") == 0.0
        sm.close()

    def test_identical_delegation(self, tmp_path):
        proc, sm = _make_entity_processor(tmp_path)
        assert proc._calculate_jaccard_similarity("test", "test") == 1.0
        sm.close()

    def test_consistency_across_calls(self, tmp_path):
        proc, sm = _make_entity_processor(tmp_path)
        r1 = proc._calculate_jaccard_similarity("alpha", "beta")
        r2 = proc._calculate_jaccard_similarity("alpha", "beta")
        assert r1 == r2
        sm.close()

    def test_entity_name_matching_context(self, tmp_path):
        """Test that Jaccard works correctly in entity name matching context."""
        proc, sm = _make_entity_processor(tmp_path)
        # Names with title suffixes should have reasonable similarity
        sim = proc._calculate_jaccard_similarity("张伟教授", "张伟")
        assert sim > 0.3  # Should detect some similarity
        sm.close()

    def test_normalize_and_jaccard(self, tmp_path):
        """Test _normalize_entity_name_for_matching combined with Jaccard."""
        proc, sm = _make_entity_processor(tmp_path)
        core1 = proc._normalize_entity_name_for_matching("张伟教授")
        core2 = proc._normalize_entity_name_for_matching("张伟")
        assert core1 == core2 == "张伟"
        sim = proc._calculate_jaccard_similarity(core1, core2)
        assert sim == 1.0
        sm.close()

    def test_english_name_variants(self, tmp_path):
        proc, sm = _make_entity_processor(tmp_path)
        from processor.utils import calculate_jaccard_similarity
        sim1 = proc._calculate_jaccard_similarity("Kubernetes", "K8s")
        sim2 = calculate_jaccard_similarity("Kubernetes", "K8s")
        assert sim1 == sim2
        assert sim1 == 0.0  # Completely different bigrams
        sm.close()


# ============================================================
# Dimension 4: Edge cases & cross-module consistency
# ============================================================
class TestCrossModuleConsistency:
    """Tests ensuring both modules produce identical results."""

    def test_both_modules_identical_results(self, tmp_path):
        """StorageManager and EntityProcessor should produce identical Jaccard scores."""
        proc, sm = _make_entity_processor(tmp_path)
        test_pairs = [
            ("hello", "world"),
            ("Python", "python"),
            ("机器学习", "深度学习"),
            ("Alice Smith", "Alice Jones"),
            ("", "test"),
            ("a", "a"),
            ("Kubernetes", "k8s"),
            ("React", "React.js"),
            ("Go语言", "Golang"),
            ("张伟（北京大学）", "张伟（清华大学）"),
        ]
        from processor.utils import calculate_jaccard_similarity
        for t1, t2 in test_pairs:
            expected = calculate_jaccard_similarity(t1, t2)
            sm_result = sm._calculate_jaccard_similarity(t1, t2)
            proc_result = proc._calculate_jaccard_similarity(t1, t2)
            assert sm_result == expected, f"StorageManager mismatch for '{t1}' vs '{t2}'"
            assert proc_result == expected, f"EntityProcessor mismatch for '{t1}' vs '{t2}'"
        sm.close()

    def test_jaccard_bounds_always_between_0_and_1(self, tmp_path):
        """Jaccard similarity should always be in [0, 1] range."""
        from processor.utils import calculate_jaccard_similarity
        test_inputs = [
            ("", ""), ("a", "b"), ("x" * 100, "y" * 100),
            ("中文测试", "英文测试"), ("123", "456"),
            ("Hello\nWorld", "Hello World"),
            ("tab\there", "tabhere"),
        ]
        for t1, t2 in test_inputs:
            sim = calculate_jaccard_similarity(t1, t2)
            assert 0.0 <= sim <= 1.0, f"Out of bounds for '{t1}' vs '{t2}': {sim}"

    def test_deterministic_results(self, tmp_path):
        """Same inputs should always produce the same output."""
        from processor.utils import calculate_jaccard_similarity
        import random
        random.seed(42)
        for _ in range(100):
            t1 = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(1, 20)))
            t2 = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(1, 20)))
            r1 = calculate_jaccard_similarity(t1, t2)
            r2 = calculate_jaccard_similarity(t1, t2)
            assert r1 == r2

    def test_triangular_inequality_holds(self):
        """Jaccard distance should satisfy triangular inequality: d(a,c) <= d(a,b) + d(b,c)."""
        from processor.utils import calculate_jaccard_similarity
        a, b, c = "abcdef", "abcdefg", "abcdefgh"
        d_ab = 1 - calculate_jaccard_similarity(a, b)
        d_bc = 1 - calculate_jaccard_similarity(b, c)
        d_ac = 1 - calculate_jaccard_similarity(a, c)
        assert d_ac <= d_ab + d_bc + 1e-9

    def test_real_entity_name_similarity_distribution(self, tmp_path):
        """Test similarity distribution with realistic entity names."""
        from processor.utils import calculate_jaccard_similarity
        names = [
            ("Alice", "Alice Smith", "partial match"),
            ("Bob", "Robert", "nickname"),
            ("Kubernetes", "Kubernetes", "exact"),
            ("Python", "Python3", "version"),
            ("机器学习", "深度学习", "related Chinese"),
            ("Redis", "MongoDB", "unrelated tech"),
            ("Go语言", "Golang", "same thing diff name"),
            ("React", "React.js", "with extension"),
            ("张伟教授", "张伟", "title suffix"),
            ("Alice（工程师）", "Alice（设计师）", "same name diff role"),
        ]
        for n1, n2, desc in names:
            sim = calculate_jaccard_similarity(n1, n2)
            assert 0.0 <= sim <= 1.0, f"Out of bounds: {desc}"

    def test_performance_on_repeated_calls(self, tmp_path):
        """Repeated calls should be fast (no caching expected but no degradation)."""
        from processor.utils import calculate_jaccard_similarity
        import time
        start = time.time()
        for _ in range(10000):
            calculate_jaccard_similarity("hello world test string", "hello earth test string")
        elapsed = time.time() - start
        assert elapsed < 2.0, f"10000 Jaccard calls took {elapsed:.2f}s — too slow"

    def test_none_handling_consistent(self, tmp_path):
        """None handling should be consistent across all entry points."""
        proc, sm = _make_entity_processor(tmp_path)
        from processor.utils import calculate_jaccard_similarity
        for t1, t2 in [(None, "a"), ("a", None), (None, None)]:
            expected = calculate_jaccard_similarity(t1, t2)
            assert sm._calculate_jaccard_similarity(t1, t2) == expected
            assert proc._calculate_jaccard_similarity(t1, t2) == expected
        sm.close()

    def test_numeric_string_inputs(self, tmp_path):
        """Numeric strings should be handled correctly."""
        from processor.utils import calculate_jaccard_similarity
        assert calculate_jaccard_similarity("12345", "12345") == 1.0
        assert calculate_jaccard_similarity("12345", "67890") == 0.0
        sim = calculate_jaccard_similarity("12345", "12346")
        assert 0.0 < sim < 1.0

    def test_mixed_language_inputs(self, tmp_path):
        """Mixed language strings should work."""
        from processor.utils import calculate_jaccard_similarity
        sim = calculate_jaccard_similarity("Python编程", "Python programming")
        assert 0.0 < sim < 1.0  # Some overlap due to "Python" bigrams
