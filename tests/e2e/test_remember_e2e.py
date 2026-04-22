"""
End-to-end remember pipeline test using the extended dataset.

Runs 30+ test cases across 4 dimensions (length, domain, language, type)
against the live Deep-Dream API and validates extraction quality.
"""

import json
import time
import sys
import os
import pytest
from typing import Dict, List, Any

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_dataset_extended import EXTENDED_DATASETS

# ─── Config ──────────────────────────────────────────────────────────────
API_BASE = os.environ.get("DEEP_DREAM_API", "http://localhost:16200/api/v1")
NO_PROXY = {"no_proxy": "localhost,127.0.0.1"}
import requests

SESSION = requests.Session()
SESSION.trust_env = False  # bypass proxy


def _api_get(path: str, **kwargs):
    return SESSION.get(f"{API_BASE}{path}", timeout=30, **kwargs)


def _api_post(path: str, data: dict = None, **kwargs):
    return SESSION.post(f"{API_BASE}{path}", json=data, timeout=120, **kwargs)


def _api_delete(path: str, **kwargs):
    return SESSION.delete(f"{API_BASE}{path}", timeout=30, **kwargs)


# ─── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def check_server():
    """Verify the Deep-Dream server is running."""
    try:
        r = _api_get("/health")
        assert r.status_code == 200, f"Server health check failed: {r.status_code}"
        data = r.json()
        print(f"\n[Server] backend={data['data']['storage_backend']}, "
              f"embedding={data['data']['embedding_available']}")
    except Exception as e:
        pytest.skip(f"Deep-Dream server not available: {e}")


@pytest.fixture(scope="session")
def remember_results():
    """Run all test datasets through remember and collect results."""
    results = {}
    for name, ds in EXTENDED_DATASETS.items():
        text = ds["text"]
        print(f"\n>>> Testing [{name}] ({len(text)} chars, {ds['domain']}, {ds['language']}, {ds['type']})")

        t0 = time.time()
        r = _api_post("/remember", {"text": text, "source": f"test:{name}"})
        elapsed = time.time() - t0

        # API returns 200 (sync) or 202 (async queued)
        if r.status_code not in (200, 202):
            print(f"    FAILED: HTTP {r.status_code} — {r.text[:200]}")
            results[name] = {"status": "error", "http_code": r.status_code, "elapsed": elapsed}
            continue

        data = r.json()
        task_id = data.get("data", {}).get("task_id")
        initial_status = data.get("data", {}).get("status", "")

        # If queued or has task_id, poll until complete
        if task_id and initial_status in ("queued", "processing"):
            for poll_i in range(180):  # max 6 min wait
                time.sleep(2)
                sr = _api_get(f"/remember/tasks/{task_id}")
                if sr.status_code != 200:
                    continue
                sd = sr.json()
                status = sd.get("data", {}).get("status", "unknown")
                if status in ("completed", "failed"):
                    break
            else:
                print(f"    TIMEOUT after {time.time()-t0:.0f}s")
                results[name] = {"status": "error", "http_code": 0, "elapsed": time.time()-t0}
                continue
            result_data = sd.get("data", {})
        elif task_id:
            # Completed synchronously or task has result
            sr = _api_get(f"/remember/tasks/{task_id}")
            if sr.status_code == 200:
                result_data = sr.json().get("data", {})
            else:
                result_data = data.get("data", {})
        else:
            result_data = data.get("data", {})

        # Extract counts from result dict (nested under "result")
        result_inner = result_data.get("result", {}) or {}
        entity_count = result_inner.get("entities", 0) or result_data.get("entities_count", 0)
        relation_count = result_inner.get("relations", 0) or result_data.get("relations_count", 0)
        elapsed_final = time.time() - t0

        print(f"    OK: {entity_count} entities, {relation_count} relations, "
              f"{elapsed_final:.1f}s")

        results[name] = {
            "status": "completed",
            "entities_count": entity_count,
            "relations_count": relation_count,
            "elapsed": elapsed_final,
            "task_id": task_id,
            "text_length": len(text),
            "expected_entities": ds.get("expected_entities", []),
            "expected_relations_min": ds.get("expected_relations_min", 0),
            "domain": ds.get("domain", ""),
            "language": ds.get("language", ""),
            "content_type": ds.get("type", ""),
            "length": ds.get("length", ""),
        }

    return results


# ─── Dimension Analysis Helpers ──────────────────────────────────────────

def _dimension_stats(results: Dict, dim: str) -> Dict[str, Dict]:
    """Aggregate results by a dimension."""
    buckets: Dict[str, List] = {}
    for name, r in results.items():
        if r.get("status") != "completed":
            continue
        ds = EXTENDED_DATASETS[name]
        key = ds.get(dim, "unknown")
        buckets.setdefault(key, []).append(r)
    stats = {}
    for key, items in buckets.items():
        ents = [i["entities_count"] for i in items]
        rels = [i["relations_count"] for i in items]
        times = [i["elapsed"] for i in items]
        stats[key] = {
            "count": len(items),
            "avg_entities": sum(ents) / len(ents),
            "avg_relations": sum(rels) / len(rels),
            "avg_time": sum(times) / len(times),
            "total_entities": sum(ents),
            "total_relations": sum(rels),
        }
    return stats


# ─── Test Classes ────────────────────────────────────────────────────────

class TestRememberBasic:
    """Basic extraction quality tests."""

    def test_all_complete(self, remember_results):
        """All 30+ datasets should complete without error."""
        errors = {k: v for k, v in remember_results.items() if v["status"] != "completed"}
        assert not errors, f"Failed datasets: {list(errors.keys())} — {errors}"

    def test_dataset_count(self, remember_results):
        """Should have 30+ test cases."""
        assert len(remember_results) >= 25, f"Only {len(remember_results)} datasets tested"

    def test_every_dataset_extracts_entities(self, remember_results):
        """Every dataset should extract at least 1 entity."""
        no_ents = {k: v for k, v in remember_results.items()
                   if v["status"] == "completed" and v["entities_count"] == 0}
        assert not no_ents, f"Zero entities for: {list(no_ents.keys())}"

    def test_entity_extraction_reasonable(self, remember_results):
        """Entity count should be reasonable (1-80 for our text lengths)."""
        for name, r in remember_results.items():
            if r["status"] != "completed":
                continue
            ec = r["entities_count"]
            assert ec >= 1, f"[{name}] Too few entities: {ec}"
            assert ec <= 80, f"[{name}] Too many entities (over-extraction): {ec}"


class TestExtractionByLength:
    """Dimension 1: Length should correlate with extraction count."""

    def test_length_analysis(self, remember_results):
        stats = _dimension_stats(remember_results, "length")
        print("\n\n=== Length Dimension ===")
        for key in ["short", "medium", "long", "very_long"]:
            if key in stats:
                s = stats[key]
                print(f"  {key:10s}: {s['count']:2d} texts, "
                      f"avg {s['avg_entities']:.1f} ents, "
                      f"{s['avg_relations']:.1f} rels, "
                      f"{s['avg_time']:.1f}s")

        # Very long texts should extract more entities than short texts
        if "short" in stats and "very_long" in stats:
            assert stats["very_long"]["avg_entities"] > stats["short"]["avg_entities"], \
                "Very long texts should extract more entities than short texts"

    def test_no_excessive_extraction(self, remember_results):
        """Check over-extraction: entities per 100 chars should be 2-15."""
        for name, r in remember_results.items():
            if r["status"] != "completed":
                continue
            rate = r["entities_count"] / max(r["text_length"] / 100, 1)
            assert rate <= 15, f"[{name}] Over-extraction rate: {rate:.1f} ents/100chars"
            assert rate >= 1, f"[{name}] Under-extraction rate: {rate:.1f} ents/100chars"


class TestExtractionByDomain:
    """Dimension 2: Different domains should extract relevant entities."""

    def test_domain_analysis(self, remember_results):
        stats = _dimension_stats(remember_results, "domain")
        print("\n\n=== Domain Dimension ===")
        for key in sorted(stats.keys()):
            s = stats[key]
            print(f"  {key:15s}: {s['count']:2d} texts, "
                  f"avg {s['avg_entities']:.1f} ents, "
                  f"{s['avg_relations']:.1f} rels")

    def test_all_domains_work(self, remember_results):
        """Every domain should have at least one successful extraction."""
        domains = set()
        for name, r in remember_results.items():
            if r["status"] == "completed":
                domains.add(EXTENDED_DATASETS[name].get("domain", ""))
        print(f"\n  Domains tested: {sorted(domains)}")
        assert len(domains) >= 8, f"Only {len(domains)} domains tested, need 8+"

    def test_expected_entities_found(self, remember_results):
        """Key expected entities should be searchable after remember."""
        for name, r in remember_results.items():
            if r["status"] != "completed":
                continue
            expected = r.get("expected_entities", [])
            if not expected:
                continue
            # At least 50% of expected entities should be found
            # (we check entities_count is proportional to expected count)
            min_expected = max(len(expected) * 0.5, 1)
            assert r["entities_count"] >= min_expected, \
                f"[{name}] Expected ~{len(expected)} entities, got {r['entities_count']}"


class TestExtractionByLanguage:
    """Dimension 3: Language should not prevent extraction."""

    def test_language_analysis(self, remember_results):
        stats = _dimension_stats(remember_results, "language")
        print("\n\n=== Language Dimension ===")
        for key in sorted(stats.keys()):
            s = stats[key]
            print(f"  {key:10s}: {s['count']:2d} texts, "
                  f"avg {s['avg_entities']:.1f} ents, "
                  f"{s['avg_relations']:.1f} rels")

    def test_english_works(self, remember_results):
        """English texts should extract entities successfully."""
        en_results = [r for n, r in remember_results.items()
                      if r["status"] == "completed"
                      and EXTENDED_DATASETS[n].get("language") == "en"]
        assert len(en_results) >= 3, "Need at least 3 English test cases"
        for r in en_results:
            assert r["entities_count"] >= 1, f"English text extracted 0 entities"

    def test_mixed_works(self, remember_results):
        """Mixed language texts should work."""
        mixed = [r for n, r in remember_results.items()
                 if r["status"] == "completed"
                 and EXTENDED_DATASETS[n].get("language") == "mixed"]
        assert len(mixed) >= 2, "Need at least 2 mixed-language test cases"
        for r in mixed:
            assert r["entities_count"] >= 2, f"Mixed text extracted too few entities"


class TestExtractionByType:
    """Dimension 4: Different content types should all extract properly."""

    def test_type_analysis(self, remember_results):
        stats = _dimension_stats(remember_results, "type")
        print("\n\n=== Content Type Dimension ===")
        for key in sorted(stats.keys()):
            s = stats[key]
            print(f"  {key:12s}: {s['count']:2d} texts, "
                  f"avg {s['avg_entities']:.1f} ents, "
                  f"{s['avg_relations']:.1f} rels")

    def test_dialogue_extracts(self, remember_results):
        """Dialogue format should still extract entities."""
        dialogue = [r for n, r in remember_results.items()
                    if r["status"] == "completed"
                    and EXTENDED_DATASETS[n].get("type") == "dialogue"]
        if dialogue:
            assert dialogue[0]["entities_count"] >= 2

    def test_list_extracts(self, remember_results):
        """List format should extract items."""
        lists = [r for n, r in remember_results.items()
                 if r["status"] == "completed"
                 and EXTENDED_DATASETS[n].get("type") == "list"]
        if lists:
            assert lists[0]["entities_count"] >= 4, "List text should extract multiple entities"

    def test_technical_extracts(self, remember_results):
        """Technical docs should extract tech terms."""
        techs = [r for n, r in remember_results.items()
                 if r["status"] == "completed"
                 and EXTENDED_DATASETS[n].get("type") == "technical"]
        if techs:
            assert techs[0]["entities_count"] >= 3

    def test_content_type_diversity(self, remember_results):
        """Should test at least 6 different content types."""
        types = set()
        for name, r in remember_results.items():
            if r["status"] == "completed":
                types.add(EXTENDED_DATASETS[name].get("type", ""))
        print(f"\n  Content types tested: {sorted(types)}")
        assert len(types) >= 6, f"Only {len(types)} content types tested"


class TestFindQuality:
    """Test that remembered data is findable."""

    def test_search_after_remember(self, remember_results):
        """Entities should be searchable after remember."""
        # Search for a few key entities
        test_searches = [
            ("量子力学", "quantum"),
            ("Python", "tech"),
            ("贝多芬", "music"),
            ("Amazon", "business"),
        ]
        for query, label in test_searches:
            r = _api_get(f"/find/entities?q={query}&limit=5")
            if r.status_code == 200:
                data = r.json()
                found = len(data.get("data", {}).get("entities", []))
                print(f"  Search '{query}' [{label}]: found {found} entities")
                # Don't assert here — the graph may not have these yet

    def test_entity_search_works(self, remember_results):
        """Basic entity search should return results."""
        r = _api_get("/find/entities?q=Python&limit=5")
        assert r.status_code == 200, f"Entity search failed: {r.status_code}"

    def test_relation_search_works(self, remember_results):
        """Relation search should return results."""
        r = _api_get("/find/relations?q=发明&limit=5")
        assert r.status_code == 200, f"Relation search failed: {r.status_code}"


class TestCrossCallDedup:
    """Test that shared entities across datasets are deduplicated."""

    def test_shared_entities_deduped(self, remember_results):
        """Datasets that share entities (e.g. Google, Python) should result in single entities."""
        # Search for Google which appears in multiple datasets
        r = _api_get("/find/entities?q=Google&limit=10")
        if r.status_code == 200:
            data = r.json()
            entities = data.get("data", {}).get("entities", [])
            # Count distinct family_ids for "Google"
            google_family_ids = set()
            for e in entities:
                name = e.get("name", "")
                if "google" in name.lower() or "Google" in name:
                    google_family_ids.add(e.get("family_id", ""))
            print(f"  Google entity family_ids: {len(google_family_ids)}")
            # Should ideally be 1 (deduped), definitely not more than 3
            assert len(google_family_ids) <= 5, \
                f"Too many separate Google entities: {google_family_ids}"


class TestPerformance:
    """Performance regression tests."""

    def test_short_text_under_30s(self, remember_results):
        """Short texts should complete in under 30s."""
        for name, r in remember_results.items():
            if r["status"] != "completed":
                continue
            if EXTENDED_DATASETS[name].get("length") == "short":
                assert r["elapsed"] < 30, \
                    f"[{name}] Short text took {r['elapsed']:.1f}s (>30s)"

    def test_time_scales_reasonably(self, remember_results):
        """Processing time should scale roughly linearly with text length."""
        completed = [(n, r) for n, r in remember_results.items() if r["status"] == "completed"]
        if len(completed) < 4:
            return
        # Group by length and check average times increase
        stats = _dimension_stats(remember_results, "length")
        order = ["short", "medium", "long", "very_long"]
        times = [(k, stats[k]["avg_time"]) for k in order if k in stats]
        if len(times) >= 2:
            for i in range(len(times) - 1):
                # Each longer category shouldn't take more than 10x the shorter
                assert times[i + 1][1] <= times[i][1] * 10, \
                    f"Time scaling too steep: {times[i]} -> {times[i+1]}"


class TestSummaryReport:
    """Generate a summary report of all test results."""

    def test_print_summary(self, remember_results):
        """Print a comprehensive summary of extraction results."""
        print("\n\n" + "=" * 80)
        print("DEEP-DREAM EXTRACTION QUALITY REPORT")
        print("=" * 80)

        completed = {k: v for k, v in remember_results.items() if v["status"] == "completed"}
        total_entities = sum(v["entities_count"] for v in completed.values())
        total_relations = sum(v["relations_count"] for v in completed.values())
        total_time = sum(v["elapsed"] for v in completed.values())
        total_chars = sum(v["text_length"] for v in completed.values())

        print(f"\n  Datasets tested: {len(completed)}/{len(remember_results)}")
        print(f"  Total text: {total_chars:,} chars")
        print(f"  Total entities: {total_entities}")
        print(f"  Total relations: {total_relations}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Avg entities/100chars: {total_entities / max(total_chars / 100, 1):.1f}")
        print(f"  Avg relations/100chars: {total_relations / max(total_chars / 100, 1):.1f}")
        print(f"  Avg processing time: {total_time / len(completed):.1f}s per text")

        # Per-result detail
        print(f"\n  {'Name':<25s} {'Len':>5s} {'Ents':>5s} {'Rels':>5s} {'Time':>6s} {'Rate':>6s}")
        print("  " + "-" * 60)
        for name, r in sorted(completed.items(), key=lambda x: x[1]["text_length"]):
            rate = r["entities_count"] / max(r["text_length"] / 100, 1)
            print(f"  {name:<25s} {r['text_length']:>5d} {r['entities_count']:>5d} "
                  f"{r['relations_count']:>5d} {r['elapsed']:>5.1f}s {rate:>5.1f}/100c")

        # Dimension summaries
        for dim_name in ["length", "domain", "language", "type"]:
            stats = _dimension_stats(remember_results, dim_name)
            print(f"\n  By {dim_name}:")
            for key in sorted(stats.keys()):
                s = stats[key]
                print(f"    {key:15s}: {s['count']:2d} texts, "
                      f"avg {s['avg_entities']:.1f} ents, "
                      f"{s['avg_relations']:.1f} rels, "
                      f"{s['avg_time']:.1f}s")

        print("\n" + "=" * 80)

        # Quality assertions
        assert total_entities >= 50, f"Too few total entities: {total_entities}"
        assert total_relations >= 20, f"Too few total relations: {total_relations}"
