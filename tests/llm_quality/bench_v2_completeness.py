#!/usr/bin/env python3
"""
Benchmark V2 full-pipeline extraction completeness.

Runs _run_v2_extraction on all EXTRACTION_GOLD cases and reports:
- Entity recall (must_find >= 80%, should_find >= 50%)
- Relation recall (must_find >= 60%)
- Entity density
- Timing per case

Usage:
    cd /home/linkco/exa/Deep-Dream
    CUDA_VISIBLE_DEVICES=0 python tests/llm_quality/bench_v2_completeness.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

from pathlib import Path
from datetime import datetime

from tests.fixtures.extraction_gold import EXTRACTION_GOLD
from tests.llm_quality.conftest import (
    _make_v2_processor,
    _run_v2_extraction,
    _fuzzy_match,
    _count_matches,
)


def _run_extraction(processor, text):
    """Run extraction using the processor's configured mode (v2/v3/legacy)."""
    from processor.models import Episode
    from datetime import datetime
    episode = Episode(
        absolute_id="test_ep",
        content=text,
        event_time=datetime.now(),
        source_document="test_doc",
        activity_type="文档处理",
    )
    return processor._extract_only(
        episode, text, "test_doc",
        verbose=False, verbose_steps=False,
    )
from processor.storage.embedding import EmbeddingClient


def load_config():
    cfg_path = Path(__file__).resolve().parent.parent.parent / "service_config.json"
    with open(cfg_path) as f:
        return json.load(f)


def make_processor(cfg, tmp_path):
    from server.config import merge_llm_extraction
    llm = cfg["llm"]
    emb_cfg = cfg.get("embedding", {})
    emb_client = EmbeddingClient(
        model_path=emb_cfg.get("model"),
        device=emb_cfg.get("device", "cpu"),
        use_local=True,
    )
    real_llm_config = {
        "llm_api_key": llm["api_key"],
        "llm_model": llm["model"],
        "llm_base_url": llm["base_url"],
        "llm_think_mode": llm.get("think", False),
        "llm_max_tokens": llm.get("max_tokens", 4000),
        "llm_context_window_tokens": llm.get("context_window_tokens", 7500),
    }
    pipeline_ext = cfg.get("pipeline", {}).get("extraction", {})
    entity_rounds = pipeline_ext.get("entity_extraction_rounds") or pipeline_ext.get("extraction_rounds")
    overrides = {}
    if entity_rounds:
        overrides["entity_extraction_rounds"] = entity_rounds
    for flag in ("v2_enable_reflection", "v2_enable_orphan_recovery"):
        if flag in pipeline_ext:
            overrides[flag] = pipeline_ext[flag]
    # V3 dual-model: pass extraction_llm config
    extraction_llm = merge_llm_extraction(llm)
    if extraction_llm.get("enabled"):
        overrides["extraction_llm"] = extraction_llm
        overrides.setdefault("remember_config", {})["mode"] = "v3"
    return _make_v2_processor(tmp_path, real_llm_config, emb_client, **overrides)


def check_relation_match(must_find_pairs, relations):
    """Check how many expected relations are found."""
    found_pairs = set()
    for r in relations:
        e1 = r.get("entity1_name", "").lower()
        e2 = r.get("entity2_name", "").lower()
        found_pairs.add((e1, e2))
        found_pairs.add((e2, e1))

    matched = 0
    missing = []
    for e1_hint, e2_hint in must_find_pairs:
        found = False
        for f1, f2 in found_pairs:
            if ((e1_hint.lower() in f1 or f1 in e1_hint.lower()) and
                (e2_hint.lower() in f2 or f2 in e2_hint.lower())):
                found = True
                break
            if ((e2_hint.lower() in f1 or f1 in e2_hint.lower()) and
                (e1_hint.lower() in f2 or f2 in e1_hint.lower())):
                found = True
                break
        if found:
            matched += 1
        else:
            missing.append((e1_hint, e2_hint))
    return matched, len(must_find_pairs), missing


def run_benchmark():
    import tempfile
    cfg = load_config()

    print("=" * 72)
    # Detect mode
    from server.config import merge_llm_extraction
    _ext = merge_llm_extraction(cfg["llm"])
    _mode = "V3 (dual-model)" if _ext.get("enabled") else "V2"
    _ext_model = _ext.get("model", cfg["llm"]["model"]) if _ext.get("enabled") else cfg["llm"]["model"]
    print(f"{_mode} Pipeline Extraction Completeness Benchmark")
    print(f"Extraction Model: {_ext_model}")
    print(f"Content Model: {cfg['llm']['model']}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    results = {}

    with tempfile.TemporaryDirectory(prefix="bench_v2_") as tmp:
        processor = make_processor(cfg, tmp)

        for case_id, case in EXTRACTION_GOLD.items():
            text = case["text"]
            expected = case["expected_entities"]
            expected_rels = case.get("expected_relations", {})
            must_find_rels = expected_rels.get("must_find", [])

            print(f"\n--- {case_id} ({len(text)} chars) ---")

            t0 = time.time()
            if processor.remember_mode == "v3":
                entities, relations = _run_extraction(processor, text)
            else:
                entities, relations = _run_v2_extraction(processor, text)
            dt = time.time() - t0

            extracted_names = {e["name"] for e in entities}

            # Entity recall
            must_matched, must_total, must_missing = _count_matches(
                expected["must_find"], extracted_names
            )
            must_recall = must_matched / must_total if must_total > 0 else 1.0

            should_matched, should_total, should_missing = _count_matches(
                expected.get("should_find", []), extracted_names
            )
            should_recall = should_matched / should_total if should_total > 0 else 1.0

            # Relation recall
            rel_matched, rel_total, rel_missing = check_relation_match(
                must_find_rels, relations
            )
            rel_recall = rel_matched / rel_total if rel_total > 0 else 1.0

            # Density
            density = len(entities) / max(1, len(text) / 100)

            result = {
                "case_id": case_id,
                "text_chars": len(text),
                "entities_found": len(entities),
                "relations_found": len(relations),
                "must_find_recall": round(must_recall, 3),
                "must_find_matched": f"{must_matched}/{must_total}",
                "should_find_recall": round(should_recall, 3),
                "should_find_matched": f"{should_matched}/{should_total}",
                "relation_recall": round(rel_recall, 3),
                "relation_matched": f"{rel_matched}/{rel_total}",
                "density": round(density, 2),
                "time_seconds": round(dt, 1),
                "must_missing": must_missing,
                "should_missing": should_missing,
                "rel_missing": [f"{a}-{b}" for a, b in rel_missing],
            }
            results[case_id] = result

            # Print
            status = "PASS" if must_recall >= 0.8 else "FAIL"
            print(f"  Entities: {len(entities)} | Relations: {len(relations)}")
            print(f"  Must-find recall: {must_recall:.0%} ({must_matched}/{must_total}) [{status}]")
            if must_missing:
                print(f"    Missing: {must_missing}")
            print(f"  Should-find recall: {should_recall:.0%} ({should_matched}/{should_total})")
            if rel_total > 0:
                rel_status = "PASS" if rel_recall >= 0.6 else "FAIL"
                print(f"  Relation recall: {rel_recall:.0%} ({rel_matched}/{rel_total}) [{rel_status}]")
                if rel_missing:
                    print(f"    Missing: {[(a, b) for a, b in rel_missing]}")
            print(f"  Density: {density:.2f}/100chars | Time: {dt:.1f}s")

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Case':<25} {'Must%':>6} {'Should%':>8} {'Rel%':>6} {'Ents':>5} {'Rels':>5} {'Time':>6}")
    print("-" * 72)

    total_must_fail = 0
    total_rel_fail = 0
    for case_id, r in results.items():
        must_pass = "PASS" if r["must_find_recall"] >= 0.8 else "FAIL"
        rel_pass = "PASS" if r["relation_recall"] >= 0.6 or r["relation_matched"] == "0/0" else "FAIL"
        if must_pass == "FAIL":
            total_must_fail += 1
        if rel_pass == "FAIL":
            total_rel_fail += 1
        print(f"{case_id:<25} {r['must_find_recall']:>5.0%} {r['should_find_recall']:>7.0%} "
              f"{r['relation_recall']:>5.0%} {r['entities_found']:>5} {r['relations_found']:>5} "
              f"{r['time_seconds']:>5.1f}s")

    print("-" * 72)
    print(f"Entity FAIL: {total_must_fail} | Relation FAIL: {total_rel_fail}")

    # Save JSON report
    report_dir = Path(__file__).resolve().parent / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"bench_completeness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nReport saved: {report_path}")

    return total_must_fail == 0


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
