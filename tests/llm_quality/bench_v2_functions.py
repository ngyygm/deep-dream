#!/usr/bin/env python3
"""
Standalone benchmark for V2 LLM functions.

Tests each V2 extraction function individually against gold standard data.
Outputs per-case and aggregate metrics. No pytest dependency.

Usage:
    cd /home/linkco/exa/Deep-Dream
    CUDA_VISIBLE_DEVICES=0 python tests/llm_quality/bench_v2_functions.py
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
from processor.llm.client import LLMClient
from processor.storage.embedding import EmbeddingClient
from processor.pipeline.new_extraction import _dedup_entity_names


def load_config():
    cfg_path = Path(__file__).resolve().parent.parent.parent / "service_config.json"
    with open(cfg_path) as f:
        return json.load(f)


def make_llm_client(cfg):
    llm = cfg["llm"]
    emb_cfg = cfg.get("embedding", {})
    emb_client = EmbeddingClient(
        model_path=emb_cfg.get("model"),
        device=emb_cfg.get("device", "cpu"),
        use_local=True,
    )
    return LLMClient(
        api_key=llm["api_key"],
        model_name=llm["model"],
        base_url=llm["base_url"],
        think_mode=llm.get("think", False),
        max_tokens=llm.get("max_tokens", 4000),
        context_window_tokens=llm.get("context_window_tokens", 7500),
        max_llm_concurrency=2,
        embedding_client=emb_client,
        content_snippet_length=300,
        relation_content_snippet_length=200,
    )


def fuzzy_match(expected, actual_names):
    """Check if expected entity name is fuzzily matched in actual names."""
    expected_lower = expected.lower()
    for name in actual_names:
        name_lower = name.lower()
        if expected_lower in name_lower or name_lower in expected_lower:
            return True
    return False


def count_matches(expected_list, actual_names):
    """Count matched, total, missing."""
    matched = 0
    missing = []
    for expected in expected_list:
        if fuzzy_match(expected, actual_names):
            matched += 1
        else:
            missing.append(expected)
    return matched, len(expected_list), missing


def bench_entity_extraction(client, case_id, case):
    """Benchmark entity name extraction (Step 2 + 2b)."""
    text = case["text"]
    
    t0 = time.time()
    raw_names = client.extract_entity_names_v2(text)
    t_step2 = time.time() - t0
    
    t0 = time.time()
    supplement = client.extract_entity_names_supplement_v2(text, raw_names)
    t_step2b = time.time() - t0
    
    existing_lower = {n.lower() for n in raw_names}
    new_names = [n for n in supplement if n.lower() not in existing_lower]
    all_names = raw_names + new_names
    deduped = _dedup_entity_names(all_names)
    
    actual_set = set(deduped)
    
    must_find = case["expected_entities"]["must_find"]
    should_find = case["expected_entities"].get("should_find", [])
    must_not_find = case["expected_entities"]["must_not_find"]
    
    must_matched, must_total, must_missing = count_matches(must_find, actual_set)
    should_matched, should_total, should_missing = count_matches(should_find, actual_set)
    
    # Check for system leaks
    leaks = []
    for name in actual_set:
        for forbidden in must_not_find:
            if forbidden.lower() in name.lower():
                leaks.append(f"{name} (contains '{forbidden}')")
    
    must_recall = must_matched / must_total if must_total > 0 else 1.0
    should_recall = should_matched / should_total if should_total > 0 else 1.0
    density = len(deduped) / max(1, len(text) / 100)
    
    return {
        "case_id": case_id,
        "step2_count": len(raw_names),
        "step2b_new": len(new_names),
        "deduped_count": len(deduped),
        "must_recall": must_recall,
        "must_matched": must_matched,
        "must_total": must_total,
        "must_missing": must_missing,
        "should_recall": should_recall,
        "should_matched": should_matched,
        "should_total": should_total,
        "should_missing": should_missing,
        "density": density,
        "leaks": leaks,
        "time_step2": t_step2,
        "time_step2b": t_step2b,
        "entities": sorted(deduped),
    }


def bench_relation_discovery(client, case_id, case, entity_names):
    """Benchmark relation pair discovery (Step 6)."""
    text = case["text"]
    if len(entity_names) < 2:
        return {"case_id": case_id, "pair_count": 0, "recall": 0, "time": 0}
    
    t0 = time.time()
    raw_pairs = client.discover_relation_pairs_v2(entity_names, text)
    elapsed = time.time() - t0
    
    found_pairs = set()
    for a, b in raw_pairs:
        found_pairs.add((a.lower(), b.lower()))
        found_pairs.add((b.lower(), a.lower()))
    
    must_find_rels = case.get("expected_relations", {}).get("must_find", [])
    matched = 0
    missing = []
    for e1_hint, e2_hint in must_find_rels:
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
    
    total = len(must_find_rels)
    recall = matched / total if total > 0 else 1.0
    
    return {
        "case_id": case_id,
        "pair_count": len(raw_pairs),
        "recall": recall,
        "matched": matched,
        "total": total,
        "missing": missing,
        "time": elapsed,
        "pairs": [(a, b) for a, b in raw_pairs],
    }


def bench_entity_content(client, case_id, case, entity_names):
    """Benchmark entity content writing (Step 4) - sample first 3 entities."""
    text = case["text"]
    results = []
    for name in entity_names[:3]:
        t0 = time.time()
        content = client.write_entity_content_v2(name, text)
        elapsed = time.time() - t0
        
        generic = any(p in content for p in ["该实体", "这是一个", "这是一"])
        results.append({
            "name": name,
            "content_len": len(content),
            "content_preview": content[:80],
            "is_generic": generic,
            "time": elapsed,
        })
    return results


def bench_relation_content(client, case_id, case, pairs):
    """Benchmark relation content writing (Step 7) - sample first 3 pairs."""
    text = case["text"]
    results = []
    for a, b in pairs[:3]:
        t0 = time.time()
        content = client.write_relation_content_v2(a, b, text)
        elapsed = time.time() - t0
        
        generic = any(p in content for p in ["有关联", "存在关系", "有关", "相关"])
        results.append({
            "pair": (a, b),
            "content_len": len(content),
            "content_preview": content[:80],
            "is_generic": generic,
            "time": elapsed,
        })
    return results


def main():
    cfg = load_config()
    client = make_llm_client(cfg)
    
    # Select cases for benchmarking
    all_case_ids = list(EXTRACTION_GOLD.keys())
    
    print(f"{'='*70}")
    print(f"V2 Functions Benchmark — {len(all_case_ids)} cases")
    print(f"LLM: {cfg['llm']['model']} @ {cfg['llm']['base_url']}")
    print(f"{'='*70}\n")
    
    entity_results = []
    relation_results = []
    
    for case_id in all_case_ids:
        case = EXTRACTION_GOLD[case_id]
        print(f"--- {case_id} ({len(case['text'])} chars) ---")
        
        # 1. Entity extraction
        ent_result = bench_entity_extraction(client, case_id, case)
        entity_results.append(ent_result)
        
        pass_mark = "PASS" if ent_result["must_recall"] >= 0.8 else "FAIL"
        print(f"  Entities: {ent_result['deduped_count']} extracted, "
              f"must_recall={ent_result['must_recall']:.0%} ({ent_result['must_matched']}/{ent_result['must_total']}) [{pass_mark}]")
        if ent_result["must_missing"]:
            print(f"    Missing: {ent_result['must_missing'][:5]}")
        if ent_result["leaks"]:
            print(f"    LEAKS: {ent_result['leaks']}")
        print(f"    Names: {ent_result['entities'][:8]}{'...' if len(ent_result['entities']) > 8 else ''}")
        print(f"    Time: step2={ent_result['time_step2']:.1f}s, step2b={ent_result['time_step2b']:.1f}s")
        
        # 2. Relation discovery (only for cases with expected relations)
        expected_rels = case.get("expected_relations", {}).get("must_find", [])
        if expected_rels:
            rel_result = bench_relation_discovery(client, case_id, case, ent_result["entities"])
            relation_results.append(rel_result)
            
            pass_mark_r = "PASS" if rel_result["recall"] >= 0.6 else "FAIL"
            print(f"  Relations: {rel_result['pair_count']} pairs, "
                  f"recall={rel_result['recall']:.0%} ({rel_result['matched']}/{rel_result['total']}) [{pass_mark_r}]")
            if rel_result["missing"]:
                print(f"    Missing: {rel_result['missing'][:5]}")
            print(f"    Time: {rel_result['time']:.1f}s")
        
        # 3. Content quality (sample)
        ent_content = bench_entity_content(client, case_id, case, ent_result["entities"])
        generic_count = sum(1 for e in ent_content if e["is_generic"])
        if generic_count > 0:
            print(f"  Content: {generic_count}/{len(ent_content)} sampled entities have generic content")
        
        print()
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    # Entity summary
    avg_must_recall = sum(r["must_recall"] for r in entity_results) / len(entity_results)
    entity_pass = sum(1 for r in entity_results if r["must_recall"] >= 0.8)
    total_leaks = sum(len(r["leaks"]) for r in entity_results)
    avg_density = sum(r["density"] for r in entity_results) / len(entity_results)
    
    print(f"\nEntity Extraction:")
    print(f"  Average must_recall: {avg_must_recall:.1%}")
    print(f"  Cases passing (>=80%): {entity_pass}/{len(entity_results)}")
    print(f"  Total system leaks: {total_leaks}")
    print(f"  Average density: {avg_density:.2f}/100chars")
    
    print(f"\n  {'Case':25s} | {'Ents':>4s} | {'Recall':>7s} | {'Density':>7s} | Status")
    print(f"  {'-'*65}")
    for r in entity_results:
        status = "PASS" if r["must_recall"] >= 0.8 else "FAIL"
        if r["leaks"]:
            status = "LEAK"
        print(f"  {r['case_id']:25s} | {r['deduped_count']:4d} | {r['must_recall']:6.0%} | {r['density']:6.2f} | {status}")
    
    # Relation summary
    if relation_results:
        avg_rel_recall = sum(r["recall"] for r in relation_results) / len(relation_results)
        rel_pass = sum(1 for r in relation_results if r["recall"] >= 0.6)
        
        print(f"\nRelation Discovery:")
        print(f"  Average recall: {avg_rel_recall:.1%}")
        print(f"  Cases passing (>=60%): {rel_pass}/{len(relation_results)}")
        
        print(f"\n  {'Case':25s} | {'Pairs':>5s} | {'Recall':>7s} | Status")
        print(f"  {'-'*55}")
        for r in relation_results:
            status = "PASS" if r["recall"] >= 0.6 else "FAIL"
            print(f"  {r['case_id']:25s} | {r['pair_count']:5d} | {r['recall']:6.0%} | {status}")
    
    # Save JSON report
    report = {
        "timestamp": datetime.now().isoformat(),
        "llm_model": cfg["llm"]["model"],
        "entity_results": entity_results,
        "relation_results": relation_results,
        "summary": {
            "avg_must_recall": avg_must_recall,
            "entity_pass_count": entity_pass,
            "entity_total_count": len(entity_results),
            "total_leaks": total_leaks,
            "avg_density": avg_density,
            "avg_rel_recall": sum(r["recall"] for r in relation_results) / max(1, len(relation_results)),
            "rel_pass_count": sum(1 for r in relation_results if r["recall"] >= 0.6) if relation_results else 0,
            "rel_total_count": len(relation_results),
        },
    }
    
    report_path = Path(__file__).parent / "bench_v2_baseline.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
