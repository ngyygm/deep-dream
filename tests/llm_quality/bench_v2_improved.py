#!/usr/bin/env python3
"""
Benchmark for improved V2 LLM functions.

Tests: multi-focus entity extraction + chunked relation discovery + expansion.
Compares against the baseline.

Usage:
    cd /home/linkco/exa/Deep-Dream
    CUDA_VISIBLE_DEVICES=0 python tests/llm_quality/bench_v2_improved.py
"""
import json, os, sys, time
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
        api_key=llm["api_key"], model_name=llm["model"],
        base_url=llm["base_url"], think_mode=llm.get("think", False),
        max_tokens=llm.get("max_tokens", 4000),
        context_window_tokens=llm.get("context_window_tokens", 7500),
        max_llm_concurrency=2, embedding_client=emb_client,
        content_snippet_length=300, relation_content_snippet_length=200,
    )


def fuzzy_match(expected, actual_names):
    expected_lower = expected.lower()
    for name in actual_names:
        name_lower = name.lower()
        if expected_lower in name_lower or name_lower in expected_lower:
            return True
    return False


def count_matches(expected_list, actual_names):
    matched, missing = 0, []
    for expected in expected_list:
        if fuzzy_match(expected, actual_names):
            matched += 1
        else:
            missing.append(expected)
    return matched, len(expected_list), missing


def bench_multi_focus_extraction(client, case_id, case):
    """Test multi-focus entity extraction: named + abstract + events + supplement."""
    text = case["text"]

    # Step 2a: Named entities
    t0 = time.time()
    named = client.extract_entity_names_named_v2(text)
    t_named = time.time() - t0

    # Step 2b: Abstract concepts
    t0 = time.time()
    abstract = client.extract_entity_names_abstract_v2(text)
    t_abstract = time.time() - t0

    # Step 2c: Events
    t0 = time.time()
    events = client.extract_entity_names_events_v2(text)
    t_events = time.time() - t0

    # Merge
    all_names = named + abstract + events
    existing_lower = set()
    merged = []
    for n in all_names:
        if n.lower() not in existing_lower:
            existing_lower.add(n.lower())
            merged.append(n)

    # Step 2d: Supplement
    t0 = time.time()
    supplement = client.extract_entity_names_supplement_v2(text, merged)
    t_supplement = time.time() - t0

    for n in supplement:
        if n.lower() not in existing_lower:
            existing_lower.add(n.lower())
            merged.append(n)

    deduped = _dedup_entity_names(merged)
    actual_set = set(deduped)

    must_find = case["expected_entities"]["must_find"]
    should_find = case["expected_entities"].get("should_find", [])
    must_not_find = case["expected_entities"]["must_not_find"]

    must_m, must_t, must_miss = count_matches(must_find, actual_set)
    should_m, should_t, should_miss = count_matches(should_find, actual_set)

    leaks = []
    for name in actual_set:
        for forbidden in must_not_find:
            if forbidden.lower() in name.lower():
                leaks.append(f"{name} (contains '{forbidden}')")

    return {
        "case_id": case_id,
        "named_count": len(named),
        "abstract_count": len(abstract),
        "events_count": len(events),
        "supplement_count": len(supplement),
        "deduped_count": len(deduped),
        "must_recall": must_m / must_t if must_t else 1.0,
        "must_matched": must_m, "must_total": must_t, "must_missing": must_miss,
        "should_recall": should_m / should_t if should_t else 1.0,
        "should_matched": should_m, "should_total": should_t, "should_missing": should_miss,
        "density": len(deduped) / max(1, len(text) / 100),
        "leaks": leaks,
        "time_named": t_named, "time_abstract": t_abstract,
        "time_events": t_events, "time_supplement": t_supplement,
        "entities": sorted(deduped),
    }


def bench_improved_relations(client, case_id, case, entity_names):
    """Test chunked relation discovery + expansion round."""
    text = case["text"]
    if len(entity_names) < 2:
        return {"case_id": case_id, "pair_count": 0, "recall": 0, "time": 0}

    # Step 6a: Chunked discovery
    t0 = time.time()
    raw_pairs = client.discover_relation_pairs_chunked_v2(entity_names, text, chunk_size=10)
    t_discover = time.time() - t0

    # Step 6b: Expansion
    t0 = time.time()
    expand_pairs = client.discover_relation_pairs_expand_v2(entity_names, raw_pairs, text)
    t_expand = time.time() - t0

    # Merge
    all_pairs_set = set(tuple(sorted(p)) for p in raw_pairs)
    new_from_expand = 0
    for p in expand_pairs:
        key = tuple(sorted(p))
        if key not in all_pairs_set:
            all_pairs_set.add(key)
            new_from_expand += 1
    all_pairs = list(all_pairs_set)

    found_pairs = set()
    for a, b in all_pairs:
        found_pairs.add((a.lower(), b.lower()))
        found_pairs.add((b.lower(), a.lower()))

    must_find_rels = case.get("expected_relations", {}).get("must_find", [])
    matched, missing = 0, []
    for e1_hint, e2_hint in must_find_rels:
        found = False
        for f1, f2 in found_pairs:
            if ((e1_hint.lower() in f1 or f1 in e1_hint.lower()) and
                (e2_hint.lower() in f2 or f2 in e2_hint.lower())):
                found = True; break
            if ((e2_hint.lower() in f1 or f1 in e2_hint.lower()) and
                (e1_hint.lower() in f2 or f2 in e1_hint.lower())):
                found = True; break
        if found:
            matched += 1
        else:
            missing.append((e1_hint, e2_hint))

    total = len(must_find_rels)
    return {
        "case_id": case_id,
        "discover_count": len(raw_pairs),
        "expand_new": new_from_expand,
        "pair_count": len(all_pairs),
        "recall": matched / total if total else 1.0,
        "matched": matched, "total": total, "missing": missing,
        "time_discover": t_discover, "time_expand": t_expand,
    }


def main():
    cfg = load_config()
    client = make_llm_client(cfg)
    all_case_ids = list(EXTRACTION_GOLD.keys())

    print(f"{'='*70}")
    print(f"IMPROVED V2 Benchmark — {len(all_case_ids)} cases")
    print(f"LLM: {cfg['llm']['model']} @ {cfg['llm']['base_url']}")
    print(f"{'='*70}\n")

    entity_results, relation_results = [], []

    for case_id in all_case_ids:
        case = EXTRACTION_GOLD[case_id]
        print(f"--- {case_id} ({len(case['text'])} chars) ---")

        # Entity extraction
        ent = bench_multi_focus_extraction(client, case_id, case)
        entity_results.append(ent)

        pm = "PASS" if ent["must_recall"] >= 0.8 else "FAIL"
        print(f"  Entities: {ent['deduped_count']} (named={ent['named_count']}, abstract={ent['abstract_count']}, events={ent['events_count']}, supp={ent['supplement_count']})")
        print(f"    must_recall={ent['must_recall']:.0%} ({ent['must_matched']}/{ent['must_total']}) [{pm}]")
        if ent["must_missing"]:
            print(f"    Missing: {ent['must_missing'][:5]}")
        if ent["leaks"]:
            print(f"    LEAKS: {ent['leaks']}")
        print(f"    Names sample: {ent['entities'][:6]}...")

        # Relation discovery
        exp_rels = case.get("expected_relations", {}).get("must_find", [])
        if exp_rels:
            rel = bench_improved_relations(client, case_id, case, ent["entities"])
            relation_results.append(rel)

            pm_r = "PASS" if rel["recall"] >= 0.6 else "FAIL"
            print(f"  Relations: {rel['pair_count']} (discover={rel['discover_count']}, expand_new={rel['expand_new']})")
            print(f"    recall={rel['recall']:.0%} ({rel['matched']}/{rel['total']}) [{pm_r}]")
            if rel["missing"]:
                print(f"    Missing: {rel['missing'][:5]}")
        print()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    avg_must = sum(r["must_recall"] for r in entity_results) / len(entity_results)
    ent_pass = sum(1 for r in entity_results if r["must_recall"] >= 0.8)
    print(f"\nEntity: avg_recall={avg_must:.1%}, passing={ent_pass}/{len(entity_results)}")
    print(f"  {'Case':25s} | {'Ents':>4s} | {'Recall':>7s} | Status")
    print(f"  {'-'*50}")
    for r in entity_results:
        st = "PASS" if r["must_recall"] >= 0.8 else "FAIL"
        if r["leaks"]: st = "LEAK"
        print(f"  {r['case_id']:25s} | {r['deduped_count']:4d} | {r['must_recall']:6.0%} | {st}")

    if relation_results:
        avg_rel = sum(r["recall"] for r in relation_results) / len(relation_results)
        rel_pass = sum(1 for r in relation_results if r["recall"] >= 0.6)
        print(f"\nRelation: avg_recall={avg_rel:.1%}, passing={rel_pass}/{len(relation_results)}")
        print(f"  {'Case':25s} | {'Pairs':>5s} | {'Recall':>7s} | Status")
        print(f"  {'-'*50}")
        for r in relation_results:
            st = "PASS" if r["recall"] >= 0.6 else "FAIL"
            print(f"  {r['case_id']:25s} | {r['pair_count']:5d} | {r['recall']:6.0%} | {st}")

    # Load baseline for comparison
    baseline_path = Path(__file__).parent / "bench_v2_baseline.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"\n--- Comparison vs Baseline ---")
        bs = baseline["summary"]
        print(f"  Entity recall:   {bs['avg_must_recall']:.1%} → {avg_must:.1%}")
        print(f"  Entity passing:  {bs['entity_pass_count']}/{bs['entity_total_count']} → {ent_pass}/{len(entity_results)}")
        if relation_results:
            print(f"  Relation recall: {bs['avg_rel_recall']:.1%} → {avg_rel:.1%}")
            print(f"  Relation passing: {bs['rel_pass_count']}/{bs['rel_total_count']} → {rel_pass}/{len(relation_results)}")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "entity_results": entity_results,
        "relation_results": relation_results,
    }
    report_path = Path(__file__).parent / "bench_v2_improved.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
