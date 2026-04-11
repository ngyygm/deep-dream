#!/usr/bin/env python3
"""
Sequential e2e test runner — submits one dataset at a time, waits for completion.
Generates a comprehensive quality report at the end.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_dataset_extended import EXTENDED_DATASETS
import requests

SESSION = requests.Session()
SESSION.trust_env = False
BASE = os.environ.get("DEEP_DREAM_API", "http://localhost:16200/api/v1")


def api_get(path):
    return SESSION.get(f"{BASE}{path}", timeout=30)


def api_post(path, data):
    return SESSION.post(f"{BASE}{path}", json=data, timeout=120)


def wait_for_task(task_id, timeout=300):
    """Poll until task completes or fails. Retries on connection errors."""
    t0 = time.time()
    consecutive_errors = 0
    while time.time() - t0 < timeout:
        try:
            r = api_get(f"/remember/tasks/{task_id}")
            consecutive_errors = 0
            if r.status_code != 200:
                time.sleep(3)
                continue
            data = r.json().get("data", {})
            status = data.get("status", "unknown")
            if status in ("completed", "failed"):
                return data
            time.sleep(3)
        except Exception:
            consecutive_errors += 1
            if consecutive_errors > 5:
                return {"status": "error", "error": "server_connection_lost"}
            time.sleep(5)
    return {"status": "timeout"}


def run_test(name, ds, retries=2):
    """Submit a single dataset and wait for result. Retries on connection errors."""
    text = ds["text"]
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            r = api_post("/remember", {"text": text, "source": f"test:{name}"})
        except Exception as e:
            if attempt < retries:
                print(f"    Connection error (attempt {attempt+1}), retrying in 15s...")
                time.sleep(15)
                continue
            return {"status": "error", "error": str(e), "elapsed": time.time() - t0,
                    "text_length": len(text)}

        if r.status_code not in (200, 202):
            return {"status": "error", "http_code": r.status_code, "elapsed": time.time() - t0,
                    "text_length": len(text)}

        resp_data = r.json().get("data", {})
        task_id = resp_data.get("task_id")
        initial_status = resp_data.get("status", "")

        if task_id and initial_status in ("queued", "processing"):
            result_data = wait_for_task(task_id)
        elif task_id:
            sr = api_get(f"/remember/tasks/{task_id}")
            result_data = sr.json().get("data", {}) if sr.status_code == 200 else resp_data
        else:
            result_data = resp_data

        # Extract counts
        result_inner = result_data.get("result", {}) or {}
        entity_count = result_inner.get("entities", 0) or result_data.get("entities_count", 0)
        relation_count = result_inner.get("relations", 0) or result_data.get("relations_count", 0)
        elapsed = time.time() - t0

        return {
            "status": result_data.get("status", "unknown"),
            "entities_count": entity_count,
            "relations_count": relation_count,
            "elapsed": elapsed,
            "text_length": len(text),
            "task_id": task_id,
        }

    return {"status": "error", "error": "all_retries_exhausted", "text_length": len(text)}


def main():
    # Check server
    try:
        r = api_get("/health")
        assert r.status_code == 200
        health = r.json()
        print(f"Server OK: backend={health['data']['storage_backend']}, "
              f"embedding={health['data']['embedding_available']}")
    except Exception as e:
        print(f"ERROR: Server not available: {e}")
        sys.exit(1)

    results = {}
    total = len(EXTENDED_DATASETS)
    errors = []

    for i, (name, ds) in enumerate(EXTENDED_DATASETS.items()):
        text = ds["text"]
        dim = f"len={ds.get('length','?')}, dom={ds.get('domain','?')}, lang={ds.get('language','?')}, type={ds.get('type','?')}"
        print(f"\n[{i+1}/{total}] {name} ({len(text)}c) {dim}")

        r = run_test(name, ds)
        results[name] = {**r, "dataset": ds}

        if r["status"] == "completed":
            rate = r["entities_count"] / max(r["text_length"] / 100, 1)
            print(f"  ✓ {r['entities_count']} ents, {r['relations_count']} rels, "
                  f"{r['elapsed']:.1f}s, rate={rate:.1f}/100c")
        else:
            print(f"  ✗ {r['status']}: {r.get('http_code', '')}")
            errors.append(name)

    # ─── Summary Report ──────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("DEEP-DREAM EXTRACTION QUALITY REPORT")
    print("=" * 80)

    completed = {k: v for k, v in results.items() if v["status"] == "completed"}
    total_ent = sum(v["entities_count"] for v in completed.values())
    total_rel = sum(v["relations_count"] for v in completed.values())
    total_time = sum(v["elapsed"] for v in completed.values())
    total_chars = sum(v["text_length"] for v in completed.values())

    print(f"\n  Datasets: {len(completed)}/{total} completed, {len(errors)} errors")
    print(f"  Total text: {total_chars:,} chars")
    print(f"  Total entities: {total_ent}")
    print(f"  Total relations: {total_rel}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg ents/100chars: {total_ent / max(total_chars/100, 1):.1f}")
    print(f"  Avg rels/100chars: {total_rel / max(total_chars/100, 1):.1f}")

    # Per-dataset detail
    print(f"\n  {'Name':<25s} {'Chars':>6s} {'Ents':>5s} {'Rels':>5s} {'Time':>7s} {'Rate':>7s}")
    print("  " + "-" * 60)
    for name, r in sorted(completed.items(), key=lambda x: x[1]["text_length"]):
        rate = r["entities_count"] / max(r["text_length"] / 100, 1)
        print(f"  {name:<25s} {r['text_length']:>6d} {r['entities_count']:>5d} "
              f"{r['relations_count']:>5d} {r['elapsed']:>6.1f}s {rate:>6.1f}/c")

    # Dimension analysis
    for dim in ["length", "domain", "language", "type"]:
        buckets = {}
        for name, r in completed.items():
            key = r["dataset"].get(dim, "unknown")
            buckets.setdefault(key, []).append(r)

        print(f"\n  By {dim}:")
        for key in sorted(buckets.keys()):
            items = buckets[key]
            ents = [i["entities_count"] for i in items]
            rels = [i["relations_count"] for i in items]
            times = [i["elapsed"] for i in items]
            print(f"    {key:15s}: {len(items):2d} texts, "
                  f"avg {sum(ents)/len(ents):.1f} ents, "
                  f"{sum(rels)/len(rels):.1f} rels, "
                  f"{sum(times)/len(times):.1f}s")

    # Quality checks
    print("\n  Quality Checks:")
    issues = []

    # Check 1: All completed
    if errors:
        issues.append(f"  ⚠ {len(errors)} datasets failed: {errors}")

    # Check 2: Every dataset extracts at least 1 entity
    no_ents = [n for n, r in completed.items() if r["entities_count"] == 0]
    if no_ents:
        issues.append(f"  ⚠ Zero entities: {no_ents}")

    # Check 3: Rate is reasonable (2-12 ents/100chars)
    for name, r in completed.items():
        rate = r["entities_count"] / max(r["text_length"] / 100, 1)
        if rate > 15:
            issues.append(f"  ⚠ [{name}] Over-extraction: {rate:.1f} ents/100chars")
        if rate < 1 and r["text_length"] > 50:
            issues.append(f"  ⚠ [{name}] Under-extraction: {rate:.1f} ents/100chars")

    # Check 4: Expected entities
    for name, r in completed.items():
        ds = r["dataset"]
        expected = ds.get("expected_entities", [])
        if expected and r["entities_count"] < len(expected) * 0.3:
            issues.append(f"  ⚠ [{name}] Expected ~{len(expected)} ents, got {r['entities_count']}")

    # Check 5: Relations exist
    no_rels = [n for n, r in completed.items() if r["relations_count"] == 0 and r["text_length"] > 50]
    if no_rels:
        issues.append(f"  ⚠ Zero relations for texts >50 chars: {no_rels}")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✓ All checks passed")

    print("\n" + "=" * 80)

    # Save results to JSON for further analysis
    report_path = os.path.join(os.path.dirname(__file__), "e2e_report.json")
    with open(report_path, "w") as f:
        # Remove non-serializable dataset refs
        clean = {}
        for k, v in results.items():
            clean[k] = {kk: vv for kk, vv in v.items() if kk != "dataset"}
            clean[k]["domain"] = v["dataset"].get("domain", "")
            clean[k]["language"] = v["dataset"].get("language", "")
            clean[k]["content_type"] = v["dataset"].get("type", "")
            clean[k]["length"] = v["dataset"].get("length", "")
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved to: {report_path}")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
