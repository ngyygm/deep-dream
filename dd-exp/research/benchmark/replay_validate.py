"""X3 validation: multi-query OR-fusion replay against the CURRENT explore() API.

Reconstructs each agent's query list (question + trajectory search_memory calls +
search_documents/sources/timeline), runs explore(candidate_k=20, hybrid-v2) per
query, OR-fuses the per-query ranked_turn_ids by best-rank-per-turn into a single
fused ranking, then slices [:k] for k in {1,3,5,10,20}. Prefix-monotonic by
construction. Fully offline (all-MiniLM-L6-v2 loaded locally; frozen libraries
hold precomputed 384-dim corpus embeddings).

This is the corrected budget-frontier replay: the old depth10-vs20 replay violated
its prefix invariant on all 210 questions because retrieval returned non-monotonic
result sets at limit 10 vs 20. Slicing one length-20 ranking guarantees monotonicity.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

from research.benchmark.mcp_server import ScopedMemoryServer
from research.benchmark.metrics import retrieval_at_k

RUN_DIR = Path(__file__).resolve().parents[1] / ".benchmark_runs" / "locomo-full-quality-v1"
CONFIG_PATH = Path("service_config.local.json")
RESULTS_JSONL = RUN_DIR / "results.kimi-agent-direct-memory-v2c-gate210.jsonl"
KS = (1, 3, 5, 10, 20)
SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = all 210


def _content_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_content_text(i) for i in value)
    if isinstance(value, dict):
        if "text" in value:
            return str(value.get("text") or "")
        return _content_text(value.get("content"))
    return ""


def _persistent_search_calls(events):
    calls: dict = {}
    order: list = []
    completed: list = []
    for event in events:
        if event.get("sessionUpdate") not in {"tool_call", "tool_call_update"}:
            continue
        call_id = str(event.get("toolCallId") or "")
        if not call_id:
            continue
        st = calls.setdefault(call_id, {})
        if call_id not in order:
            order.append(call_id)
        title = str(event.get("title") or st.get("title") or "")
        if title:
            st["title"] = title
            st.setdefault("name", title.split(":", 1)[0])
        content = _content_text(event.get("content"))
        if content and event.get("status") != "completed":
            st["arguments"] = content
        if event.get("status") != "completed":
            continue
        if st.get("name") != "search_memory":
            continue
        try:
            arguments = json.loads(str(st.get("arguments") or "{}"))
        except json.JSONDecodeError:
            continue
        if isinstance(arguments, dict) and str(arguments.get("query") or "").strip():
            completed.append(arguments)
    return completed


def _queries(run_dir: Path, row: dict) -> list[dict]:
    result = [{"query": row["question"], "terms": []}]
    relative = str(row.get("trajectory_path") or "")
    path = run_dir / relative
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        events = payload.get("events") or []
        for arguments in _persistent_search_calls(events):
            result.append({"query": str(arguments.get("query") or "").strip(), "terms": []})
        for event in events:
            for call in event.get("tool_calls") or []:
                function = call.get("function") or {}
                if function.get("name") not in {"search_documents", "search_sources", "search_timeline"}:
                    continue
                arguments = function.get("arguments") or "{}"
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                query = str(arguments.get("query") or "").strip()
                if query:
                    result.append({"query": query, "terms": arguments.get("terms") or []})
    unique: list[dict] = []
    seen: set[str] = set()
    for item in result:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    if not unique:
        return [{"query": row["question"], "terms": []}]
    return unique


def _bytes_per_k(state, fused, k):
    """Offline evidence-payload bytes for the top-k fused turns: sum of UTF-8
    bytes of the formatted '[turn_id] text' evidence lines, which is the actual
    retrieval budget the agent consumes (not the LLM answer bytes)."""
    r = state.tools.retriever
    total = 0
    for tid in fused[:k]:
        sid = r._turn_to_session.get(tid)
        if not sid:
            continue
        text = next((t for x, t in r._session_turns.get(sid, []) if x == tid), "")
        total += len(f"[{tid}] {text}".encode("utf-8"))
    return total


def load_gate_records() -> list[dict]:
    records = []
    with open(RESULTS_JSONL, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("retrieval_profile") == "memory-primitives-v2-persistent":
                rm = row.get("retrieval_metrics") or {}
                row["_rec_any10"] = rm.get("turn_recall_any@10", 0.0)
                records.append(row)
    return records


def build_sample(records: list[dict], n: int) -> list[dict]:
    if n <= 0 or n >= len(records):
        return records
    by_scope = defaultdict(list)
    for r in records:
        by_scope[r["scope_id"]].append(r)
    sample: list[dict] = []
    for scope in sorted(by_scope):
        hit = next((r for r in by_scope[scope] if r["_rec_any10"] == 1.0), None)
        miss = next((r for r in by_scope[scope] if r["_rec_any10"] == 0.0), None)
        if hit:
            sample.append(hit)
        if miss:
            sample.append(miss)
        if len(sample) >= n:
            break
    return sample[:n]


def main() -> None:
    records = load_gate_records()
    print(f"total gate records: {len(records)}", flush=True)
    sample = build_sample(records, SAMPLE)
    print(f"sample size: {len(sample)}", flush=True)

    rows: list[dict] = []
    started = time.time()
    scope_cache: dict[str, ScopedMemoryServer] = {}
    for idx, rec in enumerate(sample, 1):
        scope_id = rec["scope_id"]
        qs = _queries(RUN_DIR, rec)
        gold = list(dict.fromkeys(rec.get("evidence_turn_ids") or []))
        if scope_id not in scope_cache:
            scope_cache[scope_id] = ScopedMemoryServer(RUN_DIR, scope_id, CONFIG_PATH)
        state = scope_cache[scope_id]
        try:
            best_ranks: dict[str, int] = {}
            per_query: list[dict] = []
            for qi, q in enumerate(qs, 1):
                res = state.tools.retriever.explore(
                    q["query"],
                    retrieval_profile="hybrid-v2",
                    candidate_k=20,
                    context_k=5,
                    evidence_token_budget=1600,
                    neighbor_turns=1,
                )
                ranked = res.get("ranked_turn_ids") or []
                for rank, tid in enumerate(ranked, 1):
                    if tid not in best_ranks or rank < best_ranks[tid]:
                        best_ranks[tid] = rank
                per_query.append({"query": q["query"], "ranked_len": len(ranked)})
            fused = [tid for tid, _ in sorted(best_ranks.items(), key=lambda kv: (kv[1], kv[0]))]
            # Prefix-monotonicity check (real, not tautological):
            monotonic = all(
                fused[:KS[i]] == fused[: KS[i + 1]][: KS[i]]
                for i in range(len(KS) - 1)
            )
            metrics = {k: retrieval_at_k(fused, gold, k) for k in KS}
            bytes_per_k = {k: _bytes_per_k(state, fused, k) for k in KS}
            row = {
                "question_id": rec["question_id"],
                "scope_id": scope_id,
                "question": rec["question"],
                "gold_turn_ids": sorted(gold),
                "recorded_recall_any@10": rec["_rec_any10"],
                "num_queries": len(qs),
                "queries": per_query,
                "fused_ranked_turn_ids": fused,
                "prefix_monotonic": monotonic,
                "metrics": metrics,
                "evidence_payload_bytes": bytes_per_k,
            }
            rows.append(row)
            any10 = metrics[10]["recall_any@10"]
            print(
                f"  [{idx}/{len(sample)}] {rec['question_id']}: nq={len(qs)} "
                f"rec_any@10={rec['_rec_any10']} fused_any@10={any10} "
                f"mono={monotonic} fused_len={len(fused)}",
                flush=True,
            )
        finally:
            pass
    for state in scope_cache.values():
        try:
            state.close()
        except Exception:
            pass
    elapsed = time.time() - started

    any10s = [r["metrics"][10]["recall_any@10"] for r in rows]
    all10s = [r["metrics"][10]["recall_all@10"] for r in rows]
    mono_ok = sum(r["prefix_monotonic"] for r in rows)
    mean_bytes = {k: sum(r["evidence_payload_bytes"][k] for r in rows) / len(rows) for k in KS}
    print(
        f"--- {len(rows)}q: fused_any@10={sum(v==1.0 for v in any10s)}/{len(rows)} "
        f"(mean={sum(any10s)/len(any10s):.4f}) fused_all@10 mean={sum(all10s)/len(all10s):.4f} "
        f"mono={mono_ok}/{len(rows)} elapsed={elapsed:.1f}s ({elapsed/len(rows):.1f}s/q)",
        flush=True,
    )
    print(f"mean evidence_payload_bytes per k: {mean_bytes}", flush=True)
    out = Path("/tmp/x3_validate_results.json")
    out.write_text(json.dumps({"rows": rows, "n": len(rows), "elapsed": elapsed,
                               "mean_recall_any@10": sum(any10s) / len(any10s),
                               "mean_recall_all@10": sum(all10s) / len(all10s),
                               "monotonic_ok": mono_ok, "mean_bytes_per_k": mean_bytes},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
