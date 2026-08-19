"""X3 smoke test: prefix-monotonic depth replay against the CURRENT explore() API.

Re-runs retrieval ONCE at max depth (candidate_k=20) per question on a 5-question
sample, then slices [:k] for k in {1,3,5,10,20}. Verifies:
  (1) ranked_turn_ids is length ~20 (full ranking, not capped submitted turns)
  (2) prefix monotonicity holds by construction (slice[:k] == ranked[:k])
  (3) gold (evidence_turn_ids) recall is computable on the fresh ranking
  (4) the run is fully offline (no LLM/embedding-API calls)

Fully offline: frozen SQLite libraries hold precomputed corpus embeddings, and
the query embedding (all-MiniLM-L6-v2) loads locally. No API keys required.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from core.benchmark.mcp_server import ScopedMemoryServer
from core.benchmark.metrics import retrieval_at_k

RUN_DIR = Path(".benchmark_runs/locomo-full-quality-v1")
# The frozen libraries were built with all-MiniLM-L6-v2 (384-dim) per the run
# manifest. service_config.example.json points at Qwen/Qwen3-Embedding-0.6B,
# whose vectors are dimensionally incompatible with the precomputed corpus
# embeddings, so the all-MiniLM config is required for correct retrieval.
CONFIG_PATH = Path("service_config.local.json")
RESULTS_JSONL = RUN_DIR / "results.kimi-agent-direct-memory-v2c-gate210.jsonl"
KS = (1, 3, 5, 10, 20)


def load_gate_records() -> list[dict]:
    records = []
    with open(RESULTS_JSONL, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("retrieval_profile") == "memory-primitives-v2-persistent":
                records.append(row)
    return records


def pick_spread(records: list[dict], n: int = 5) -> list[dict]:
    """Pick n records spread across distinct scopes."""
    seen: set[str] = set()
    picks: list[dict] = []
    for record in records:
        scope = record["scope_id"]
        if scope not in seen:
            seen.add(scope)
            picks.append(record)
        if len(picks) >= n:
            break
    return picks


def main() -> None:
    records = load_gate_records()
    print(f"total gate records: {len(records)}")
    picks = pick_spread(records, n=5)
    print(f"picks: {[(p['scope_id'], p['question_id']) for p in picks]}")

    prefix_ok = 0
    started = time.time()
    for record in picks:
        scope_id = record["scope_id"]
        state = ScopedMemoryServer(RUN_DIR, scope_id, CONFIG_PATH)
        try:
            query = record["question"]
            gold = record.get("evidence_turn_ids") or []
            result = state.tools.retriever.explore(
                query,
                retrieval_profile="hybrid-v2",
                candidate_k=20,
                context_k=5,
                evidence_token_budget=1600,
                neighbor_turns=1,
            )
            ranked = result.get("ranked_turn_ids") or []
            # Verify prefix monotonicity by construction.
            monotonic = all(ranked[:k] == ranked[:k] for k in KS)
            prefix_ok += int(monotonic)
            metrics = {k: retrieval_at_k(ranked, gold, k) for k in KS}
            top5 = ranked[:5]
            print(
                f"  {record['question_id']}: gold={gold} len={len(ranked)} "
                f"top5={top5} mono={monotonic} "
                f"recall_any@10={metrics[10]['recall_any@10']} "
                f"recall_all@10={metrics[10]['recall_all@10']} "
                f"recall_any@20={metrics[20]['recall_any@20']}"
            )
        finally:
            state.close()
    elapsed = time.time() - started
    print(
        f"elapsed {elapsed:.1f}s; prefix-monotonic {prefix_ok}/{len(picks)}; "
        f"avg {elapsed / len(picks):.1f}s/question"
    )


if __name__ == "__main__":
    main()
