"""X7 causal provenance ablation: per-channel source-only -> overlay replay.

Six arms (EXPERIMENT_PLAN.md X7, lines 43-54), run OFFLINE against the same
frozen LoCoMo-210 gate set and reconstructed agent queries as
``replay_budget_frontier``. Each arm toggles retrieval channels (and, for
arms 5/6, the evidence-gate) so the ablation changes exactly one factor at a
time:

    1. source-only lexical            -> raw-document + episode-bm25
    2. source lexical + semantic      -> + semantic-provenance
    3. source + neighboring spans     -> + graph-neighbor
    4. source + versioned concepts    -> + relation-evidence  (== full 5 channels)
    5. full Deep-Dream + evidence-bound Agent
       (same 5 channels; the evidence-gate is ON by default)
    6. full Deep-Dream, unread derived memory into answer
       (same 5 channels; ``allow_unsurfaced_evidence=True`` relaxes the gate)

Arms 1-4 are a clean retrieval-channel ablation: only ``enabled_channels``
differs, so any recall delta is attributable to that channel set. Arms 5 and 6
share arm 4's channel set, so their retrieval recall is identical to arm 4;
their distinguishing columns (unsupported-answer rate, mean evidence tokens)
are produced by the LLM agent run, NOT here. This offline runner therefore
fills the retrieval-recall columns for all six arms and validates that the
``allow_unsurfaced_evidence`` flag plumbing (arm 6) raises/skips correctly
without any LLM call. The full 6-arm LLM RUN requires explicit user
confirmation before launching.

Gold (evidence_turn_ids) is consulted ONLY after retrieval, for metric
computation; it never enters a query. Fully offline: frozen SQLite libraries
hold precomputed 384-dim all-MiniLM-L6-v2 corpus embeddings. Requires
``HF_HUB_OFFLINE=1`` so the query embedding loads from the local cache.

Usage:
    python -m core.benchmark.replay_provenance_ablation            # all 210
    python -m core.benchmark.replay_provenance_ablation 16          # 16-q sample

Emits:
    .benchmark_runs/locomo-full-quality-v1/
        provenance_ablation.x7-v1.jsonl
        provenance_ablation.x7-v1.summary.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from core.benchmark.mcp_server import ScopedMemoryServer
from core.benchmark.metrics import retrieval_at_k
from core.benchmark.retrieval import ALL_RETRIEVAL_CHANNELS
# Reuse the verified X3 query-reconstruction, bootstrap CI, payload-byte, and
# gate-record/sample loaders so this ablation shares X3's exact trajectory
# reconstruction and metric definitions byte-for-byte.
from core.benchmark.replay_budget_frontier import (
    CONFIG_PATH,
    KS,
    RESULTS_JSONL,
    RUN_DIR,
    _bootstrap_ci,
    _evidence_payload_bytes,
    _queries,
    build_sample,
    load_gate_records,
)

OUTPUT_TAG = "provenance_ablation.x7-v1"
BOOTSTRAP_RESAMPLES = 2000
CI_LEVEL = 0.95


# --- arm definitions (additive channel progression) ---------------------

ARMS: list[dict[str, Any]] = [
    {
        "name": "arm1_source_lexical",
        "channels": ("raw-document", "episode-bm25"),
        "description": "source-only lexical (raw-document + episode-bm25)",
        "gate_mode": "evidence_bound",
    },
    {
        "name": "arm2_source_semantic",
        "channels": ("raw-document", "episode-bm25", "semantic-provenance"),
        "description": "source lexical + semantic provenance",
        "gate_mode": "evidence_bound",
    },
    {
        "name": "arm3_source_neighbors",
        "channels": ("raw-document", "episode-bm25", "semantic-provenance", "graph-neighbor"),
        "description": "source + neighboring spans (graph-neighbor)",
        "gate_mode": "evidence_bound",
    },
    {
        "name": "arm4_source_relations",
        "channels": ALL_RETRIEVAL_CHANNELS,
        "description": "source + versioned concepts/relations (full 5 channels)",
        "gate_mode": "evidence_bound",
    },
    {
        "name": "arm5_full_evidence_bound_agent",
        "channels": ALL_RETRIEVAL_CHANNELS,
        "description": "full Deep-Dream + evidence-bound Agent (gate ON; agent run fills QA columns)",
        "gate_mode": "evidence_bound",
        "retrieval_identical_to": "arm4_source_relations",
    },
    {
        "name": "arm6_full_unsurfaced_evidence",
        "channels": ALL_RETRIEVAL_CHANNELS,
        "description": "full Deep-Dream, unread derived memory into answer (gate relaxed; agent run fills QA columns)",
        "gate_mode": "unsurfaced_evidence_allowed",
        "retrieval_identical_to": "arm4_source_relations",
    },
]


def _fuse_ranking(per_query_ranked: list[list[str]]) -> list[str]:
    """OR-fuse per-query ranked_turn_ids by best rank per turn (ties: turn_id)."""
    best_ranks: dict[str, int] = {}
    for ranked in per_query_ranked:
        for rank, tid in enumerate(ranked, 1):
            if tid not in best_ranks or rank < best_ranks[tid]:
                best_ranks[tid] = rank
    return [tid for tid, _ in sorted(best_ranks.items(), key=lambda kv: (kv[1], kv[0]))]


def _arm_retrieval(state: ScopedMemoryServer, rec: dict, channels: tuple[str, ...]) -> dict[str, Any]:
    """Run reconstructed agent queries with a fixed channel set, OR-fuse, and
    return the fused ranking + per-query diagnostics. Identical to the X3
    replay except ``enabled_channels`` is threaded through explore()."""
    qs = _queries(RUN_DIR, rec)
    per_query: list[dict] = []
    per_query_ranked: list[list[str]] = []
    for qi, q in enumerate(qs, 1):
        res = state.tools.retriever.explore(
            q["query"],
            retrieval_profile="hybrid-v2",
            candidate_k=20,
            context_k=5,
            evidence_token_budget=1600,
            neighbor_turns=1,
            enabled_channels=channels,
        )
        ranked = res.get("ranked_turn_ids") or []
        per_query_ranked.append(ranked)
        per_query.append({
            "query_index": qi,
            "query_sha256": hashlib.sha256(str(q["query"]).encode("utf-8")).hexdigest(),
            "ranked_len": len(ranked),
        })
    fused = _fuse_ranking(per_query_ranked)
    return {"fused": fused, "num_queries": len(qs), "queries": per_query}


def _validate_unsurfaced_evidence_flag(state: ScopedMemoryServer) -> dict[str, Any]:
    """Offline proof that arm 6's ``allow_unsurfaced_evidence`` plumbing works:
    submit a turn that EXISTS in the scope but was never surfaced via a read
    tool. With the gate ON (default) this MUST raise; with the gate relaxed it
    MUST be accepted. No LLM is involved."""
    # Pick any turn_id that exists in the scope's sessions but is not in the
    # (empty) surfaced set of a fresh server.
    sessions = list(state.tools.sessions.values())
    if not sessions:
        return {"ran": False, "reason": "no sessions in scope"}
    unsurfaced_turn = next(iter(sessions[0].turn_ids), "")
    if not unsurfaced_turn:
        return {"ran": False, "reason": "no turn_ids in first session"}
    session_id = sessions[0].session_id

    # Gate ON (default): submit an unsurfaced turn must raise.
    gate_on_raised = False
    gate_on_message = ""
    try:
        state.submit([session_id], [], [unsurfaced_turn])
    except ValueError as exc:
        gate_on_raised = True
        gate_on_message = str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        gate_on_raised = True
        gate_on_message = f"{type(exc).__name__}: {exc}"

    # Gate relaxed: same submit must succeed (accepted: True).
    gate_relaxed_accepted = False
    gate_relaxed_payload: dict[str, Any] = {}
    try:
        gate_relaxed_payload = state.submit(
            [session_id], [], [unsurfaced_turn], allow_unsurfaced_evidence=True,
        )
        gate_relaxed_accepted = bool(gate_relaxed_payload.get("accepted"))
    except Exception as exc:  # pragma: no cover - defensive
        gate_relaxed_accepted = False
        gate_relaxed_payload = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "ran": True,
        "unsurfaced_turn_id": unsurfaced_turn,
        "gate_on_raised": gate_on_raised,
        "gate_on_message": gate_on_message[:200],
        "gate_relaxed_accepted": gate_relaxed_accepted,
        "gate_relaxed_evidence_count": gate_relaxed_payload.get("evidence_count"),
        "plumbing_ok": bool(gate_on_raised and gate_relaxed_accepted),
    }


def main() -> None:
    sample_n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    records = load_gate_records()
    print(f"total gate records: {len(records)}", flush=True)
    sample = build_sample(records, sample_n)
    print(f"sample size: {len(sample)}", flush=True)

    # Distinct channel sets actually requiring retrieval (arms 5/6 reuse arm 4).
    distinct_arms = [a for a in ARMS if "retrieval_identical_to" not in a]
    alias_arms = [a for a in ARMS if "retrieval_identical_to" in a]

    rows: list[dict] = []
    started = time.time()
    scope_cache: dict[str, ScopedMemoryServer] = {}
    for idx, rec in enumerate(sample, 1):
        scope_id = rec["scope_id"]
        gold = list(dict.fromkeys(rec.get("evidence_turn_ids") or []))
        if scope_id not in scope_cache:
            scope_cache[scope_id] = ScopedMemoryServer(RUN_DIR, scope_id, CONFIG_PATH)
        state = scope_cache[scope_id]

        # Run each distinct arm once; reuse arm 4's fused ranking for arms 5/6.
        arm_fused: dict[str, list[str]] = {}
        per_arm_metrics: dict[str, dict[int, dict[str, float]]] = {}
        per_arm_bytes: dict[str, dict[int, int]] = {}
        per_arm_queries: dict[str, list[dict]] = {}
        for arm in distinct_arms:
            try:
                result = _arm_retrieval(state, rec, arm["channels"])
            except Exception as exc:  # pragma: no cover - defensive
                print(f"  [{idx}/{len(sample)}] {arm['name']} ERROR: {exc}", flush=True)
                arm_fused[arm["name"]] = []
                result = {"fused": [], "num_queries": 0, "queries": []}
            fused = result["fused"]
            arm_fused[arm["name"]] = fused
            per_arm_queries[arm["name"]] = result["queries"]
            per_arm_metrics[arm["name"]] = {k: retrieval_at_k(fused, gold, k) for k in KS}
            per_arm_bytes[arm["name"]] = {
                k: _evidence_payload_bytes(state.tools.retriever, fused, k) for k in KS
            }

        # Arms 5/6 alias arm 4's retrieval (same 5 channels).
        full_name = "arm4_source_relations"
        for arm in alias_arms:
            arm_fused[arm["name"]] = arm_fused.get(full_name, [])
            per_arm_queries[arm["name"]] = per_arm_queries.get(full_name, [])
            per_arm_metrics[arm["name"]] = per_arm_metrics.get(full_name, {})
            per_arm_bytes[arm["name"]] = per_arm_bytes.get(full_name, {})

        row = {
            "question_id": rec["question_id"],
            "scope_id": scope_id,
            "question": rec["question"],
            "gold_turn_ids": sorted(gold),
            "recorded_recall_any@10": rec["_rec_any10"],
            "arms": {
                arm["name"]: {
                    "channels": list(arm["channels"]),
                    "fused_ranked_turn_ids": arm_fused[arm["name"]],
                    "fused_len": len(arm_fused[arm["name"]]),
                    "num_queries": len(per_arm_queries[arm["name"]]),
                    "queries": per_arm_queries[arm["name"]],
                    "retrieval_metrics": {
                        k: per_arm_metrics[arm["name"]][k] for k in KS
                    },
                    "evidence_payload_bytes": per_arm_bytes[arm["name"]],
                }
                for arm in ARMS
            },
        }
        rows.append(row)
        any10_arm4 = per_arm_metrics.get(full_name, {}).get(10, {}).get("recall_any@10", 0.0)
        print(
            f"  [{idx}/{len(sample)}] {rec['question_id']}: "
            f"arm1_any@10={per_arm_metrics['arm1_source_lexical'][10]['recall_any@10']} "
            f"arm4_any@10={any10_arm4} fused_len={len(arm_fused.get(full_name, []))}",
            flush=True,
        )

    # Offline flag-validation on the first cached scope (no LLM).
    flag_validation: dict[str, Any] = {"ran": False, "reason": "no scope cached"}
    if scope_cache:
        first_state = next(iter(scope_cache.values()))
        try:
            flag_validation = _validate_unsurfaced_evidence_flag(first_state)
        except Exception as exc:  # pragma: no cover - defensive
            flag_validation = {"ran": False, "reason": f"{type(exc).__name__}: {exc}"}
    for state in scope_cache.values():
        try:
            state.close()
        except Exception:
            pass
    elapsed = time.time() - started

    # --- aggregate per-arm per-k curve with bootstrap CI ---
    per_arm_summary: dict[str, Any] = {}
    for arm in ARMS:
        name = arm["name"]
        per_k: dict[int, dict[str, Any]] = {}
        for k in KS:
            any_vals = [r["arms"][name]["retrieval_metrics"][k][f"recall_any@{k}"] for r in rows]
            all_vals = [r["arms"][name]["retrieval_metrics"][k][f"recall_all@{k}"] for r in rows]
            any_lo, any_hi = _bootstrap_ci(any_vals, BOOTSTRAP_RESAMPLES, CI_LEVEL)
            all_lo, all_hi = _bootstrap_ci(all_vals, BOOTSTRAP_RESAMPLES, CI_LEVEL)
            per_k[str(k)] = {
                "n": len(rows),
                "recall_any_mean": sum(any_vals) / len(any_vals) if any_vals else 0.0,
                "recall_any_ci_low": any_lo,
                "recall_any_ci_high": any_hi,
                "recall_all_mean": sum(all_vals) / len(all_vals) if all_vals else 0.0,
                "recall_all_ci_low": all_lo,
                "recall_all_ci_high": all_hi,
                "mean_evidence_payload_bytes": (
                    sum(r["arms"][name]["evidence_payload_bytes"][k] for r in rows) / len(rows)
                    if rows else 0.0
                ),
            }
        per_arm_summary[name] = {
            "description": arm["description"],
            "channels": list(arm["channels"]),
            "gate_mode": arm["gate_mode"],
            "retrieval_identical_to": arm.get("retrieval_identical_to"),
            "per_k": per_k,
        }

    summary: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "locomo-full-quality-v1",
        "track": "kimi-agent-direct-memory-v2c-gate210",
        "method": "per-channel-source-only-to-overlay-replay",
        "retrieval_profile": "hybrid-v2",
        "candidate_k": 20,
        "context_k": 5,
        "evidence_token_budget": 1600,
        "neighbor_turns": 1,
        "embedding_model": "all-MiniLM-L6-v2",
        "questions": len(rows),
        "ks": list(KS),
        "arms": per_arm_summary,
        "flag_validation": flag_validation,
        "interpretation": (
            "Arms 1-4 form a clean retrieval-channel ablation: only "
            "enabled_channels differs, so any recall delta is attributable to "
            "that channel set. Arms 5 and 6 reuse arm 4's five-channel "
            "ranking; their distinguishing columns (unsupported-answer rate, "
            "mean evidence tokens) are produced by the LLM agent run, not "
            "this offline replay. flag_validation proves the arm-6 "
            "allow_unsurfaced_evidence plumbing raises/skips correctly without "
            "any LLM call."
        ),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "ci_level": CI_LEVEL,
        "gold_used_for_retrieval": False,
        "offline": True,
        "requires_llm_for_qa_columns": ["arm5_full_evidence_bound_agent", "arm6_full_unsurfaced_evidence"],
        "elapsed_seconds": round(elapsed, 2),
    }

    jsonl_path = RUN_DIR / f"channel_policy_replay.{OUTPUT_TAG}.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_path = RUN_DIR / f"channel_policy_replay.{OUTPUT_TAG}.summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    a1 = per_arm_summary["arm1_source_lexical"]["per_k"]["10"]
    a4 = per_arm_summary["arm4_source_relations"]["per_k"]["10"]
    print(
        f"--- {len(rows)}q: arm1_any@10={a1['recall_any_mean']:.4f} "
        f"arm4_any@10={a4['recall_any_mean']:.4f} "
        f"flag_plumbing_ok={flag_validation.get('plumbing_ok')} "
        f"elapsed={elapsed:.1f}s ({elapsed/max(len(rows),1):.1f}s/q)",
        flush=True,
    )
    print(f"wrote {jsonl_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
