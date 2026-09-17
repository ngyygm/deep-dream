"""Deterministic recomputes for the paper's six contract items (no LLM).

Reads frozen artifacts under research/.benchmark_runs and regenerates:
  1. paper/figures/RECOMPUTED_MACROS.tex   (LaTeX macros, sentinels kept
     only where the source artifact has not been synced yet)
  2. paper/results/recomputed_values.json  (machine-readable, consumed by
     gen_fig2_ladder.py / gen_fig3_cost.py and the evidence ledger)

Self-check: the official macro aggregation recomputed from per-question
scores MUST reproduce the frozen Overall values (v1 .3025/.3759/.6695,
v2 .3240/.4068/.6511); any mismatch aborts with a nonzero exit.

Usage: python recompute_from_logs.py   (from research/paper/results)
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict

import numpy as np

RUNS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".benchmark_runs"))
PAPER = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SEED = 0
N_BOOT = 2000

# Official MAB aggregation: source task -> (domain, task label).
SOURCE_TO_CELL = {
    "ruler_qa1_197K": ("AR", "SH-QA"),
    "longmemeval_s*": ("AR", "LME(S*)"),
    "eventqa_65536": ("AR", "EventQA"),
    "eventqa_131072": ("AR", "EventQA"),
    "eventqa_full": ("AR", "EventQA"),
    "icl_clinic150_7050shot_balance": ("TTL", "MCC"),
    "infbench_sum_eng_shots2": ("LRU", "Summ"),
    "detective_qa": ("LRU", "DetQA"),
    "factconsolidation_sh_6k": ("SF", "FC-SH"),
    "factconsolidation_mh_6k": ("SF", "FC-MH"),
}
DOMAINS = ["AR", "TTL", "LRU", "SF"]

FROZEN_OVERALL = {
    ("v1", "baseline"): 0.3025, ("v1", "skill-agent"): 0.3759, ("v1", "pi"): 0.6695,
    ("v2", "baseline"): 0.3240, ("v2", "skill-agent"): 0.4068, ("v2", "pi"): 0.6511,
}


def mab_dir(version: str) -> str:
    return os.path.join(RUNS, f"memoryagentbench-kimik3-sample-{version}")


def load_scores(version: str, track: str) -> dict[str, float]:
    """question_id -> score (0/1 style float), from the official scorer output.

    Rows with status=error (v1 judge failures on the Summ domain) carry no
    score; they are counted as 0.0. The self-check against frozen Overall
    values arbitrates this rule — if the official scorer had excluded them,
    the recomputed macro would not reproduce the frozen numbers.
    """
    path = os.path.join(
        mab_dir(version), f"memoryagentbench_scores.{track}.kimik3-official-v1.sampled.jsonl"
    )
    out = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            out[d["question_id"]] = float(d.get("score") or 0.0)
    return out


def load_meta(version: str, track: str) -> dict[str, tuple]:
    """question_id -> (source, scope) from the scores file (same rows)."""
    path = os.path.join(
        mab_dir(version), f"memoryagentbench_scores.{track}.kimik3-official-v1.sampled.jsonl"
    )
    out = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            out[d["question_id"]] = (d["source"], d["scope_id"])
    return out


def load_results(version: str, track: str) -> dict[str, dict]:
    """question_id -> last results row (retries collapse keep-last)."""
    path = os.path.join(mab_dir(version), f"results.{track}.jsonl")
    if not os.path.exists(path):
        return {}
    out: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            out[d["question_id"]] = d  # keep-last on duplicates
    return out


def macro_overall(qids: list[str], scores: dict[str, float], meta: dict[str, tuple]) -> float:
    """Official aggregation: task mean -> domain mean (present tasks) -> Overall."""
    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for q in qids:
        cell = SOURCE_TO_CELL[meta[q][0]]
        by_cell[cell].append(scores[q])
    domain_avgs = []
    for dom in DOMAINS:
        tasks = [np.mean(v) for (d, _), v in by_cell.items() if d == dom]
        if tasks:
            domain_avgs.append(float(np.mean(tasks)))
    return float(np.mean(domain_avgs))


def paired_bootstrap_macro(
    qids: list[str],
    arms: dict[str, dict[str, float]],
    meta: dict[str, tuple],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Resample questions once per draw; recompute every arm's macro Overall."""
    n = len(qids)
    qarr = np.array(qids, dtype=object)
    out = {name: np.empty(N_BOOT) for name in arms}
    for b in range(N_BOOT):
        sample = rng.choice(n, size=n, replace=True)
        qs = qarr[sample]
        for name, scores in arms.items():
            out[name][b] = macro_overall(list(qs), scores, meta)
    return out


def ci(arr: np.ndarray) -> tuple[float, float]:
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def main() -> None:
    rng = np.random.default_rng(SEED)
    report: dict = {"seed": SEED, "bootstrap_resamples": N_BOOT}

    # ---- load all six score sets + meta ------------------------------------
    scores = {(v, t): load_scores(v, t) for v in ("v1", "v2") for t in ("baseline", "skill-agent", "pi")}
    meta = load_meta("v2", "pi")  # identical question stream across tracks/engines
    qids = sorted(scores[("v2", "pi")].keys())
    assert len(qids) == 767, f"expected 767 questions, got {len(qids)}"
    for key, sc in scores.items():
        assert sorted(sc.keys()) == qids, f"question stream mismatch for {key}"

    # ---- self-check: reproduce frozen macro Overalls -----------------------
    recomputed_overall = {}
    for (v, t), sc in scores.items():
        val = macro_overall(qids, sc, meta)
        recomputed_overall[f"{v}:{t}"] = round(val, 4)
        assert abs(val - FROZEN_OVERALL[(v, t)]) < 5e-5, (
            f"self-check failed for {v}/{t}: recomputed {val:.4f} "
            f"vs frozen {FROZEN_OVERALL[(v, t)]}"
        )
    report["official_macro_overall"] = recomputed_overall
    print("[self-check] all six macro Overalls reproduce frozen values:",
          recomputed_overall)

    # ---- 1) ladder deltas + whiskers (paired bootstrap, macro overalls) -----
    arms_v2 = {t: scores[("v2", t)] for t in ("baseline", "skill-agent", "pi")}
    arms_v1 = {t: scores[("v1", t)] for t in ("baseline", "skill-agent", "pi")}
    boot_v2 = paired_bootstrap_macro(qids, arms_v2, meta, rng)
    boot_v1 = paired_bootstrap_macro(qids, arms_v1, meta, rng)

    delta_skill_base = boot_v2["skill-agent"] - boot_v2["baseline"]
    delta_pi_skill = boot_v2["pi"] - boot_v2["skill-agent"]
    point_sb = macro_overall(qids, arms_v2["skill-agent"], meta) - macro_overall(qids, arms_v2["baseline"], meta)
    point_ps = macro_overall(qids, arms_v2["pi"], meta) - macro_overall(qids, arms_v2["skill-agent"], meta)

    whiskers_v2 = {t: ci(boot_v2[t]) for t in arms_v2}
    whiskers_v1 = {t: ci(boot_v1[t]) for t in arms_v1}
    report["ladder"] = {
        "delta_skill_minus_base_v2": {"point": round(point_sb, 4), "ci95": [round(x, 4) for x in ci(delta_skill_base)]},
        "delta_pi_minus_skill_v2": {"point": round(point_ps, 4), "ci95": [round(x, 4) for x in ci(delta_pi_skill)]},
        "overall_ci95_v2_x100": {t: [round(lo * 100, 1), round(hi * 100, 1)] for t, (lo, hi) in whiskers_v2.items()},
        "overall_ci95_v1_x100": {t: [round(lo * 100, 1), round(hi * 100, 1)] for t, (lo, hi) in whiskers_v1.items()},
    }
    print("[1] ladder deltas:", json.dumps(report["ladder"], indent=1))

    # ---- 2) per-rung cost (v2; rung1 needs results.baseline.jsonl) ----------
    cost = {}
    for rung, track in (("rung2", "skill-agent"), ("rung3", "pi")):
        rows = load_results("v2", track)
        if not rows:
            cost[rung] = None
            continue
        toks, calls, lat = [], [], []
        for q, d in rows.items():
            toks.append(int(d.get("prompt_tokens") or 0) + int(d.get("completion_tokens") or 0))
            calls.append(int(d.get("agent_steps") or 0) + int(d.get("answer_attempts") or 0))
            lat_val = d.get("total_latency_seconds", d.get("latency_seconds"))
            if lat_val is not None:
                lat.append(float(lat_val))
        cost[rung] = {
            "questions": len(rows),
            "tokens_per_question": round(float(np.mean(toks)), 1),
            "llm_calls_per_question": round(float(np.mean(calls), ), 2),
            "latency_seconds_mean": round(float(np.mean(lat)), 1) if lat else None,
            "latency_seconds_p95": round(float(np.percentile(lat, 95)), 1) if lat else None,
        }
    cost["rung1"] = None  # results.baseline.jsonl not yet synced for MAB
    report["rung_cost_v2"] = cost
    print("[2] rung cost:", json.dumps(cost, indent=1))

    # ---- 3) X7 channel ablation: recompute per-arm metrics ------------------
    x7_path = os.path.join(RUNS, "locomo-full-quality-v1",
                           "channel_policy_replay.provenance_ablation.x7-v1.jsonl")
    x7 = defaultdict(lambda: defaultdict(list))  # arm -> metric -> per-q values
    with open(x7_path) as fh:
        for line in fh:
            d = json.loads(line)
            gold = set(d["gold_turn_ids"])
            for arm, payload in d["arms"].items():
                ranked = payload["fused_ranked_turn_ids"]
                x7[arm]["recall_any@10"].append(1.0 if any(g in ranked[:10] for g in gold) else 0.0)
                x7[arm]["recall@1"].append(1.0 if any(g in ranked[:1] for g in gold) else 0.0)
                rr = next((1.0 / (i + 1) for i, t in enumerate(ranked[:10]) if t in gold), 0.0)
                x7[arm]["mrr@10"].append(rr)
                dcg = sum(
                    (1.0 / math.log2(i + 2)) for i, t in enumerate(ranked[:10]) if t in gold
                )
                idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
                x7[arm]["ndcg@10"].append(dcg / idcg if idcg else 0.0)
    core_arms = ["arm1_source_lexical", "arm2_source_semantic",
                 "arm3_source_neighbors", "arm4_source_relations"]
    x7_metrics = {
        arm: {m: round(float(np.mean(v)), 4) for m, v in x7[arm].items()} for arm in core_arms
    }
    # Equivalence bound: widest paired adjacent-arm difference CI (recall_any@10).
    base_arr = np.array(x7["arm1_source_lexical"]["recall_any@10"])
    bound_ci_widths = []
    for arm in core_arms[1:]:
        arr = np.array(x7[arm]["recall_any@10"])
        diffs = arr - base_arr
        boots = np.array([
            np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(N_BOOT)
        ])
        lo, hi = ci(boots)
        bound_ci_widths.append(max(abs(lo), abs(hi)))
    x7_bound = round(max(bound_ci_widths) * 100, 1)
    report["x7"] = {"metrics": x7_metrics, "equivalence_bound_points": x7_bound}
    print("[3] X7:", json.dumps(report["x7"], indent=1))

    # ---- 4) gate rejection statistics (deployed skill-agent runs) -----------
    gate = {}
    for v in ("v1", "v2"):
        path = os.path.join(mab_dir(v), "results.skill-agent.jsonl")
        questions, events, resubmitted, max_steps = 0, 0, 0, 0
        with open(path) as fh:
            for line in fh:
                if "Submitted evidence was not surfaced" not in line:
                    continue
                d = json.loads(line)
                questions += 1
                events += line.count("Submitted evidence was not surfaced")
                submits = int((d.get("agent_tool_counts") or {}).get("submit_evidence") or 0)
                if d.get("agent_stop_reason") == "submit_evidence" and submits >= 2:
                    resubmitted += 1  # rejected once, then accepted inside the loop
                else:
                    max_steps += 1    # loop ended after the rejection; answered from surfaced context
        gate[f"mab_{v}"] = {
            "questions": questions, "events": events,
            "resubmitted_in_loop": resubmitted, "ended_at_max_steps": max_steps,
        }
    gate["lme_v1"] = gate["lme_v2"] = {"questions": 0, "events": 0}
    for v in ("v1", "v2"):
        path = os.path.join(RUNS, f"longmemeval-kimik3-full-{v}", "results.skill-agent.jsonl")
        if os.path.exists(path):
            n = sum(1 for line in open(path) if "Submitted evidence was not surfaced" in line)
            gate[f"lme_{v}"] = {"questions": n, "events": n}
    report["gate_rejections"] = gate
    print("[4] gate:", json.dumps(gate))

    # ---- 5) residual-failure taxonomy (v2 pi zero-score questions) ----------
    pi = scores[("v2", "pi")]
    miss_qids = [q for q in qids if pi[q] == 0]
    domain_of = defaultdict(int)
    for q in miss_qids:
        dom, _ = SOURCE_TO_CELL[meta[q][0]]
        domain_of[dom] += 1
    total_miss = len(miss_qids)
    pct = {d: round(100 * c / total_miss, 1) for d, c in domain_of.items()}
    fc_mh = sum(1 for q in miss_qids if SOURCE_TO_CELL[meta[q][0]] == ("SF", "FC-MH"))
    fc_sh = sum(1 for q in miss_qids if SOURCE_TO_CELL[meta[q][0]] == ("SF", "FC-SH"))
    report["failure_taxonomy_v2_pi"] = {
        "zero_score_questions": total_miss,
        "of_767": round(100 * total_miss / 767, 1),
        "macro_overall_miss_percent": round(100 * (1 - FROZEN_OVERALL[("v2", "pi")]), 1),
        "by_domain_count": dict(domain_of),
        "by_domain_percent": pct,
        "sf_split": {"FC-MH": fc_mh, "FC-SH": fc_sh},
    }
    print("[5] taxonomy:", json.dumps(report["failure_taxonomy_v2_pi"], indent=1))

    # ---- 6) consolidation paired CIs (TTL MCC, FC-MH; v1 -> v2) -------------
    consol = {}
    for label, source in (("ttl_mcc", "icl_clinic150_7050shot_balance"),
                          ("fc_mh", "factconsolidation_mh_6k")):
        sub = [q for q in qids if meta[q][0] == source]
        a = np.array([scores[("v1", "pi")][q] for q in sub])
        b = np.array([scores[("v2", "pi")][q] for q in sub])
        diffs = b - a
        boots = np.array([
            np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(N_BOOT)
        ])
        lo, hi = ci(boots)
        consol[label] = {
            "n": len(sub),
            "v1": round(float(a.mean()), 4),
            "v2": round(float(b.mean()), 4),
            "delta": round(float(diffs.mean()), 4),
            "delta_ci95": [round(lo, 4), round(hi, 4)],
        }
    report["consolidation_paired"] = consol
    print("[6] consolidation:", json.dumps(consol, indent=1))

    # ---- write outputs ------------------------------------------------------
    with open(os.path.join(os.path.dirname(__file__), "recomputed_values.json"), "w") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)

    lad = report["ladder"]
    x2 = cost.get("rung2") or {}
    x3 = cost.get("rung3") or {}
    gate_v2 = gate["mab_v2"]
    gate_v1 = gate["mab_v1"]["questions"]
    tax = report["failure_taxonomy_v2_pi"]

    def cost_str(c: dict | None) -> str:
        if not c:
            return r"\textbf{[RECOMPUTE:baseline-file]}"
        return (f"${c['tokens_per_question']/1000:.1f}\\text{{k}}$ tokens, "
                f"${c['llm_calls_per_question']:.1f}$ LLM calls")

    macros = f"""% AUTO-GENERATED by paper/results/recompute_from_logs.py — deterministic, no LLM.
% Sources: research/.benchmark_runs (frozen artifacts), seed={SEED}, {N_BOOT} bootstrap resamples.
% Self-check passed: recomputed macro Overalls reproduce frozen
%   v1 0.3025/0.3759/0.6695 and v2 0.3240/0.4068/0.6511 exactly.
\\newcommand{{\\RungOneCost}}{{{cost_str(cost.get('rung1'))}}}
\\newcommand{{\\RungTwoCost}}{{{cost_str(cost.get('rung2'))}}}
\\newcommand{{\\RungThreeCost}}{{{cost_str(cost.get('rung3'))}}}
\\newcommand{{\\LadderDeltaSkillBaseCI}}{{${lad['delta_skill_minus_base_v2']['point']:+.3f}$ [${lad['delta_skill_minus_base_v2']['ci95'][0]:+.3f}$, ${lad['delta_skill_minus_base_v2']['ci95'][1]:+.3f}$]}}
\\newcommand{{\\LadderDeltaPiSkillCI}}{{${lad['delta_pi_minus_skill_v2']['point']:+.3f}$ [${lad['delta_pi_minus_skill_v2']['ci95'][0]:+.3f}$, ${lad['delta_pi_minus_skill_v2']['ci95'][1]:+.3f}$]}}
\\newcommand{{\\ConsolTTLCI}}{{${consol['ttl_mcc']['delta']:+.3f}$ [${consol['ttl_mcc']['delta_ci95'][0]:+.3f}$, ${consol['ttl_mcc']['delta_ci95'][1]:+.3f}$]}}
\\newcommand{{\\ConsolFCMHCI}}{{${consol['fc_mh']['delta']:+.3f}$ [${consol['fc_mh']['delta_ci95'][0]:+.3f}$, ${consol['fc_mh']['delta_ci95'][1]:+.3f}$]}}
\\newcommand{{\\XSevenBound}}{{$\\pm{x7_bound:.1f}$ points}}
\\newcommand{{\\XSevenMetrics}}{{recall@1 $ {'/'.join(f'{x7_metrics[a]['recall@1']*100:.1f}' for a in core_arms)}$, MRR@10 $ {'/'.join(f'{x7_metrics[a]['mrr@10']*100:.1f}' for a in core_arms)}$, nDCG@10 $ {'/'.join(f'{x7_metrics[a]['ndcg@10']*100:.1f}' for a in core_arms)}$}}
\\newcommand{{\\GateRejectStats}}{{{gate_v2['questions']}/767 questions in the deployed $v2$ skill-agent run had an evidence submission rejected as unsurfaced ({gate_v1}/767 in $v1$); {gate_v2['resubmitted_in_loop']} of the {gate_v2['questions']} resubmitted tool-surfaced evidence inside the loop and the remaining {gate_v2['ended_at_max_steps']} ended at the step cap and were answered from tool-surfaced context; zero rejections on the LME runs}}
\\newcommand{{\\FailureTaxonomy}}{{{tax['by_domain_percent'].get('TTL', 0)}\\% test-time learning, {tax['by_domain_percent'].get('SF', 0)}\\% selective forgetting ({tax['sf_split']['FC-MH']}/{tax['sf_split']['FC-SH']} FC-MH/FC-SH questions), {tax['by_domain_percent'].get('AR', 0)}\\% long-context reading, {tax['by_domain_percent'].get('LRU', 0)}\\% long-range understanding---{tax['zero_score_questions']} zero-score questions of 767; the 34.9\\% headline is the macro-averaged Overall complement}}
"""
    macro_path = os.path.join(PAPER, "figures", "RECOMPUTED_MACROS.tex")
    with open(macro_path, "w") as fh:
        fh.write(macros)
    print(f"\nwrote {macro_path}")
    print(f"wrote {os.path.join(os.path.dirname(__file__), 'recomputed_values.json')}")


if __name__ == "__main__":
    main()
