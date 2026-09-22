"""Deterministic recomputes for the paper's six contract items (no LLM).

Reads frozen artifacts under research/.benchmark_runs and regenerates:
  1. paper/figures/RECOMPUTED_MACROS.tex   (LaTeX macros)
  2. paper/results/recomputed_values.json  (machine-readable, consumed by
     gen_fig2_ladder.py / gen_fig3_cost.py and the evidence ledger)

Self-check: the official macro aggregation recomputed from per-question
scores MUST reproduce the frozen Overall values (v1 .3025/.3759/.6695 at
the paired 767-question set; v2 .3333/.4000/.7101 at the extended
1074-question set); any mismatch aborts with a nonzero exit.

Calibers (2026-09-21 holefill): the v2 (Align+) run was extended in place
with four scopes (mab-ttl-000 200q, mab-ar-001 100q, mab-lru-101 6q,
mab-lru-001 1q) to 1074 questions; the v1 (Base) run was NOT extended, so
every v1<->v2 paired statistic stays on the original 767-question
intersection, and every v2 headline moves to 1074. The full-context anchor
(fullctx-neutral-v1) covers 614 of the 1074 (ttl-000 and ar-001 exceed the
1,042,576-char strict budget and are excluded by manifest).

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
    "ruler_qa2_421K": ("AR", "MH-QA"),
    "longmemeval_s*": ("AR", "LME(S*)"),
    "eventqa_65536": ("AR", "EventQA"),
    "eventqa_131072": ("AR", "EventQA"),
    "eventqa_full": ("AR", "EventQA"),
    "icl_clinic150_7050shot_balance": ("TTL", "MCC"),
    "recsys_redial_full": ("TTL", "Recom"),
    "infbench_sum_eng_shots2": ("LRU", "Summ"),
    "detective_qa": ("LRU", "DetQA"),
    "factconsolidation_sh_6k": ("SF", "FC-SH"),
    "factconsolidation_mh_6k": ("SF", "FC-MH"),
}
DOMAINS = ["AR", "TTL", "LRU", "SF"]

# v1 (Base) froze at the paired 767-question caliber; v2 (Align+) was
# extended 2026-09-21 to 1074 questions (holefill chain, official scorer
# rerun in place, summaries 0.333274/0.399959/0.710078).
FROZEN_OVERALL = {
    ("v1", "baseline"): 0.3025, ("v1", "skill-agent"): 0.3759, ("v1", "pi"): 0.6695,
    ("v2", "baseline"): 0.3333, ("v2", "skill-agent"): 0.4000, ("v2", "pi"): 0.7101,
}
N_PAIRED, N_EXTENDED = 767, 1074


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


def load_cost_fields(version: str, track: str) -> dict[str, dict]:
    """question_id -> scalar cost fields (keep-last on retry duplicates).

    Extracts only the cost scalars per row: results.baseline.jsonl embeds the
    full retrieved context (~1.4 MB/row), so holding whole rows for 1074
    questions would pin ~1.5 GB of RAM for a handful of numbers.
    """
    path = os.path.join(mab_dir(version), f"results.{track}.jsonl")
    if not os.path.exists(path):
        return {}
    out: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            out[d["question_id"]] = {
                "tokens": int(d.get("prompt_tokens") or 0)
                + int(d.get("completion_tokens") or 0),
                "calls": int(d.get("agent_steps") or 0)
                + int(d.get("answer_attempts") or 0),
                "latency": d.get("total_latency_seconds", d.get("latency_seconds")),
            }
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
    qids = sorted(scores[("v2", "pi")].keys())          # extended caliber, 1074
    qids_paired = sorted(scores[("v1", "pi")].keys())   # frozen Base caliber, 767
    assert len(qids) == N_EXTENDED, f"expected {N_EXTENDED} questions, got {len(qids)}"
    assert len(qids_paired) == N_PAIRED, f"expected {N_PAIRED} paired questions, got {len(qids_paired)}"
    assert set(qids_paired) <= set(qids), "paired v1 set must be a subset of extended v2"
    qids_of = {"v1": qids_paired, "v2": qids}
    for (v, t), sc in scores.items():
        assert sorted(sc.keys()) == qids_of[v], f"question stream mismatch for {v}/{t}"

    # ---- self-check: reproduce frozen macro Overalls -----------------------
    recomputed_overall = {}
    for (v, t), sc in scores.items():
        val = macro_overall(qids_of[v], sc, meta)
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
    boot_v1 = paired_bootstrap_macro(qids_paired, arms_v1, meta, rng)

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

    # ---- 2) per-rung cost (v2; rung1 from results.baseline.jsonl) -----------
    cost = {}
    for rung, track in (("rung1", "baseline"), ("rung2", "skill-agent"), ("rung3", "pi")):
        rows = load_cost_fields("v2", track)
        if not rows:
            cost[rung] = None
            continue
        toks = [d["tokens"] for d in rows.values()]
        calls = [d["calls"] for d in rows.values()]
        lat = [float(d["latency"]) for d in rows.values() if d["latency"] is not None]
        cost[rung] = {
            "questions": len(rows),
            "tokens_per_question": round(float(np.mean(toks)), 1),
            "llm_calls_per_question": round(float(np.mean(calls)), 2),
            "latency_seconds_mean": round(float(np.mean(lat)), 1) if lat else None,
            "latency_seconds_p95": round(float(np.percentile(lat, 95)), 1) if lat else None,
        }
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
        f"of_{N_EXTENDED}": round(100 * total_miss / N_EXTENDED, 1),
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
        sub = [q for q in qids_paired if meta[q][0] == source]
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

    # ---- 7) domain/task matrices + exact overalls (wires gen_fig2 panels) ---
    def domain_task_x100(qid_list, sc):
        by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
        for q in qid_list:
            by_cell[SOURCE_TO_CELL[meta[q][0]]].append(sc[q])
        domains = {
            dom: round(100 * float(np.mean([np.mean(v) for (d, _), v in by_cell.items() if d == dom])), 1)
            for dom in DOMAINS if any(d == dom for (d, _) in by_cell)
        }
        tasks = {task: round(100 * float(np.mean(v)), 1)
                 for (_, task), v in sorted(by_cell.items())}
        return domains, tasks

    for v in ("v1", "v2"):
        qv = qids_of[v]
        overalls, domains_v, tasks_v = {}, {}, {}
        for t in ("baseline", "skill-agent", "pi"):
            overalls[t] = round(macro_overall(qv, scores[(v, t)], meta) * 100, 1)
            domains_v[t], tasks_v[t] = domain_task_x100(qv, scores[(v, t)])
        report["ladder"][f"overall_{v}_x100"] = overalls
        report["ladder"][f"domains_{v}_x100"] = domains_v
        report["ladder"][f"tasks_{v}_x100"] = tasks_v
    print("[7] fig2 matrices: v2 overall", report["ladder"]["overall_v2_x100"])

    # ---- 8) same-question alignment vs the full-context anchor -------------
    anchor_track = "full-context-kimik3-neutral-v1"
    anchor = load_scores("v2", anchor_track)
    inter = sorted(set(qids) & set(anchor))
    assert len(inter) == 614, f"expected 614 anchor-covered questions, got {len(inter)}"
    align_tracks = ["baseline", "skill-agent", "pi", "fullctx-neutral"]
    arms_align = {**{t: scores[("v2", t)] for t in ("baseline", "skill-agent", "pi")},
                  "fullctx-neutral": anchor}
    boot_align = paired_bootstrap_macro(inter, arms_align, meta, rng)
    delta_pi_anchor = boot_align["pi"] - boot_align["fullctx-neutral"]
    align: dict = {"n": len(inter), "tracks": align_tracks,
                   "overall_x100": {}, "overall_ci95_x100": {}, "domains_x100": {},
                   "delta_pi_minus_anchor_x100": {
                       "point": round((macro_overall(inter, arms_align["pi"], meta)
                                       - macro_overall(inter, anchor, meta)) * 100, 1),
                       "ci95": [round(x * 100, 1) for x in ci(delta_pi_anchor)]}}
    for t in align_tracks:
        align["overall_x100"][t] = round(macro_overall(inter, arms_align[t], meta) * 100, 1)
        lo, hi = ci(boot_align[t])
        align["overall_ci95_x100"][t] = [round(lo * 100, 1), round(hi * 100, 1)]
        align["domains_x100"][t] = domain_task_x100(inter, arms_align[t])[0]
    report["same_question_alignment"] = align
    print(f"[8] same-question alignment n={len(inter)}:",
          json.dumps(align["overall_x100"]))

    # ---- 8b) retained fullctx-v1 artifact at its own coverage --------------
    fc_v1 = load_scores("v2", "full-context-kimik3-v1")
    inter_v1 = sorted(set(qids) & set(fc_v1))
    report["fullctx_v1_artifact"] = {
        "n": len(inter_v1),
        "overall_x100": round(macro_overall(inter_v1, fc_v1, meta) * 100, 1),
        "note": "prompt-sensitivity artifact (normalized-v1 profile) at its own coverage; no bootstrap",
    }
    print("[8b] fullctx-v1 artifact:", json.dumps(report["fullctx_v1_artifact"]))

    # ---- 8c) anchor operating cost (efficiency axis) ------------------------
    ac = load_cost_fields("v2", anchor_track)
    if ac:
        atoks = [d["tokens"] for d in ac.values()]
        report["anchor_cost_neutral_v1"] = {
            "questions": len(ac),
            "tokens_per_question_mean": round(float(np.mean(atoks)), 0),
            "tokens_total_per_round_millions": round(float(np.sum(atoks)) / 1e6, 1),
            "llm_calls_per_question": round(
                float(np.mean([d["calls"] for d in ac.values()])), 2),
        }
        print("[8c] anchor cost:", json.dumps(report["anchor_cost_neutral_v1"]))

    # ---- write outputs ------------------------------------------------------
    with open(os.path.join(os.path.dirname(__file__), "recomputed_values.json"), "w") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)

    lad = report["ladder"]
    x2 = cost.get("rung2") or {}
    x3 = cost.get("rung3") or {}
    gate_v2 = gate["mab_v2"]
    gate_v1 = gate["mab_v1"]["questions"]
    tax = report["failure_taxonomy_v2_pi"]
    align_rep = report.get("same_question_alignment", {})
    align_n = align_rep.get("n", 0)
    anchor_ov_str = f"{align_rep.get('overall_x100', {}).get('fullctx-neutral', float('nan')):.1f}"
    pi_ext_str = f"{report['ladder']['overall_v2_x100']['pi']:.1f}"
    gap = align_rep.get("delta_pi_minus_anchor_x100",
                        {"point": float("nan"), "ci95": [float("nan"), float("nan")]})
    _at = report.get("anchor_cost_neutral_v1", {}).get("tokens_total_per_round_millions")
    anchor_tok_str = (f"${_at:.0f}\\text{{M}}$ tokens per full anchor round"
                      if _at else r"\textbf{[RECOMPUTE:anchor-cost]}")

    def cost_str(c: dict | None) -> str:
        if not c:
            return r"\textbf{[RECOMPUTE:baseline-file]}"
        return (f"${c['tokens_per_question']/1000:.1f}\\text{{k}}$ tokens, "
                f"${c['llm_calls_per_question']:.1f}$ LLM calls")

    macros = f"""% AUTO-GENERATED by paper/results/recompute_from_logs.py — deterministic, no LLM.
% Sources: research/.benchmark_runs (frozen artifacts), seed={SEED}, {N_BOOT} bootstrap resamples.
% Self-check passed: recomputed macro Overalls reproduce frozen
%   v1 0.3025/0.3759/0.6695 (paired n=767) and v2 0.3333/0.4000/0.7101
%   (extended n=1074) exactly. Anchor: fullctx-neutral-v1, n=614.
\\newcommand{{\\DirectRetrievalCost}}{{{cost_str(cost.get('rung1'))}}}
\\newcommand{{\\MemoryToolCost}}{{{cost_str(cost.get('rung2'))}}}
\\newcommand{{\\SourceGroundedCost}}{{{cost_str(cost.get('rung3'))}}}
\\newcommand{{\\LadderDeltaSkillBaseCI}}{{${lad['delta_skill_minus_base_v2']['point']:+.3f}$ [${lad['delta_skill_minus_base_v2']['ci95'][0]:+.3f}$, ${lad['delta_skill_minus_base_v2']['ci95'][1]:+.3f}$]}}
\\newcommand{{\\LadderDeltaPiSkillCI}}{{${lad['delta_pi_minus_skill_v2']['point']:+.3f}$ [${lad['delta_pi_minus_skill_v2']['ci95'][0]:+.3f}$, ${lad['delta_pi_minus_skill_v2']['ci95'][1]:+.3f}$]}}
\\newcommand{{\\ConsolTTLCI}}{{${consol['ttl_mcc']['delta']:+.3f}$ [${consol['ttl_mcc']['delta_ci95'][0]:+.3f}$, ${consol['ttl_mcc']['delta_ci95'][1]:+.3f}$]}}
\\newcommand{{\\ConsolFCMHCI}}{{${consol['fc_mh']['delta']:+.3f}$ [${consol['fc_mh']['delta_ci95'][0]:+.3f}$, ${consol['fc_mh']['delta_ci95'][1]:+.3f}$]}}
\\newcommand{{\\XSevenBound}}{{$\\pm{x7_bound:.1f}$ points}}
\\newcommand{{\\XSevenMetrics}}{{recall@1 $ {'/'.join(f'{x7_metrics[a]['recall@1']*100:.1f}' for a in core_arms)}$, MRR@10 $ {'/'.join(f'{x7_metrics[a]['mrr@10']*100:.1f}' for a in core_arms)}$, nDCG@10 $ {'/'.join(f'{x7_metrics[a]['ndcg@10']*100:.1f}' for a in core_arms)}$}}
\\newcommand{{\\GateRejectStats}}{{{gate_v2['questions']}/{N_EXTENDED} questions in the deployed $v2$ skill-agent run had an evidence submission rejected as unsurfaced ({gate_v1}/{N_PAIRED} in $v1$); {gate_v2['resubmitted_in_loop']} of the {gate_v2['questions']} resubmitted tool-surfaced evidence inside the loop and the remaining {gate_v2['ended_at_max_steps']} ended at the step cap and were answered from tool-surfaced context; zero rejections on the LME runs}}
\\newcommand{{\\FullctxAnchorOverall}}{{{anchor_ov_str}}}
\\newcommand{{\\FullctxAnchorCoverage}}{{{align_n}/{N_EXTENDED} questions}}
\\newcommand{{\\PiExtendedOverall}}{{{pi_ext_str}}}
\\newcommand{{\\PiMinusAnchorCI}}{{${gap['point']:+.1f}$ [${gap['ci95'][0]:+.1f}$, ${gap['ci95'][1]:+.1f}$]}}
\\newcommand{{\\AnchorTokensPerRound}}{{{anchor_tok_str}}}
\\newcommand{{\\FailureTaxonomy}}{{{tax['by_domain_percent'].get('TTL', 0)}\\% test-time learning, {tax['by_domain_percent'].get('SF', 0)}\\% selective forgetting ({tax['sf_split']['FC-MH']}/{tax['sf_split']['FC-SH']} FC-MH/FC-SH questions), {tax['by_domain_percent'].get('AR', 0)}\\% long-context reading, {tax['by_domain_percent'].get('LRU', 0)}\\% long-range understanding}}
"""
    macro_path = os.path.join(PAPER, "figures", "RECOMPUTED_MACROS.tex")
    with open(macro_path, "w") as fh:
        fh.write(macros)
    print(f"\nwrote {macro_path}")
    print(f"wrote {os.path.join(os.path.dirname(__file__), 'recomputed_values.json')}")


if __name__ == "__main__":
    main()
