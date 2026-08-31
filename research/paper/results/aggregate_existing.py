"""Build the paper's machine-readable evidence ledger from frozen run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def source_meta(relpath: str) -> dict:
    raw = (ROOT / relpath).read_bytes()
    return {"path": relpath, "sha256": hashlib.sha256(raw).hexdigest()}


def load(relpath: str) -> tuple[dict, dict]:
    return json.loads((ROOT / relpath).read_text()), source_meta(relpath)


def pct(value: float) -> float:
    return round(100.0 * value, 4)


def main() -> dict:
    locomo_original_path = ".benchmark_runs/locomo-full-quality-v1/summary.baseline.json"
    locomo_path = (
        ".benchmark_runs/locomo-full-quality-v1/"
        "judge_summary.kimi-agent-direct-qwen37-full-thinking-off."
        "qwen37-mem0-current-exact-direct.json"
    )
    locomo_kimi_path = (
        ".benchmark_runs/locomo-full-quality-v1/"
        "judge_comparison.kimik3-mem0-current-exact-direct-v2-nothink.json"
    )
    longmem_path = (
        ".benchmark_runs/longmemeval-source-v24-full500/"
        "judge_summary.kimi-agent-direct-longmem500-memory-v24-full."
        "qwen37-longmem-official-memory-v24-full500.json"
    )
    locomo_plus_path = (
        ".benchmark_runs/locomo-plus-cognitive-qwen35-cues-full-v1/"
        "judge_summary.kimi-agent-direct-locomo-plus-qwen37-qwen35-cues-full-v1."
        "qwen37-locomo-plus-official-qwen35-cues-full-v1-final2.json"
    )
    channel_path = (
        ".benchmark_runs/locomo-full-quality-v1/"
        "channel_policy_replay.lexical-vs-semantic-independent-v1.summary.json"
    )
    fused_path = (
        ".benchmark_runs/locomo-full-quality-v1/"
        "channel_policy_replay.lexical-vs-fused-independent-v1.summary.json"
    )
    neighbor_path = (
        ".benchmark_runs/locomo-full-quality-v1/"
        "retrieval_ablation.source-v2-neighbor-expansion.wrong.summary.json"
    )
    depth_invalid_path = (
        ".benchmark_runs/locomo-full-quality-v1/"
        "channel_policy_replay.depth10-vs20-v1.summary.json"
    )
    depth_sweep_path = (
        ".benchmark_runs/locomo-full-quality-v1/"
        "channel_policy_replay.depth-sweep.prefix-monotonic-v1.summary.json"
    )
    meme_path = (
        ".benchmark_runs/meme-filler32k-qwen35-full100-k3-v1/"
        "meme_official_judge_kimik3_summary.json"
    )
    provenance_ablation_path = (
        ".benchmark_runs/locomo-full-quality-v1/"
        "channel_policy_replay.provenance_ablation.x7-v1.summary.json"
    )
    # K3-as-answerer diagnostic run: same 1540 protocol + dataset + library +
    # binary judge prompt, but Kimi-K3 substituted for GPT-4o-mini as BOTH
    # answerer and judge. Two judge tracks exist on the same K3 answers: a
    # same-model (kimi-k3) judge and an independent Qwen3.7-plus cross-judge.
    k3_answerer_kimik3_judge_path = (
        ".benchmark_runs/locomo-k3-agent-judge-diagnostic-v1/"
        "judge_summary.kimi-agent-direct-k3-agent-full-thinking-off-v1."
        "kimik3-agent-v1-legacy-fingerprint-diagnostic.json"
    )
    k3_answerer_qwen37_judge_path = (
        ".benchmark_runs/locomo-k3-agent-judge-diagnostic-v1/"
        "judge_summary.kimi-agent-direct-k3-agent-full-thinking-off-v1."
        "qwen37-k3-agent-v1-legacy-fingerprint-diagnostic.json"
    )
    k3_answerer_assessment_path = (
        ".benchmark_runs/locomo-k3-agent-judge-diagnostic-v1/"
        "assessment.kimik3-agent-v1-legacy-fingerprint-diagnostic.json"
    )

    locomo_original, s0 = load(locomo_original_path)
    locomo, s1 = load(locomo_path)
    locomo_kimi, s2 = load(locomo_kimi_path)
    longmem, s3 = load(longmem_path)
    locomo_plus, s4 = load(locomo_plus_path)
    lexical_semantic, s5 = load(channel_path)
    lexical_fused, s6 = load(fused_path)
    neighbors, s7 = load(neighbor_path)
    depth_invalid, s8 = load(depth_invalid_path)
    depth_sweep, s8b = load(depth_sweep_path)
    meme, s9 = load(meme_path)
    provenance_ablation, s10 = load(provenance_ablation_path)
    k3_judge, s_k3j = load(k3_answerer_kimik3_judge_path)
    qwen_judge, s_qj = load(k3_answerer_qwen37_judge_path)
    k3_assessment, s_k3a = load(k3_answerer_assessment_path)

    track_name = "kimi-agent-direct-qwen37-full-thinking-off"
    kimi_track = locomo_kimi["tracks"][track_name]
    nprof = neighbors["profiles"]

    def neighbor_row(key: str) -> dict:
        row = nprof[key]
        return {
            "recall_any_pct": pct(row["recall_any_rate"]),
            "recall_all_pct": pct(row["recall_all_rate"]),
            "mean_gold_recall_pct": pct(row["mean_gold_recall"]),
            "mean_response_bytes": round(row["average_page_bytes"], 2),
        }

    sources = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s8b, s9, s10, s_k3j, s_qj, s_k3a]
    for extra in [
        ".benchmark_data/locomo10.json",
        ".benchmark_data/longmemeval_s_cleaned.json",
        ".benchmark_data/locomo-plus/locomo_plus.json",
        ".benchmark_runs/locomo-full-quality-v1/judge_results.kimi-agent-direct-qwen37-full-thinking-off.qwen37-mem0-current-exact-direct.jsonl",
        ".benchmark_runs/locomo-full-quality-v1/judge_results.kimi-agent-direct-qwen37-full-thinking-off.kimik3-mem0-current-exact-direct-v2-nothink.jsonl",
        ".benchmark_runs/longmemeval-source-v24-full500/judge_results.kimi-agent-direct-longmem500-memory-v24-full.qwen37-longmem-official-memory-v24-full500.jsonl",
        ".benchmark_runs/locomo-plus-cognitive-qwen35-cues-full-v1/judge_results.kimi-agent-direct-locomo-plus-qwen37-qwen35-cues-full-v1.qwen37-locomo-plus-official-qwen35-cues-full-v1-final2.jsonl",
        ".benchmark_runs/locomo-full-quality-v1/channel_policy_replay.lexical-vs-semantic-independent-v1.jsonl",
        ".benchmark_runs/locomo-full-quality-v1/channel_policy_replay.lexical-vs-fused-independent-v1.jsonl",
        ".benchmark_runs/locomo-full-quality-v1/retrieval_ablation.source-v2-neighbor-expansion.wrong.jsonl",
        ".benchmark_runs/locomo-full-quality-v1/channel_policy_replay.depth-sweep.prefix-monotonic-v1.jsonl",
        ".benchmark_runs/locomo-full-quality-v1/channel_policy_replay.provenance_ablation.x7-v1.jsonl",
        ".benchmark_runs/locomo-k3-agent-judge-diagnostic-v1/results.kimi-agent-direct-k3-agent-full-thinking-off-v1.jsonl",
        ".benchmark_runs/locomo-k3-agent-judge-diagnostic-v1/judge_results.kimi-agent-direct-k3-agent-full-thinking-off-v1.kimik3-agent-v1-legacy-fingerprint-diagnostic.jsonl",
        ".benchmark_runs/locomo-k3-agent-judge-diagnostic-v1/judge_results.kimi-agent-direct-k3-agent-full-thinking-off-v1.qwen37-k3-agent-v1-legacy-fingerprint-diagnostic.jsonl",
    ]:
        sources.append(source_meta(extra))

    return {
        "schema_version": 2,
        "scope_note": (
            "Protocol-aligned internal and cross-judge results; not a strict external leaderboard. "
            "Retrieval diagnostics are associations unless their comparison changes only one factor."
        ),
        "main_results": {
            "locomo_1986_token_f1_pct": pct(locomo_original["overall"]),
            "locomo_1986_total": locomo_original["total"],
            "locomo_1540_qwen37_accuracy_pct": pct(locomo["overall"]),
            "locomo_1540_qwen37_correct": round(locomo["overall"] * locomo["total"]),
            "locomo_1540_kimik3_accuracy_pct": pct(kimi_track["overall"]),
            "locomo_1540_kimik3_correct": round(kimi_track["overall"] * kimi_track["total"]),
            "longmemeval_s_500_qwen37_accuracy_pct": pct(longmem["overall"]),
            "longmemeval_s_500_correct": round(longmem["overall"] * longmem["total"]),
            "longmemeval_by_type_pct": {
                key: pct(value["score"]) for key, value in longmem["by_type"].items()
            },
            "longmemeval_by_type_counts": {
                key: value["count"] for key, value in longmem["by_type"].items()
            },
            "locomo_1540_by_category_pct": {
                "single_hop": pct(locomo["by_type"]["4"]["score"]),
                "multi_hop": pct(locomo["by_type"]["1"]["score"]),
                "open_domain": pct(locomo["by_type"]["3"]["score"]),
                "temporal": pct(locomo["by_type"]["2"]["score"]),
            },
            "locomo_1540_by_category_counts": {
                "single_hop": locomo["by_type"]["4"]["count"],
                "multi_hop": locomo["by_type"]["1"]["count"],
                "open_domain": locomo["by_type"]["3"]["count"],
                "temporal": locomo["by_type"]["2"]["count"],
            },
            "locomo_1540_kimik3_by_category_pct": {
                "single_hop": pct(kimi_track["by_type"]["4"]["score"]),
                "multi_hop": pct(kimi_track["by_type"]["1"]["score"]),
                "open_domain": pct(kimi_track["by_type"]["3"]["score"]),
                "temporal": pct(kimi_track["by_type"]["2"]["score"]),
            },
            # K3-as-answerer run: same 1540 protocol / dataset / library / binary
            # judge prompt, but Kimi-K3 replaces GPT-4o-mini as BOTH answerer and
            # judge (the user's "all GPT->K3" substitution). Two judge tracks on
            # the same K3 answers: same-model (kimi-k3) judge and an independent
            # Qwen3.7-plus cross-judge. See the integrity caveat below.
            "locomo_1540_k3_answerer_accuracy_pct": pct(k3_judge["overall"]),
            "locomo_1540_k3_answerer_correct": round(k3_judge["overall"] * k3_judge["total"]),
            "locomo_1540_k3_answerer_total": k3_judge["total"],
            "locomo_1540_k3_answerer_by_category_pct": {
                "single_hop": pct(k3_judge["by_type"]["4"]["score"]),
                "multi_hop": pct(k3_judge["by_type"]["1"]["score"]),
                "open_domain": pct(k3_judge["by_type"]["3"]["score"]),
                "temporal": pct(k3_judge["by_type"]["2"]["score"]),
            },
            "locomo_1540_k3_answerer_qwen37_judge_accuracy_pct": pct(qwen_judge["overall"]),
            "locomo_1540_k3_answerer_qwen37_judge_correct": round(
                qwen_judge["overall"] * qwen_judge["total"]
            ),
            "locomo_1540_k3_answerer_qwen37_judge_by_category_pct": {
                "single_hop": pct(qwen_judge["by_type"]["4"]["score"]),
                "multi_hop": pct(qwen_judge["by_type"]["1"]["score"]),
                "open_domain": pct(qwen_judge["by_type"]["3"]["score"]),
                "temporal": pct(qwen_judge["by_type"]["2"]["score"]),
            },
            "locomo_1540_k3_answerer_cross_judge_agreement": {
                "agreement": round(
                    k3_assessment["cross_judge_agreement"]["agreement"], 4
                ),
                "both_correct": k3_assessment["cross_judge_agreement"]["both_correct"],
                "qwen37_only_correct": k3_assessment["cross_judge_agreement"][
                    "qwen37_only_correct"
                ],
                "kimi_k3_only_correct": k3_assessment["cross_judge_agreement"][
                    "kimi_k3_only_correct"
                ],
                "both_wrong": k3_assessment["cross_judge_agreement"]["both_wrong"],
            },
            "locomo_1540_k3_answerer_paired_vs_qwen37_answerer": {
                "repairs": k3_assessment["paired_comparison"]["repairs"],
                "regressions": k3_assessment["paired_comparison"]["regressions"],
                "net": k3_assessment["paired_comparison"]["net"],
                "accuracy_delta_points": round(
                    k3_assessment["paired_comparison"]["accuracy_delta_points"], 4
                ),
            },
            "locomo_1540_k3_answerer_roles": {
                "answerer": "kimi-k3",
                "judge_primary": "kimi-k3 (same-model)",
                "judge_independent": "qwen3.7-plus cross-judge",
                "retrieval_model": "kimi-k3",
                "metric": (
                    "binary CORRECT/WRONG accuracy (Mem0's J methodology: "
                    "GPT-4o-mini answerer+judge replaced by K3 answerer+judge)"
                ),
                "prompt_repository": "mem0ai/memory-benchmarks",
                "prompt_commit": "4b61c5d31b9c668a12b4f5e78064248a02c82d2b",
                "prompt_mode": "unified-no-evidence-exact",
                "category_mapping_note": (
                    "Local by_type codes map to Mem0's LoCoMo categories by question "
                    "count: type4=Single-hop(841), type1=Multi-hop(282), "
                    "type3=Open-domain(96), type2=Temporal(321). Total 1540."
                ),
                "integrity_caveat": (
                    "Diagnostic cross-judge run: answer rows are unchanged from the "
                    "locomo-full-quality-v1 source run (answer_rows_unchanged=true, "
                    "sha 76af5772...) and the per-item JSONL carries runtime_code_sha256 "
                    "on all 2063 rows, but the run is flagged formal_claim_allowed=false "
                    "because 1529 historical answer rows predate runtime-code "
                    "fingerprinting in the legacy source run. Reported as a diagnostic "
                    "cross-judge result, not a fingerprint-complete formal result."
                ),
                "run_dir": ".benchmark_runs/locomo-k3-agent-judge-diagnostic-v1",
                "library_snapshot_sha256": k3_judge["library_snapshot_sha256"],
                "single_run_note": (
                    "Single run, vs Mem0's published J as mean+/-std over 10 runs."
                ),
            },
            "locomo_1540_roles": {
                "answerer": "qwen3.7-plus",
                "judge_qwen37": "qwen3.7-plus, exact prompt",
                "judge_kimik3": "kimi-k3 cross-judge",
                "metric": "binary CORRECT/WRONG accuracy (same J methodology as Mem0)",
                "prompt_repository": "mem0ai/memory-benchmarks",
                "prompt_commit": "4b61c5d31b9c668a12b4f5e78064248a02c82d2b",
                "category_mapping_note": (
                    "Local by_type codes map to Mem0's LoCoMo categories by question "
                    "count: type4=Single-hop(841), type1=Multi-hop(282), "
                    "type3=Open-domain(96), type2=Temporal(321). Total 1540."
                ),
            },
            "roles": {
                "locomo_1540_qwen37": {
                    "answerer": "qwen3.7-plus",
                    "judge": "qwen3.7-plus",
                    "prompt_repository": "mem0ai/memory-benchmarks",
                    "prompt_commit": "4b61c5d31b9c668a12b4f5e78064248a02c82d2b",
                },
                "locomo_1540_kimik3": {
                    "answerer": "qwen3.7-plus",
                    "judge": "kimi-k3",
                    "prompt_repository": "mem0ai/memory-benchmarks",
                    "prompt_commit": "4b61c5d31b9c668a12b4f5e78064248a02c82d2b",
                },
                "locomo_1540_k3_answerer": {
                    "answerer": "kimi-k3",
                    "judge": "kimi-k3 (same-model) + qwen3.7-plus cross-judge",
                    "retrieval_model": "kimi-k3",
                    "prompt_repository": "mem0ai/memory-benchmarks",
                    "prompt_commit": "4b61c5d31b9c668a12b4f5e78064248a02c82d2b",
                    "prompt_mode": "unified-no-evidence-exact",
                    "run_dir": ".benchmark_runs/locomo-k3-agent-judge-diagnostic-v1",
                },
                "longmemeval_s_500": {
                    "answerer": "qwen3.7-plus",
                    "judge": "qwen3.7-plus",
                    "prompt_repository": "xiaowu0162/LongMemEval",
                    "prompt_mode": "official-compatible-prompt",
                },
            },
            "locomo_plus_401_qwen37_accuracy_pct": pct(locomo_plus["overall"]),
            "locomo_plus_401_correct": round(locomo_plus["overall"] * locomo_plus["total"]),
        },
        "channel_diagnostic_pct": {
            "population": "210-question development/gate set",
            "causal_scope": "Channel comparison; not a held-out final QA result.",
            "lexical": {
                "recall_any": pct(lexical_semantic["baseline_primary_recall_any"]),
                "recall_all": pct(lexical_semantic["baseline_primary_recall_all"]),
            },
            "semantic": {
                "recall_any": pct(lexical_semantic["candidate_primary_recall_any"]),
                "recall_all": pct(lexical_semantic["candidate_primary_recall_all"]),
            },
            "fused": {
                "recall_any": pct(lexical_fused["candidate_primary_recall_any"]),
                "recall_all": pct(lexical_fused["candidate_primary_recall_all"]),
            },
            "counts": {
                "questions": lexical_semantic["questions"],
                "lexical_recall_any": round(lexical_semantic["baseline_primary_recall_any"] * lexical_semantic["questions"]),
                "lexical_recall_all": round(lexical_semantic["baseline_primary_recall_all"] * lexical_semantic["questions"]),
                "semantic_recall_any": round(lexical_semantic["candidate_primary_recall_any"] * lexical_semantic["questions"]),
                "semantic_recall_all": round(lexical_semantic["candidate_primary_recall_all"] * lexical_semantic["questions"]),
                "fused_recall_any": round(lexical_fused["candidate_primary_recall_any"] * lexical_fused["questions"]),
                "fused_recall_all": round(lexical_fused["candidate_primary_recall_all"] * lexical_fused["questions"]),
            },
        },
        "source_expansion_diagnostic": {
            "population": "105 cases selected from prior errors",
            "causal_scope": (
                "The lexical-semantic-span and neighbor rows keep channel composition fixed; the sample is "
                "selected from prior errors and remains a development diagnostic rather than a held-out test."
            ),
            "lexical_span": neighbor_row("lexical-span"),
            "lexical_semantic_span": neighbor_row("lexical-semantic-span"),
            "neighbors_1": neighbor_row("lexical-semantic-span-neighbors1"),
            "neighbors_2": neighbor_row("lexical-semantic-span-neighbors2"),
            "legacy_context_3": neighbor_row("lexical-semantic-context3-legacy"),
            "counts": {
                key: {
                    "questions": nprof[key]["questions"],
                    "recall_any": nprof[key]["recall_any"],
                    "recall_all": nprof[key]["recall_all"],
                }
                for key in [
                    "lexical-semantic-span",
                    "lexical-semantic-span-neighbors1",
                    "lexical-semantic-span-neighbors2",
                    "lexical-semantic-context3-legacy",
                ]
            },
        },
        "depth_diagnostic": {
            "population": "210-question development/gate set",
            "valid_prefix_ablation": depth_sweep["prefix_monotonic_passed"],
            "invariant_violations": depth_sweep["prefix_monotonic_violations"],
            "interpretation": (
                "Retrieval is run once per reconstructed agent query at max depth "
                "(candidate_k=20) and OR-fused by best rank per turn; each k-slice is a "
                "prefix of one fixed ranking, so prefix monotonicity holds by "
                "construction (0 violations). The earlier depth10-vs20 replay changed "
                "primary result sets on every question and is retained below as a "
                "negative integrity finding, not a clean ablation."
            ),
            "method": depth_sweep["method"],
            "embedding_model": depth_sweep["embedding_model"],
            "ks": depth_sweep["ks"],
            "per_k": {
                k: {
                    "n": v["n"],
                    "recall_any_pct": pct(v["recall_any_mean"]),
                    "recall_any_ci_low_pct": pct(v["recall_any_ci_low"]),
                    "recall_any_ci_high_pct": pct(v["recall_any_ci_high"]),
                    "recall_all_pct": pct(v["recall_all_mean"]),
                    "recall_all_ci_low_pct": pct(v["recall_all_ci_low"]),
                    "recall_all_ci_high_pct": pct(v["recall_all_ci_high"]),
                    "mean_evidence_payload_bytes": round(v["mean_evidence_payload_bytes"], 2),
                }
                for k, v in depth_sweep["per_k"].items()
            },
            "bootstrap_resamples": depth_sweep["bootstrap_resamples"],
            "ci_level": depth_sweep["ci_level"],
            "offline": depth_sweep["offline"],
            # Backward-compatible depth_10/depth_20 aliases (now drawn from the
            # prefix-monotonic sweep at k=10 and k=20, not the invalid replay).
            "depth_10": {
                "recall_any_pct": pct(depth_sweep["per_k"]["10"]["recall_any_mean"]),
                "recall_all_pct": pct(depth_sweep["per_k"]["10"]["recall_all_mean"]),
                "mean_evidence_payload_bytes": round(depth_sweep["per_k"]["10"]["mean_evidence_payload_bytes"], 2),
            },
            "depth_20": {
                "recall_any_pct": pct(depth_sweep["per_k"]["20"]["recall_any_mean"]),
                "recall_all_pct": pct(depth_sweep["per_k"]["20"]["recall_all_mean"]),
                "mean_evidence_payload_bytes": round(depth_sweep["per_k"]["20"]["mean_evidence_payload_bytes"], 2),
            },
            "retained_invalid_replay": {
                "primary_result_invariant_passed": depth_invalid["primary_result_invariant_passed"],
                "primary_result_invariant_violations": depth_invalid["primary_result_invariant_violations"],
                "interpretation": (
                    "The invalid depth10-vs20 replay changed primary result sets on "
                    "all 210 questions; retained in the evidence ledger as an "
                    "integrity finding, not a clean ablation."
                ),
            },
        },
        "meme_limitation": {
            "episodes": meme["episodes"],
            "after_raw_accuracy_pct": pct(meme["after_raw"]["accuracy"]),
            "after_raw_correct": meme["after_raw"]["pass"],
            "after_raw_total": meme["after_raw"]["total"],
            "task_accuracy_pct": {
                "ER": pct(meme["per_task"]["ER"]["raw_accuracy"]),
                "Agg": pct(meme["per_task"]["Agg"]["raw_accuracy"]),
                "Tr": pct(meme["per_task"]["Tr"]["raw_accuracy"]),
                "Del": pct(meme["per_task"]["Del"]["real_accuracy"]),
                "Cas_real": pct(meme["per_task"]["Cas"]["real_accuracy"]),
                "Abs_real": pct(meme["per_task"]["Abs"]["real_accuracy"]),
            },
        },
        "provenance_ablation_x7": {
            "population": "210-question development/gate set",
            "offline": provenance_ablation["offline"],
            "embedding_model": provenance_ablation["embedding_model"],
            "method": provenance_ablation["method"],
            "questions": provenance_ablation["questions"],
            "ks": provenance_ablation["ks"],
            "bootstrap_resamples": provenance_ablation["bootstrap_resamples"],
            "ci_level": provenance_ablation["ci_level"],
            "flag_validation": {
                "ran": provenance_ablation["flag_validation"].get("ran"),
                "gate_on_raised": provenance_ablation["flag_validation"].get("gate_on_raised"),
                "gate_relaxed_accepted": provenance_ablation["flag_validation"].get("gate_relaxed_accepted"),
                "plumbing_ok": provenance_ablation["flag_validation"].get("plumbing_ok"),
            },
            "requires_llm_for_qa_columns": provenance_ablation["requires_llm_for_qa_columns"],
            "interpretation": provenance_ablation["interpretation"],
            "arms": {
                name: {
                    "description": arm["description"],
                    "channels": arm["channels"],
                    "gate_mode": arm["gate_mode"],
                    "retrieval_identical_to": arm.get("retrieval_identical_to"),
                    "per_k": {
                        k: {
                            "n": v["n"],
                            "recall_any_pct": pct(v["recall_any_mean"]),
                            "recall_any_ci_low_pct": pct(v["recall_any_ci_low"]),
                            "recall_any_ci_high_pct": pct(v["recall_any_ci_high"]),
                            "recall_all_pct": pct(v["recall_all_mean"]),
                            "recall_all_ci_low_pct": pct(v["recall_all_ci_low"]),
                            "recall_all_ci_high_pct": pct(v["recall_all_ci_high"]),
                            "mean_evidence_payload_bytes": round(v["mean_evidence_payload_bytes"], 2),
                        }
                        for k, v in arm["per_k"].items()
                    },
                }
                for name, arm in provenance_ablation["arms"].items()
            },
        },
        "published_baselines": {
            "scope_note": (
                "Published numbers reproduced verbatim from the cited papers and used "
                "as contextual baselines only. They are NOT strict head-to-head results: "
                "Deep-Dream's answerer (Qwen3.7-plus), judges, prompts, and memory "
                "implementation differ from both Mem0 (GPT-4o-mini answerer/judge, "
                "rubric J) and Zep/Graphiti (GPT-4o/4o-mini answerers). Per the paper's "
                "protocol-aware stance these are cross-protocol references, not a shared "
                "leaderboard. Cited directly rather than re-run, as authorized."
            ),
            "mem0_locomo": {
                "source": "Chhikara et al., 2025 (arXiv:2504.19413), Table 1",
                "benchmark": "LoCoMo",
                "answerer": "GPT-4o-mini",
                "metrics": ["F1", "BLEU-1", "J (GPT-4o-mini rubric, mean+/-std over 10 runs)"],
                "categories": {
                    "single_hop": 841,
                    "multi_hop": 282,
                    "open_domain": 96,
                    "temporal": 321,
                },
                "weighted_j": {"mem0": 62.14, "mem0_g": 61.36},
                "rows": [
                    {"method": "LoCoMo",       "f1": [25.02,12.04,40.36,18.41], "bleu1": [19.75,11.16,29.05,14.77], "j": None},
                    {"method": "ReadAgent",    "f1": [9.15,5.31,9.67,12.60],   "bleu1": [6.48,5.12,7.66,8.87],    "j": None},
                    {"method": "MemoryBank",   "f1": [5.00,5.56,6.61,9.68],     "bleu1": [4.77,5.94,5.16,6.99],    "j": None},
                    {"method": "MemGPT",       "f1": [26.65,9.15,41.04,25.52],  "bleu1": [17.72,7.44,34.34,19.44], "j": None},
                    {"method": "A-Mem",        "f1": [27.02,12.14,44.65,45.85], "bleu1": [20.09,12.00,37.06,36.67],"j": None},
                    {"method": "A-Mem*",       "f1": [20.76,9.22,33.34,35.40],  "bleu1": [14.90,8.81,27.58,31.08], "j": [39.79,18.85,54.05,49.91], "j_std": [0.38,0.31,0.22,0.31]},
                    {"method": "LangMem",      "f1": [35.51,26.04,40.91,30.75], "bleu1": [26.86,22.32,33.63,25.84],"j": [62.23,47.92,71.12,23.43], "j_std": [0.75,0.47,0.20,0.39]},
                    {"method": "Zep",          "f1": [35.74,19.37,49.56,42.00],  "bleu1": [23.30,14.82,38.92,34.53],"j": [61.70,41.35,76.60,49.31], "j_std": [0.32,0.48,0.13,0.50]},
                    {"method": "OpenAI",       "f1": [34.30,20.09,39.31,14.04],  "bleu1": [23.72,15.42,31.16,11.25],"j": [63.79,42.92,62.29,21.71], "j_std": [0.46,0.63,0.12,0.20]},
                    {"method": "Mem0",        "f1": [38.72,28.64,47.65,48.93],  "bleu1": [27.13,21.58,38.72,40.51],"j": [67.13,51.15,72.93,55.51], "j_std": [0.65,0.31,0.11,0.34]},
                    {"method": "Mem0^g",      "f1": [38.09,24.32,49.27,51.55],  "bleu1": [26.03,18.82,40.30,40.28],"j": [65.71,47.19,75.71,58.13], "j_std": [0.45,0.67,0.21,0.44]},
                ],
                "category_order": ["single_hop", "multi_hop", "open_domain", "temporal"],
            },
            "zep_dmr": {
                "source": "Rasmussen et al., 2025 (arXiv:2501.13956), DMR table",
                "benchmark": "Deep Memory Retrieval (DMR)",
                "description": "500 multi-session conversations (5 sessions each, <=12 messages/session), one single-hop fact-retrieval question per conversation.",
                "rows": [
                    {"method": "Recursive Summarization",     "answerer": "GPT-4-turbo",  "accuracy_pct": 35.3},
                    {"method": "Conversation Summaries",      "answerer": "GPT-4-turbo",  "accuracy_pct": 78.6},
                    {"method": "MemGPT",                       "answerer": "GPT-4-turbo",  "accuracy_pct": 93.4},
                    {"method": "Full-conversation",            "answerer": "GPT-4-turbo",  "accuracy_pct": 94.4},
                    {"method": "Zep",                          "answerer": "GPT-4-turbo",  "accuracy_pct": 94.8},
                    {"method": "Conversation Summaries",      "answerer": "GPT-4o-mini",  "accuracy_pct": 88.0},
                    {"method": "Full-conversation",            "answerer": "GPT-4o-mini",  "accuracy_pct": 98.0},
                    {"method": "Zep",                          "answerer": "GPT-4o-mini",  "accuracy_pct": 98.2},
                ],
            },
            "zep_longmemeval": {
                "source": "Rasmussen et al., 2025 (arXiv:2501.13956), LongMemEval table",
                "benchmark": "LongMemEval",
                "main": [
                    {"answerer": "GPT-4o-mini", "system": "Full-context", "accuracy_pct": 55.4, "latency_s": 31.3, "avg_context_tokens": "115k"},
                    {"answerer": "GPT-4o-mini", "system": "Zep",          "accuracy_pct": 63.8, "latency_s": 3.20, "avg_context_tokens": "1.6k"},
                    {"answerer": "GPT-4o",      "system": "Full-context", "accuracy_pct": 60.2, "latency_s": 28.9, "avg_context_tokens": "115k"},
                    {"answerer": "GPT-4o",      "system": "Zep",          "accuracy_pct": 71.2, "latency_s": 2.58, "avg_context_tokens": "1.6k"},
                ],
                "category_breakdown_full_to_zep": {
                    "single-session-preference": {"gpt4o_mini": [30.0, 53.3], "gpt4o": [20.0, 56.7]},
                    "single-session-assistant":  {"gpt4o_mini": [81.8, 75.0], "gpt4o": [94.6, 80.4]},
                    "temporal-reasoning":        {"gpt4o_mini": [36.5, 54.1], "gpt4o": [45.1, 62.4]},
                    "multi-session":             {"gpt4o_mini": [40.6, 47.4], "gpt4o": [44.3, 57.9]},
                    "knowledge-update":          {"gpt4o_mini": [76.9, 74.4], "gpt4o": [78.2, 83.3]},
                    "single-session-user":       {"gpt4o_mini": [81.4, 92.9], "gpt4o": [81.4, 92.9]},
                },
            },
        },
        "sources": sources,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(main(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
