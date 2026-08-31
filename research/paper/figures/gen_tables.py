"""Generate the two compact main-text tables from the evidence ledger."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results" / "benchmark_summary.json").read_text())
OUT = ROOT / "figures"


def write(name: str, text: str) -> None:
    (OUT / name).write_text(text.strip() + "\n")


main = DATA["main_results"]
write(
    "TABLE_main_results.tex",
    rf"""
\begin{{table*}}[t]
\centering
\caption{{Deep-Dream results under explicitly labeled local protocols. The K3-as-answerer row substitutes Kimi-K3 for GPT-4o-mini as both answerer and judge on Mem0's $J$ protocol (same 1,540-question subset); other rows use the local Qwen/Kimi cross-judge setup. Binary $J$ is not numerically comparable to LoCoMo token-F1.}}
\label{{tab:main-results}}
\small
\setlength{{\tabcolsep}}{{4.5pt}}
\begin{{tabular}}{{lrrll}}
\toprule
Benchmark & $N$ & Result & Answerer & Judge / protocol \\
\midrule
LoCoMo original & {main['locomo_1986_total']} & {main['locomo_1986_token_f1_pct']:.2f} token-F1 & Qwen3.6-27B & quality-v1 baseline \\
LoCoMo Mem0-compatible & 1540 & {main['locomo_1540_qwen37_accuracy_pct']:.2f}\% (1435/1540) & Qwen3.7-plus & Qwen3.7-plus, exact prompt \\
LoCoMo Mem0-compatible & 1540 & {main['locomo_1540_kimik3_accuracy_pct']:.2f}\% (1441/1540) & Qwen3.7-plus & Kimi-K3 cross-judge \\
LoCoMo Mem0-compatible & 1540 & \textbf{{{main['locomo_1540_k3_answerer_accuracy_pct']:.2f}\% ({main['locomo_1540_k3_answerer_correct']}/1540)}} & \textbf{{Kimi-K3}} & \textbf{{K3 answerer$+$judge (Mem0 $J$, GPT$\to$K3)}} \\
LongMemEval-S & 500 & {main['longmemeval_s_500_qwen37_accuracy_pct']:.2f}\% (423/500) & Qwen3.7-plus & Qwen3.7-plus cross-judge \\
LoCoMo-Plus cognitive & 401 & {main['locomo_plus_401_qwen37_accuracy_pct']:.2f}\% (259/401) & Qwen3.7-plus & Qwen3.7-plus cross-judge \\
\bottomrule
\end{{tabular}}
\end{{table*}}
""",
)

diag = DATA["channel_diagnostic_pct"]
exp = DATA["source_expansion_diagnostic"]
write(
    "TABLE_diagnostics.tex",
    rf"""
\begin{{table*}}[t]
\centering
\caption{{Diagnostic retrieval results. The channel rows use a 210-question development/gate set. The expansion rows use 105 cases selected from prior errors; lexical+semantic channels are held fixed across the span and neighbor rows.}}
\label{{tab:diagnostics}}
\small
\begin{{tabular}}{{lrrr}}
\toprule
Configuration & Recall-any & Recall-all & Mean bytes \\
\midrule
Lexical & {diag['lexical']['recall_any']:.2f}\% & {diag['lexical']['recall_all']:.2f}\% & --- \\
Semantic & {diag['semantic']['recall_any']:.2f}\% & {diag['semantic']['recall_all']:.2f}\% & --- \\
Fused & {diag['fused']['recall_any']:.2f}\% & {diag['fused']['recall_all']:.2f}\% & --- \\
\midrule
Lexical+semantic span & {exp['lexical_semantic_span']['recall_any_pct']:.2f}\% & {exp['lexical_semantic_span']['recall_all_pct']:.2f}\% & {exp['lexical_semantic_span']['mean_response_bytes']:.0f} \\
Span + $\pm1$ turn & {exp['neighbors_1']['recall_any_pct']:.2f}\% & {exp['neighbors_1']['recall_all_pct']:.2f}\% & {exp['neighbors_1']['mean_response_bytes']:.0f} \\
Span + $\pm2$ turns & {exp['neighbors_2']['recall_any_pct']:.2f}\% & {exp['neighbors_2']['recall_all_pct']:.2f}\% & {exp['neighbors_2']['mean_response_bytes']:.0f} \\
\bottomrule
\end{{tabular}}
\end{{table*}}
""",
)

# X7 provenance ablation (offline retrieval replay). Arms 1-4 are the clean
# channel ablation shown here; arms 5/6 reuse arm-4 retrieval and their agent
# columns are pending the LLM run, so they are described in the caption rather
# than re-printed as independent (identical) measurements.
x7 = DATA["provenance_ablation_x7"]
_arm_rows = [
    ("arm1_source_lexical", "1", "raw-doc + episode-bm25"),
    ("arm2_source_semantic", "2", "+ semantic-provenance"),
    ("arm3_source_neighbors", "3", "+ graph-neighbor"),
    ("arm4_source_relations", "4", "+ relation-evidence (full 5)"),
]
_x7_body = []
for _key, _num, _chans in _arm_rows:
    _pk = x7["arms"][_key]["per_k"]["10"]
    _x7_body.append(
        rf"{_num} & {_chans} & "
        rf"{_pk['recall_any_pct']:.2f} [{_pk['recall_any_ci_low_pct']:.2f}, {_pk['recall_any_ci_high_pct']:.2f}] & "
        rf"{_pk['recall_all_pct']:.2f} [{_pk['recall_all_ci_low_pct']:.2f}, {_pk['recall_all_ci_high_pct']:.2f}] & "
        rf"{_pk['mean_evidence_payload_bytes']:.0f} \\"
    )
_x7_body_text = "\n".join(_x7_body)
_x7_plumbing = str(x7["flag_validation"].get("plumbing_ok")).lower()
write(
    "TABLE_provenance_ablation.tex",
    rf"""
\begin{{table*}}[t]
\centering
\caption{{X7 provenance ablation, offline retrieval replay on the 210-question gate set. Arms 1--4 form a clean channel ablation: only \texttt{{enabled\_channels}} differs, so any recall delta is attributable to that channel set. Arms 5 (evidence-bound agent, gate ON) and 6 (unsurfaced evidence, gate relaxed) reuse arm~4's five-channel ranking and are retrieval-identical to arm~4; their distinguishing columns (QA accuracy, unsupported-answer rate, mean evidence tokens) require the LLM agent run and are pending. The evidence-gate plumbing was proved offline: with the gate ON, submitting an unsurfaced turn raises; with the gate relaxed the same submission is accepted (\texttt{{plumbing\_ok}}={_x7_plumbing}). Bootstrap CIs: 2{{,}}000 resamples, 95\%.}}
\label{{tab:provenance-ablation}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{llrrr}}
\toprule
Arm & Channels (additive) & Recall-any@10 [95\% CI] & Recall-all@10 [95\% CI] & Mean bytes@10 \\
\midrule
{_x7_body_text}
\bottomrule
\end{{tabular}}
\end{{table*}}
""",
)

# === Published baseline tables (cited directly, not re-run) =====================
# User instruction: Mem0 and Zep/Graphiti published result tables/parameters can
# be used directly here; no re-run. These are cross-protocol contextual baselines,
# clearly labeled as such (different answerer/judge/prompt/implementation).
pb = DATA["published_baselines"]

# --- Mem0 LoCoMo Table 1 (F1 / BLEU-1 / J) ---
_m = pb["mem0_locomo"]
# LoCoMo category order used by Mem0: Single-hop / Multi-hop / Open-domain / Temporal.
_m_cat_order = ["single_hop", "multi_hop", "open_domain", "temporal"]
# Deep-Dream J column: K3 substituted for GPT-4o-mini as BOTH answerer and judge
# (the user's "all GPT->K3" substitution), so Deep-Dream's binary CORRECT/WRONG
# accuracy sits in the SAME J column as Mem0's published J. Two rows: the
# primary (same-model K3 judge, matching Mem0's answerer=judge setup) and the
# daggered (independent Qwen3.7-plus cross-judge on the same K3 answers).
_dd_k3_ans = main["locomo_1540_k3_answerer_by_category_pct"]
_dd_k3_qwen = main["locomo_1540_k3_answerer_qwen37_judge_by_category_pct"]
_dd_k3_ans_vals = [_dd_k3_ans[k] for k in _m_cat_order]
_dd_k3_qwen_vals = [_dd_k3_qwen[k] for k in _m_cat_order]
_dd_k3_j = " / ".join(f"{v:.2f}" for v in _dd_k3_ans_vals)
_dd_k3_qwen_j = " / ".join(f"{v:.2f}" for v in _dd_k3_qwen_vals)
_dd_k3_agg = main["locomo_1540_k3_answerer_accuracy_pct"]
_dd_k3_qwen_agg = main["locomo_1540_k3_answerer_qwen37_judge_accuracy_pct"]
_xj_pct = main["locomo_1540_k3_answerer_cross_judge_agreement"]["agreement"] * 100

_m_rows = []
for _r in _m["rows"]:
    if _r.get("j") is None:
        _j = "--- / --- / --- / ---"
    else:
        _j = " / ".join(f"{v:.2f}" for v in _r["j"])
    # Superscript method suffixes (e.g. Mem0^g) must be wrapped in $...$ for LaTeX.
    _name = _r["method"]
    if "^" in _name:
        _name = "$" + _name + "$"
    # Single J column: Deep-Dream's K3-substituted binary accuracy is placed in
    # the SAME J column as Mem0's published J (K3 for GPT-4o-mini, same prompt).
    _m_rows.append(rf"{_name} & {_j} \\")
_m_rows.append(rf"\textbf{{Deep-Dream}} & {_dd_k3_j} \\")
_m_rows.append(rf"\textbf{{Deep-Dream}}$^\dagger$ & {_dd_k3_qwen_j} \\")
_m_rows_text = "\n".join(_m_rows)
_m_wj = _m["weighted_j"]
write(
    "TABLE_mem0_locomo_baselines.tex",
    rf"""
\begin{{table*}}[t]
\centering
\caption{{LoCoMo per-category $J$ (binary CORRECT/WRONG accuracy, LLM-judged). Published rows reproduced verbatim from \citet{{chhikara2025mem0}} (arXiv:2504.19413, Table~1): Mem0 reports mean$\pm$std over 10 runs with GPT-4o-mini as both answerer and judge (std omitted, all $\leq$0.75; appendix). \textbf{{Deep-Dream}} applies the \emph{{same}} $J$ methodology with Kimi-K3 substituted for GPT-4o-mini as both answerer and judge (single run): the primary row uses the same-model judge as Mem0; the $\dagger$ row judges the same K3 answers with an independent Qwen3.7-plus cross-judge (cross-judge agreement {_xj_pct:.2f}\%). Same 841/282/96/321 question populations (1{{,}}540 total, Mem0-compatible subset). \emph{{Caveat:}} the K3-answerer run is flagged \texttt{{formal\_claim\_allowed=false}}---1{{,}}529 historical answer rows predate runtime-code fingerprinting (answers unchanged, sha 76af5772; per-item JSONL carries \texttt{{runtime\_code\_sha256}}); reported as a diagnostic cross-judge result, not a fingerprint-complete formal result. Deterministic F1/BLEU-1 omitted for space (Table~\ref{{tab:mem0-f1-bleu}}). Weighted $J$: Mem0 $\approx${_m_wj['mem0']:.2f}, $Mem0^g$ $\approx${_m_wj['mem0_g']:.2f}. Deep-Dream agg.\ $J$: {_dd_k3_agg:.2f}\% / {_dd_k3_qwen_agg:.2f}\% ($\dagger$).}}
\label{{tab:mem0-baselines}}
\footnotesize
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{lc}}
\toprule
Method & $J$ (S/M/O/T) \\
\midrule
{_m_rows_text}
\bottomrule
\end{{tabular}}
\end{{table*}}
""",
)

# --- Appendix: Mem0 F1 / BLEU-1 (deterministic token-overlap metrics) ---
_mf_rows = []
for _r in _m["rows"]:
    _f1 = " / ".join(f"{v:.2f}" for v in _r["f1"])
    _bleu = " / ".join(f"{v:.2f}" for v in _r["bleu1"])
    _name = _r["method"]
    if "^" in _name:
        _name = "$" + _name + "$"
    _mf_rows.append(rf"{_name} & {_f1} & {_bleu} \\")
_mf_rows_text = "\n".join(_mf_rows)
write(
    "TABLE_mem0_f1_bleu_appendix.tex",
    rf"""
\begin{{table}}[t]
\centering
\caption{{Mem0 LoCoMo deterministic metrics, reproduced verbatim from \citet{{chhikara2025mem0}} (Table~1). F1 / BLEU-1 over the four categories (Single-hop / Multi-hop / Open-domain / Temporal). Deep-Dream does not compute token-overlap metrics (its metric is binary LLM-judge accuracy, Table~\ref{{tab:mem0-baselines}}), so no Deep-Dream column appears here; this table is included for completeness of the published reproduction only.}}
\label{{tab:mem0-f1-bleu}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lcc}}
\toprule
Method & F1 (S/M/O/T) & BLEU-1 (S/M/O/T) \\
\midrule
{_mf_rows_text}
\bottomrule
\end{{tabular}}
\end{{table}}
""",
)

# --- Zep DMR table ---
_d = pb["zep_dmr"]
_d_rows = []
for _r in _d["rows"]:
    _d_rows.append(rf"{_r['method']} & {_r['answerer']} & {_r['accuracy_pct']:.1f}\% \\")
_d_rows_text = "\n".join(_d_rows)
write(
    "TABLE_zep_dmr_baselines.tex",
    rf"""
\begin{{table}}[t]
\centering
\caption{{Published Deep Memory Retrieval (DMR) baselines reproduced verbatim from \citet{{rasmussen2025zep}} (arXiv:2501.13956). DMR has 500 multi-session conversations (5 sessions, $\leq$12 messages each) with one single-hop fact-retrieval question per conversation. Deep-Dream does not currently run DMR, so this table is a contextual reference only.}}
\label{{tab:zep-dmr}}
\small
\begin{{tabular}}{{llr}}
\toprule
Method & Answerer & Accuracy \\
\midrule
{_d_rows_text}
\bottomrule
\end{{tabular}}
\end{{table}}
""",
)

# --- Zep LongMemEval main + category breakdown ---
_l = pb["zep_longmemeval"]
_l_main_rows = []
for _r in _l["main"]:
    _l_main_rows.append(
        rf"{_r['answerer']} & {_r['system']} & {_r['accuracy_pct']:.1f}\% & {_r['latency_s']:.2f} & {_r['avg_context_tokens']} \\"
    )
# Deep-Dream LongMemEval-S-500 row: Qwen3.7-plus answerer+cross-judge, 84.60% aggregate.
_l_main_rows.append(
    rf"\textbf{{Deep-Dream}} & \textbf{{Qwen3.7-plus}} & \textbf{{84.6\%}} & --- & --- \\"
)
_l_main_text = "\n".join(_l_main_rows)
write(
    "TABLE_zep_longmemeval_baselines.tex",
    rf"""
\begin{{table}}[t]
\centering
\caption{{LongMemEval main results. Published rows (GPT-4o-mini / GPT-4o, Full-context vs.\ Zep) reproduced verbatim from \citet{{rasmussen2025zep}} (arXiv:2501.13956). The \textbf{{Deep-Dream}} row uses Qwen3.7-plus as both answerer and cross-judge---a different protocol; its accuracy (84.6\%) is placed beside the published numbers, not as a shared leaderboard. Latency/avg-context for Deep-Dream are not measured under the published protocol (---). The six-category breakdown is in Table~\ref{{tab:zep-longmemeval-cat}}.}}
\label{{tab:zep-longmemeval}}
\footnotesize
\begin{{tabular}}{{llrrl}}
\toprule
Answerer & System & Accuracy & Latency (s) & Avg.\ context \\
\midrule
{_l_main_text}
\bottomrule
\end{{tabular}}
\end{{table}}
""",
)

# --- Zep LongMemEval six-category breakdown (separate compact table) ---
_lme = main["longmemeval_by_type_pct"]
_l_cat_keys = ["single-session-preference", "single-session-assistant",
               "temporal-reasoning", "multi-session", "knowledge-update",
               "single-session-user"]
_l_cat_disp = {
    "single-session-preference": r"Sess.\ pref.",
    "single-session-assistant": r"Sess.\ asst.",
    "temporal-reasoning": "Temporal",
    "multi-session": "Multi-sess.",
    "knowledge-update": "Knowledge upd.",
    "single-session-user": r"Sess.\ user",
}
# One row per system; columns = the six categories. Published Zep rows show
# Full-context -> Zep (two values) but the Zep endpoint is the memory-system
# number; we list Zep's endpoint plus full-context baseline plus Deep-Dream.
def _zep_end(_cb, _key):
    """Zep endpoint value from the Full->Zep pair."""
    return _cb[_key][1]

def _full_end(_cb, _key):
    """Full-context baseline value from the Full->Zep pair."""
    return _cb[_key][0]

# Systems as rows, categories as columns -> 6 data columns, fits as a single-column table.
_sys_rows = []
# Full-context (GPT-4o-mini) row
_sys_rows.append(rf"Full-ctx (4o-mini) & " + " & ".join(
    f"{_full_end(_l['category_breakdown_full_to_zep'][k], 'gpt4o_mini'):.1f}" for k in _l_cat_keys) + r" \\")
_sys_rows.append(rf"Zep (4o-mini) & " + " & ".join(
    f"{_zep_end(_l['category_breakdown_full_to_zep'][k], 'gpt4o_mini'):.1f}" for k in _l_cat_keys) + r" \\")
_sys_rows.append(rf"Full-ctx (4o) & " + " & ".join(
    f"{_full_end(_l['category_breakdown_full_to_zep'][k], 'gpt4o'):.1f}" for k in _l_cat_keys) + r" \\")
_sys_rows.append(rf"Zep (4o) & " + " & ".join(
    f"{_zep_end(_l['category_breakdown_full_to_zep'][k], 'gpt4o'):.1f}" for k in _l_cat_keys) + r" \\")
_sys_rows.append(rf"\textbf{{Deep-Dream (Q.)}} & " + " & ".join(
    rf"\textbf{{{_lme[k]:.1f}}}" for k in _l_cat_keys) + r" \\")
_sys_rows_text = "\n".join(_sys_rows)
_cat_header = " & ".join(_l_cat_disp[k] for k in _l_cat_keys)
write(
    "TABLE_zep_longmemeval_cat.tex",
    rf"""
\begin{{table*}}[t]
\centering
\caption{{LongMemEval six-category breakdown (binary accuracy, \%). Published Full-context and Zep values reproduced verbatim from \citet{{rasmussen2025zep}} (arXiv:2501.13956); the \textbf{{Deep-Dream (Q.)}} row is Qwen3.7-plus answerer+cross-judge on the same six categories. Different answerer/judge, so this is a matched-granularity contextual reference, not a shared leaderboard. Sess.\ pref.\ is the hardest category for all systems.}}
\label{{tab:zep-longmemeval-cat}}
\footnotesize
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{lcccccc}}
\toprule
System & {_cat_header} \\
\midrule
{_sys_rows_text}
\bottomrule
\end{{tabular}}
\end{{table*}}
""",
)
