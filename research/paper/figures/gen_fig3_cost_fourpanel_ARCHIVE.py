"""Generate the four-panel cost/diagnostics figure.

Canvas is 5.5 x 3.5 in = the ICLR \\textwidth; the figure is included at
width=\\textwidth, so every font renders 1:1 (no LaTeX rescale).  Minimum
font floor: 6pt (asserted at the end).

All values are wired from frozen artifacts with loud assertions:
  (a) per-rung latency/tokens/calls  <- results/recomputed_values.json
  (b) per-doc ingest cost            <- results/evidence_ledger.json
      four_benchmark.engine_comparison.{mab.per_doc_intersection,lme.per_doc},
      cross-checked against the ledger's own calls_delta_pct
  (c) retrieval-budget frontier      <- ledger depth_diagnostic.per_k
  (d) evidence-channel ablation      <- ledger provenance_ablation_x7.arms
"""

import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import save

NAVY = "#1E4E8C"
BLUE = "#2F6FB2"
TEAL = "#238E8B"
ROSE = "#C14D5A"
ORANGE = "#D58A45"
INK = "#17202A"
MUTED = "#52606D"
GRID = "#D8E1EA"
PANEL = "#F8FBFE"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.2,
    "axes.labelsize": 7.2,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6.4,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    # paper_plot_style defaults to savefig.bbox="tight", which would crop the
    # canvas below the declared 5.5x3.5in; we keep the exact final size so
    # the \\textwidth include is a true 1:1 (no rescale, fonts render exactly)
    "savefig.bbox": None,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "results")


def load_inputs() -> tuple[dict, dict, dict]:
    with open(os.path.join(RESULT_DIR, "recomputed_values.json")) as fh:
        values = json.load(fh)
    with open(os.path.join(RESULT_DIR, "evidence_ledger.json")) as fh:
        ledger = json.load(fh)

    # ---- panel (b) wiring + assertions (hard-coded numbers are gone) ----
    ec = ledger["four_benchmark"]["engine_comparison"]
    mab = ec["mab"]["per_doc_intersection"]
    lme = ec["lme"]["per_doc"]
    cost_b = {
        "call_v1": [mab["v1"]["llm_calls_per_doc"], lme["v1"]["llm_calls_per_doc"]],
        "call_v2": [mab["v2"]["llm_calls_per_doc"], lme["v2"]["llm_calls_per_doc"]],
        "tok_v1": [mab["v1"]["tokens_per_doc"] / 1000, lme["v1"]["tokens_per_doc"] / 1000],
        "tok_v2": [mab["v2"]["tokens_per_doc"] / 1000, lme["v2"]["tokens_per_doc"] / 1000],
        "calls_delta_pct": [ec["mab"]["calls_delta_pct"], ec["lme"]["calls_delta_pct"]],
        "tokens_delta_pct": [ec["mab"]["tokens_delta_pct"], ec["lme"]["tokens_delta_pct"]],
    }
    for i, bench in enumerate(("MAB", "LME")):
        assert abs(100 * cost_b["call_v2"][i] / cost_b["call_v1"][i] - 100
                   - cost_b["calls_delta_pct"][i]) < 0.15, f"{bench} calls delta mismatch"
        assert abs(100 * cost_b["tok_v2"][i] / cost_b["tok_v1"][i] - 100
                   - cost_b["tokens_delta_pct"][i]) < 0.15, f"{bench} tokens delta mismatch"
    return values, ledger, cost_b


def style_axis(ax, accent):
    ax.set_facecolor(PANEL)
    for side in ax.spines:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(accent)
        ax.spines[side].set_linewidth(.65)
    ax.grid(axis="y", color=GRID, linewidth=.45, alpha=.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=2.2, width=.55, colors=MUTED, pad=2)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def panel_label(ax, label, title, accent):
    ax.text(.03, .96, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, fontweight="bold", color=accent)
    ax.text(.13, .96, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=7.2, fontweight="bold", color=INK)


def audit_min_font(fig, floor=6.0) -> float:
    sizes = [t.get_fontsize() for t in fig.findobj(mpl.text.Text) if t.get_text().strip()]
    smallest = min(sizes)
    assert smallest >= floor, f"min font {smallest:.2f}pt is below the {floor}pt floor"
    return smallest


def make_figure():
    values, ledger, cost_b = load_inputs()

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.5))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # (a) Measured answer latency: mean -> p95 interval per rung, with the
    # tokens/calls annotation that used to live in the old bar chart.
    panel_label(ax_a, "(a)", "Answer-path latency", BLUE)
    rung_labels = ["Direct", "Memory tool", "Source-grounded"]
    means, p95, tokens, calls = [], [], [], []
    for i in (1, 2, 3):
        c = values["rung_cost_v2"][f"rung{i}"]
        assert c["questions"] == 767, f"rung{i} is not the 767-question run"
        means.append(c["latency_seconds_mean"])
        p95.append(c["latency_seconds_p95"])
        tokens.append(c["tokens_per_question"] / 1000)
        calls.append(c["llm_calls_per_question"])
    y = np.arange(3)
    ax_a.hlines(y, means, p95, color=BLUE, linewidth=3.1, alpha=.35, zorder=2)
    ax_a.scatter(means, y, color=BLUE, s=22, zorder=3)
    ax_a.scatter(p95, y, color=ROSE, s=19, marker="D", zorder=3)
    for yy, m, hi, tok, call in zip(y, means, p95, tokens, calls):
        ax_a.text(hi + 5.0, yy + 0.06, f"{tok:.1f}k tok\n{call:.1f} calls",
                  va="center", ha="left", fontsize=6.0, color=INK, linespacing=1.25)
    # direct labels for the two marker types (cleaner than a legend here):
    # "mean" left of its marker, "p95" below its marker — both stay clear of
    # the panel title and of the tokens/calls annotation to the right.
    ax_a.text(means[0] - 7.0, 0, "mean", ha="right", va="center",
              fontsize=6.0, color=BLUE)
    ax_a.text(p95[0], 0.36, "p95", ha="center", va="top",
              fontsize=6.0, color=ROSE)
    ax_a.set_yticks(y, rung_labels)
    ax_a.set_xlabel("seconds / question")
    ax_a.set_xlim(0, max(p95) * 1.45)
    ax_a.set_ylim(-.55, 2.55)
    ax_a.invert_yaxis()
    ax_a.text(.03, .04, "767 shared questions · Align+ engine",
              transform=ax_a.transAxes, fontsize=6.0, color=MUTED)
    style_axis(ax_a, BLUE)

    # (b) Paired document-ingest cost, normalized to Base=100 with raw
    # Base -> Align+ values printed above each bar.
    panel_label(ax_b, "(b)", "Ingest cost: Base → Align+", TEAL)
    datasets = ["MAB", "LongMemEval"]
    call_v1, call_v2 = cost_b["call_v1"], cost_b["call_v2"]
    tok_v1, tok_v2 = cost_b["tok_v1"], cost_b["tok_v2"]
    ratio_calls = [100 * b / a for a, b in zip(call_v1, call_v2)]
    ratio_tokens = [100 * b / a for a, b in zip(tok_v1, tok_v2)]
    x = np.arange(2); w = .29
    ax_b.bar(x - w / 2, ratio_calls, w, color=ORANGE, alpha=.9, label="calls / doc")
    ax_b.bar(x + w / 2, ratio_tokens, w, color=TEAL, alpha=.9, label="tokens / doc")
    for i in range(2):
        ax_b.text(x[i] - w / 2, ratio_calls[i] + 3, f"{call_v1[i]:.0f}→{call_v2[i]:.0f}",
                  ha="center", va="bottom", fontsize=6.0, color=INK)
        ax_b.text(x[i] + w / 2, ratio_tokens[i] + 3,
                  f"{tok_v1[i]:.0f}k→{tok_v2[i]:.0f}k",
                  ha="center", va="bottom", fontsize=6.0, color=INK)
    ax_b.axhline(100, color=MUTED, linestyle=(0, (3, 2)), linewidth=.65)
    ax_b.set_xticks(x, datasets)
    ax_b.set_ylabel("Align+/Base cost (%)")
    ax_b.set_ylim(0, 132)
    # fixed ticks: AutoLocator also emits a 140 tick above the view, and tick
    # labels have clip_on=False, so it would bleed past the canvas edge
    ax_b.set_yticks(range(0, 121, 20))
    ax_b.legend(loc="upper right", bbox_to_anchor=(1.0, 0.87), frameon=False,
                ncol=2, handlelength=1.1, borderpad=.1, columnspacing=.6)
    style_axis(ax_b, TEAL)

    # (c) Retrieval-budget frontier: recall vs. delivered evidence payload
    # (log x).  Each k contributes one x position; both recall series share
    # the y axis, and k is annotated once per position.  This replaces the
    # old twin-axis version (unrelated variables on one pair of axes).
    panel_label(ax_c, "(c)", "Retrieval budget frontier", BLUE)
    depth = ledger["depth_diagnostic"]
    ks = np.array(depth["ks"], dtype=int)
    per_k = depth["per_k"]
    any_recall = np.array([per_k[str(k)]["recall_any_pct"] for k in ks])
    all_recall = np.array([per_k[str(k)]["recall_all_pct"] for k in ks])
    payload = np.array([per_k[str(k)]["mean_evidence_payload_bytes"] / 1000 for k in ks])
    assert np.all(np.diff(payload) > 0), "payload must be monotone in k"
    ax_c.plot(payload, any_recall, color=BLUE, marker="o", markersize=3.4,
              linewidth=1.2, label="Recall-any@k")
    ax_c.plot(payload, all_recall, color=TEAL, marker="s", markersize=3.1,
              linewidth=1.1, linestyle=(0, (4, 2)), label="Recall-all@k")
    # k labels sit in each point's upper-left quadrant: the frontier rises
    # to the right, so that corner is always empty (incl. the k=1 left edge).
    # The rightmost point instead labels between the two curves — the panel
    # title occupies its upper-left quadrant.
    for xx, k, hi, lo in zip(payload, ks, any_recall, all_recall):
        if k == ks[-1]:
            ax_c.text(xx * 0.94, (hi + lo) / 2, f"k={k}", ha="right",
                      va="center", fontsize=6.0, color=MUTED)
        else:
            ax_c.text(xx * 0.94, hi + 4.0, f"k={k}", ha="right", va="bottom",
                      fontsize=6.0, color=MUTED)
    ax_c.set_xscale("log")
    ax_c.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax_c.set_xticks([0.2, 0.5, 1, 2, 4])
    ax_c.set_xticklabels(["0.2", "0.5", "1", "2", "4"])
    ax_c.set_xlabel("evidence payload (kB)")
    ax_c.set_ylabel("recall (%)")
    ax_c.set_xlim(0.15, 5.6)
    ax_c.set_ylim(0, 96)
    ax_c.legend(loc="lower right", bbox_to_anchor=(1.0, 0.16), frameon=False,
                handlelength=1.4, borderpad=.1)
    ax_c.text(.03, .04, "fixed-ranking prefix slice · offline n=210",
              transform=ax_c.transAxes, fontsize=6.0, color=MUTED)
    style_axis(ax_c, BLUE)

    # (d) Source-only channel ablation: recall-any and recall-all at k=10.
    panel_label(ax_d, "(d)", "Evidence-channel ablation", ROSE)
    arms = ["lexical", "+ semantic", "+ neighbor", "+ relation"]
    arm_keys = ["arm1_source_lexical", "arm2_source_semantic",
                "arm3_source_neighbors", "arm4_source_relations"]
    x = np.arange(4)
    ab = ledger["provenance_ablation_x7"]["arms"]
    recall_any = np.array([ab[k]["per_k"]["10"]["recall_any_pct"] for k in arm_keys])
    recall_all = np.array([ab[k]["per_k"]["10"]["recall_all_pct"] for k in arm_keys])
    ax_d.plot(x, recall_any, color=BLUE, marker="o", markersize=3.5,
              linewidth=1.3, label="recall-any@10")
    ax_d.plot(x, recall_all, color=TEAL, marker="s", markersize=3.2,
              linewidth=1.2, linestyle=(0, (4, 2)), label="recall-all@10")
    # per-point value labels dropped: the any-series is flat at ~72.4 and the
    # exact four values are already printed in §4 via the \XSevenMetrics macro
    ax_d.set_xticks(x, arms)
    ax_d.set_ylabel("recall at $k=10$ (%)")
    ax_d.set_ylim(35, 96)
    ax_d.legend(loc="upper right", bbox_to_anchor=(1.0, 0.84), frameon=False,
                ncol=2, handlelength=1.1, borderpad=.1, columnspacing=.5)
    ax_d.text(.03, .04, "only enabled channels change · offline n=210",
              transform=ax_d.transAxes, fontsize=6.0, color=MUTED)
    style_axis(ax_d, ROSE)

    fig.tight_layout(pad=.5, h_pad=1.1, w_pad=.9)
    print(f"[fig3] min font = {audit_min_font(fig):.2f}pt")
    save(fig, "fig3_cost")


if __name__ == "__main__":
    make_figure()
