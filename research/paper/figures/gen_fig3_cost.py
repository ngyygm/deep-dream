"""Figure 3: per-question compute by rung (tokens and LLM calls).

Values are read from paper/results/recomputed_values.json (written by
paper/results/recompute_from_logs.py from the frozen MAB v2 run logs,
including rung 1 from results.baseline.jsonl). A rung whose source log is
missing renders as an explicit "pending" marker, never an estimate.
Regenerate: python gen_fig3_cost.py
"""

import json
import os

from matplotlib import pyplot as plt

from paper_plot_style import COLORS, save

_VAL = os.path.join(os.path.dirname(__file__), "..", "results", "recomputed_values.json")
RUNG_COST = None
try:
    with open(_VAL) as fh:
        RUNG_COST = json.load(fh)["rung_cost_v2"]
except (OSError, KeyError):
    pass

fig, ax = plt.subplots(figsize=(4.4, 2.8))

if RUNG_COST is None:
    ax.axis("off")
    ax.text(0.5, 0.55, "Per-question compute by rung", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#1F2937")
    ax.text(0.5, 0.32, "Recomputed from frozen run logs by\n"
            "paper/results/recompute_from_logs.py;\n"
            "recomputed_values.json not found.",
            ha="center", va="center", fontsize=8, color="#6B7280")
else:
    labels, tokens, calls = [], [], []
    pending = False
    for i in (1, 2, 3):
        c = RUNG_COST.get(f"rung{i}")
        if c is None:
            labels.append(f"Rung {i}\n(pending)")
            tokens.append(0.0)
            calls.append(0.0)
            pending = True
        else:
            labels.append(f"Rung {i}")
            tokens.append(c["tokens_per_question"] / 1000.0)
            calls.append(c["llm_calls_per_question"])

    xs = range(len(labels))
    bars = ax.bar(xs, tokens, width=0.55, color=COLORS["blue"],
                  edgecolor=COLORS["blue"])
    for xi, t, c in zip(xs, tokens, calls):
        if t > 0:
            ax.text(xi, t + 2.0, f"{t:.1f}k tok\n{c:.1f} calls", ha="center",
                    va="bottom", fontsize=7.2)
    if pending:
        ax.text(0, 3.0, "per-question file\nnot yet synced", ha="center",
                va="bottom", fontsize=6.8, color="#6B7280", rotation=90)

    ax.set_xticks(list(xs), labels)
    ax.set_ylabel("tokens / question ($\\times$1000)")
    ax.set_ylim(0, max(tokens) * 1.32)

save(fig, "fig3_cost")
