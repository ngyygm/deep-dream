import json

import numpy as np
from matplotlib import pyplot as plt

from paper_plot_style import COLORS, DATA_PATH, save


data = json.loads(DATA_PATH.read_text())
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55))

# Panel a: channel decomposition.
labels = ["Lexical", "Semantic", "Fused"]
keys = ["lexical", "semantic", "fused"]
any_vals = [data["channel_diagnostic_pct"][k]["recall_any"] for k in keys]
all_vals = [data["channel_diagnostic_pct"][k]["recall_all"] for k in keys]
x = np.arange(len(labels))
w = 0.34
axes[0].bar(x - w / 2, any_vals, w, label="Recall-any", color=COLORS["blue"])
axes[0].bar(x + w / 2, all_vals, w, label="Recall-all", color=COLORS["orange"])
axes[0].set_xticks(x, labels)
axes[0].set_ylabel("Evidence recall (%)")
axes[0].set_ylim(60, 100)
axes[0].text(0.02, 0.96, "(a) n=210 development questions", transform=axes[0].transAxes, va="top", fontsize=8)

# Panel b: selected difficult subset.
labels = ["Span", "±1", "±2", "Context-3"]
keys = ["lexical_semantic_span", "neighbors_1", "neighbors_2", "legacy_context_3"]
any_vals = [data["source_expansion_diagnostic"][k]["recall_any_pct"] for k in keys]
all_vals = [data["source_expansion_diagnostic"][k]["recall_all_pct"] for k in keys]
x = np.arange(len(labels))
axes[1].bar(x - w / 2, any_vals, w, label="Recall-any", color=COLORS["blue"])
axes[1].bar(x + w / 2, all_vals, w, label="Recall-all", color=COLORS["orange"])
axes[1].set_xticks(x, labels)
axes[1].set_ylim(40, 100)
axes[1].text(0.02, 0.96, "(b) n=105 prior-error cases", transform=axes[1].transAxes, va="top", fontsize=8)

for ax in axes:
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    ax.set_axisbelow(True)

handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
fig.subplots_adjust(top=0.83, bottom=0.18, left=0.09, right=0.99, wspace=0.20)

save(fig, "fig2_retrieval_ablation")
