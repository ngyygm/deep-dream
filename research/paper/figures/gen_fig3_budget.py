import json

from matplotlib import pyplot as plt

from paper_plot_style import COLORS, DATA_PATH, save


data = json.loads(DATA_PATH.read_text())
fig, ax = plt.subplots(figsize=(3.65, 2.65))

# Difficult-subset source expansion.
source_keys = ["lexical_span", "neighbors_1", "neighbors_2", "legacy_context_3"]
source_labels = ["Exact", "±1", "±2", "Context-3"]
xs = [data["source_expansion_diagnostic"][k]["mean_response_bytes"] for k in source_keys]
ys = [data["source_expansion_diagnostic"][k]["recall_all_pct"] for k in source_keys]
ax.plot(xs, ys, "o-", color=COLORS["green"], label="Source expansion (n=105)")
for x, y, label in zip(xs, ys, source_labels):
    ax.annotate(label, (x, y), xytext=(3, 4), textcoords="offset points", fontsize=7)

# Corrected budget frontier: prefix-monotonic depth sweep (k=1,3,5,10,20).
depth = data["depth_diagnostic"]
sweep_ks = [str(k) for k in depth["ks"]]
xd = [depth["per_k"][k]["mean_evidence_payload_bytes"] for k in sweep_ks]
yd = [depth["per_k"][k]["recall_all_pct"] for k in sweep_ks]
ax.plot(xd, yd, "s--", color=COLORS["purple"], label="Budget frontier (n=210)")
for x, y, label in zip(xd, yd, sweep_ks):
    ax.annotate(label, (x, y), xytext=(3, -10), textcoords="offset points", fontsize=7)

ax.set_xlabel("Evidence payload (bytes)")
ax.set_ylabel("Evidence recall-all (%)")
ax.set_xlim(0, 4200)
ax.set_ylim(0, 90)
ax.grid(color="#E5E7EB", linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")

save(fig, "fig3_budget_frontier")
