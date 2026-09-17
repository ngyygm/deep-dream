"""Figure 2: the answer-path ladder on MemoryAgentBench.

Static numbers, kept in sync with Table tab:track-ladder and the vetted
horizontal report (research/reports/memory_systems_horizontal_2026-08-31.md):
Overall x100, official scorer @455306d, n=767 sampled subset, kimi-k3 answerer.
  rung1 v1 30.3  v2 32.4 | rung2 v1 37.6  v2 40.7 | rung3 v1 66.9  v2 65.1
BM25 published row 41.5 (official paper, full set, GPT-4o-mini backbone).

WHISKERS: paired-bootstrap 95% CIs of the macro Overall over the 767 shared
questions, read from paper/results/recomputed_values.json (written by
paper/results/recompute_from_logs.py, seed 0, 2000 resamples); if that file
is absent the bars are drawn without whiskers.
Regenerate: python gen_fig2_ladder.py
"""

import json
import os

from matplotlib import pyplot as plt
import numpy as np

from paper_plot_style import COLORS, save

RUNGS = ["Rung 1\nsingle-shot", "Rung 2\ntool agent", "Rung 3\nsandbox"]
V1 = [30.3, 37.6, 66.9]
V2 = [32.4, 40.7, 65.1]
BM25_PUBLISHED = 41.5

_VAL = os.path.join(os.path.dirname(__file__), "..", "results", "recomputed_values.json")
try:
    with open(_VAL) as _fh:
        _CI = json.load(_fh)["ladder"]["overall_ci95_v2_x100"]
    WHISKERS_V2 = [tuple(_CI[t]) for t in ("baseline", "skill-agent", "pi")]
except (OSError, KeyError):
    WHISKERS_V2 = None

x = np.arange(len(RUNGS))
w = 0.34

fig, ax = plt.subplots(figsize=(4.4, 3.0))
b1 = ax.bar(x - w / 2, V1, w, facecolor="white", edgecolor=COLORS["gray"],
            hatch="////", linewidth=1.1, label="engine $v$1")
b2 = ax.bar(x + w / 2, V2, w, facecolor=COLORS["blue"], edgecolor=COLORS["blue"],
            linewidth=1.1, label="engine $v$2")
if WHISKERS_V2 is not None:
    ys = np.array(V2)
    los = np.array([lo for lo, _ in WHISKERS_V2])
    his = np.array([hi for _, hi in WHISKERS_V2])
    ax.errorbar(x + w / 2, ys, yerr=[ys - los, his - ys], fmt="none",
                ecolor="#1F2937", capsize=3, lw=1.1)
    ax.text(0.99, 0.02, "whiskers: macro Overall, paired bootstrap 95% CI ($n{=}767$)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#374151")

ax.axhline(BM25_PUBLISHED, color=COLORS["orange"], linestyle="--", lw=1.1)
ax.text(2.42, BM25_PUBLISHED + 1.2,
        "BM25 41.5 (published row,\nfull set, GPT-4o-mini)",
        ha="right", va="bottom", fontsize=6.8, color=COLORS["orange"])

for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.0,
                f"{r.get_height():.1f}", ha="center", va="bottom", fontsize=7.2,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))

ax.set_xticks(x, RUNGS)
ax.set_ylabel("Overall ($\\times$100)")
ax.set_ylim(0, 84)
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()

save(fig, "fig2_ladder")
