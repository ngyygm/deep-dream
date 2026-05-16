"""
gen_fig_ablation_study.py — Horizontal bar chart showing ICEWS14 MRR and Hits@10
for the full EMN model and each ablation variant.
Data source: experimental_log.md ablation table
"""
import sys
sys.path.insert(0, '.')
from paper_plot_style import *
import numpy as np

# Ablation data from experimental_log.md
variants = [
    'Full EMN',
    'w/o Temporal\nPosition Encoder',
    'w/o Episodic\nGrouping',
    'w/o Gated\nAttention',
]
mrr_values = [0.440, 0.408, 0.395, 0.412]
h10_values = [0.620, 0.592, 0.580, 0.598]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)

# MRR subplot
ax = axes[0]
colors = [ABLATION_COLORS[v] for v in variants]
y_pos = np.arange(len(variants))
bars = ax.barh(y_pos, mrr_values, color=colors, edgecolor='white', linewidth=0.5, height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(variants, fontsize=9)
ax.set_xlabel('MRR')
ax.set_xlim(0.35, 0.47)
ax.invert_yaxis()

# Add value labels
for bar, val in zip(bars, mrr_values):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8)

# Highlight the drop
for i in range(1, len(mrr_values)):
    drop = mrr_values[0] - mrr_values[i]
    ax.text(0.355, i + 0.15, f'{-drop:.3f}', color='red', fontsize=7, fontstyle='italic')

# Hits@10 subplot
ax = axes[1]
bars = ax.barh(y_pos, h10_values, color=colors, edgecolor='white', linewidth=0.5, height=0.6)
ax.set_xlabel('Hits@10')
ax.set_xlim(0.54, 0.66)
ax.invert_yaxis()
ax.set_yticks([])

for bar, val in zip(bars, h10_values):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8)

for i in range(1, len(h10_values)):
    drop = h10_values[0] - h10_values[i]
    ax.text(0.548, i + 0.15, f'{-drop:.3f}', color='red', fontsize=7, fontstyle='italic')

fig.tight_layout()
save_fig(fig, 'fig_ablation_study')
