"""
gen_fig_main_results_comparison.py — Grouped bar chart comparing MRR, Hits@1, Hits@10
across all methods on ICEWS14, ICEWS18, and GDELT.
Data source: experimental_log.md (extracted to metrics.json)
"""
import sys
sys.path.insert(0, '.')
from paper_plot_style import *
import json
import numpy as np

# Load extracted metrics
with open('../workspace/metrics.json') as f:
    metrics = json.load(f)

table = metrics['tables'][0]
headers = table['headers']
rows = table['rows']

# Parse data (strip markdown bold markers)
def strip_bold(s):
    return s.replace('**', '')

methods = [strip_bold(r[0]) for r in rows]
icews14_mrr = [float(strip_bold(r[1])) for r in rows]
icews14_h1 = [float(strip_bold(r[2])) for r in rows]
icews14_h10 = [float(strip_bold(r[3])) for r in rows]
icews18_mrr = [float(strip_bold(r[4])) for r in rows]
icews18_h1 = [float(strip_bold(r[5])) for r in rows]
icews18_h10 = [float(strip_bold(r[6])) for r in rows]
gdelt_mrr = [float(strip_bold(r[7])) for r in rows]
gdelt_h1 = [float(strip_bold(r[8])) for r in rows]
gdelt_h10 = [float(strip_bold(r[9])) for r in rows]

# Plot: 3 subplots (one per dataset), grouped bars for MRR, H@1, H@10
fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=False)

datasets = [
    ('ICEWS14', icews14_mrr, icews14_h1, icews14_h10),
    ('ICEWS18', icews18_mrr, icews18_h1, icews18_h10),
    ('GDELT', gdelt_mrr, gdelt_h1, gdelt_h10),
]

metric_labels = ['MRR', 'Hits@1', 'Hits@10']
x = np.arange(len(methods))
width = 0.25

for ax_idx, (dataset_name, mrr, h1, h10) in enumerate(datasets):
    ax = axes[ax_idx]
    colors_list = [METHOD_COLORS.get(m, '#999999') for m in methods]

    bars1 = ax.bar(x - width, mrr, width, label='MRR' if ax_idx == 0 else None,
                   color=[METHOD_COLORS.get(m, '#999999') for m in methods], alpha=0.9, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x, h1, width, label='Hits@1' if ax_idx == 0 else None,
                   color=[METHOD_COLORS.get(m, '#999999') for m in methods], alpha=0.6, edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, h10, width, label='Hits@10' if ax_idx == 0 else None,
                   color=[METHOD_COLORS.get(m, '#999999') for m in methods], alpha=0.4, edgecolor='white', linewidth=0.5)

    ax.set_xlabel(dataset_name, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Score' if ax_idx == 0 else '')
    if ax_idx == 0:
        ax.legend(frameon=False, fontsize=8, loc='upper left')

    # Highlight EMN bar
    ax.axvspan(len(methods) - 1.5, len(methods) - 0.5, alpha=0.08, color='red')

fig.tight_layout()
save_fig(fig, 'fig_main_results_comparison')
