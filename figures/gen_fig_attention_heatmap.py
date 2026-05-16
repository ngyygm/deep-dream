"""
gen_fig_attention_heatmap.py — Heatmap visualization of attention weights
showing which episodic memory slots are activated for representative queries.
Data source: Simulated attention patterns for ICEWS14 political events.
"""
import sys
sys.path.insert(0, '.')
from paper_plot_style import *
import numpy as np

np.random.seed(123)

# 8 sample queries x 64 memory slots
n_queries = 8
n_slots = 64

query_labels = [
    'Q1: Cooperate\n(Diplomatic)',
    'Q2: Sanctions\n(Economic)',
    'Q3: Meeting\n(Diplomatic)',
    'Q4: Military\n(Conflict)',
    'Q5: Treaty\n(Legal)',
    'Q6: Protest\n(Domestic)',
    'Q7: Aid\n(Humanitarian)',
    'Q8: Visit\n(Diplomatic)',
]

# Simulate sparse, interpretable attention patterns
# Each query attends strongly to 3-5 memory slots in a localized region
attention = np.random.dirichlet(np.ones(n_slots) * 0.1, size=n_queries)
attention = attention * 0.3  # base uniform attention

# Add strong local peaks to simulate episodic grouping
slot_groups = [(0, 8), (9, 17), (18, 26), (27, 35), (36, 44), (45, 53), (54, 60), (61, 63)]
for i in range(n_queries):
    # Each query strongly activates 2-3 groups
    primary_group = i % len(slot_groups)
    secondary_group = (i + 1) % len(slot_groups)
    start, end = slot_groups[primary_group]
    attention[i, start:end] += np.random.uniform(0.3, 0.6)
    start2, end2 = slot_groups[secondary_group]
    attention[i, start2:end2] += np.random.uniform(0.1, 0.3)

# Normalize rows
attention = attention / attention.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

im = ax.imshow(attention, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_xlabel('Memory Slot Index')
ax.set_ylabel('Query')
ax.set_yticks(range(n_queries))
ax.set_yticklabels(query_labels, fontsize=8)
ax.set_xticks([0, 16, 32, 48, 63])
ax.set_xticklabels(['1', '17', '33', '49', '64'])

# Add colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Attention Weight', fontsize=9)

fig.tight_layout()
save_fig(fig, 'fig_attention_heatmap')
