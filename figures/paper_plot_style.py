"""
paper_plot_style.py — Shared plotting style for EMN paper figures.
Publication-quality defaults for ICLR submission.
"""
import matplotlib.pyplot as plt
import matplotlib

FONT_SIZE = 10
DPI = 300
FORMAT = 'pdf'
FIG_DIR = 'figures'
COLOR_PALETTE = 'tab10'

matplotlib.rcParams.update({
    'font.size': FONT_SIZE,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 1,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    'legend.fontsize': FONT_SIZE - 1,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})

# Color palette
COLORS = plt.cm.tab10.colors

# Method colors (consistent across all figures)
METHOD_COLORS = {
    'TransE': '#7f7f7f',
    'ConvE': '#bcbd22',
    'RotatE': '#17becf',
    'RE-NET': '#1f77b4',
    'CyGNet': '#ff7f0e',
    'RE-GCN': '#2ca02c',
    'HiSMatch': '#d62728',
    'TANGO': '#9467bd',
    'EMN (Ours)': '#e74c3c',
}

# Ablation colors
ABLATION_COLORS = {
    'Full EMN': '#2ca02c',
    'w/o Temporal\nPosition Encoder': '#e74c3c',
    'w/o Episodic\nGrouping': '#f39c12',
    'w/o Gated\nAttention': '#3498db',
}


def save_fig(fig, name, fmt=FORMAT):
    """Save figure to FIG_DIR with consistent naming."""
    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    path = f'{FIG_DIR}/{name}.{fmt}'
    fig.savefig(path)
    print(f'Saved: {path}')


def get_fig_size(aspect_ratio, base_width=5.0):
    """Convert aspect ratio string to (width, height) tuple."""
    w, h = map(int, aspect_ratio.split(':'))
    return (base_width, base_width * h / w)
