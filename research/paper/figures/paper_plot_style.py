from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


FIG_DIR = Path(__file__).resolve().parent
DATA_PATH = FIG_DIR.parent / "results" / "benchmark_summary.json"
COLORS = {
    "blue": "#3366A3",
    "orange": "#D97A2B",
    "green": "#4C956C",
    "purple": "#7B61A8",
    "gray": "#6B7280",
    "light_blue": "#EAF2F8",
    "light_green": "#EAF5EE",
    "light_orange": "#FFF2E6",
    "light_gray": "#F3F4F6",
}

mpl.rcParams.update(
    {
        "font.size": 9,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=180)
    plt.close(fig)
