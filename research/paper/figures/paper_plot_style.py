from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


FIG_DIR = Path(__file__).resolve().parent
DATA_PATH = FIG_DIR.parent / "results" / "benchmark_summary.json"
COLORS = {
    # Deep-Dream paper palette: blue/teal carry the address and evidence
    # paths; coral/orange mark warnings and cost.  All colours are distinct
    # in grayscale when paired with the marker/hatch encodings used below.
    "blue": "#2E6FAD",
    "navy": "#1E5A92",
    "teal": "#2A9D8F",
    "coral": "#D85A71",
    "orange": "#E39B4A",
    "green": "#2A8C68",
    "purple": "#7763A8",
    "gray": "#6B7280",
    "ink": "#1F2937",
    "muted": "#5B6773",
    "grid": "#D9E3EC",
    "panel": "#F7FAFC",
    "light_blue": "#EEF5FB",
    "light_green": "#EEF9F7",
    "light_orange": "#FFF6EA",
    "light_coral": "#FFF1F4",
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
        "savefig.bbox": None,
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    """Export all publication and QA formats at the declared final size."""
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches=None)
    fig.savefig(FIG_DIR / f"{name}.svg", format="svg", bbox_inches=None)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches=None)
    # Grayscale preview is intentionally part of the artifact: it catches
    # colour-only distinctions before the figure reaches the manuscript.
    try:
        from PIL import Image, ImageOps
        ImageOps.grayscale(Image.open(FIG_DIR / f"{name}.png")).save(
            FIG_DIR / f"{name}_gray.png")
    except Exception:
        pass
    plt.close(fig)
