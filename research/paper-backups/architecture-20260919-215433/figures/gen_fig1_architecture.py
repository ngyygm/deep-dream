"""Academic architecture figure for Deep-Dream.

The layout separates the agent's progressive read loop from the memory's
three linked layers: concept addresses, temporal versions, and source
evidence.  It is intentionally rendered as a vector PDF for the paper.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle


OUT = Path(__file__).resolve().parent

BLUE = "#1769aa"
BLUE_DARK = "#0d477a"
TEAL = "#188c91"
TEAL_DARK = "#08666b"
PINK = "#d14b72"
PINK_LIGHT = "#fff0f4"
BLUE_LIGHT = "#edf6fc"
TEAL_LIGHT = "#eefafa"
GOLD = "#d98b32"
GOLD_LIGHT = "#fff6e8"
INK = "#1f2933"
MUTED = "#64748b"
GRID = "#cbd5df"


def rounded(ax, xy, w, h, text="", *, fc="white", ec=GRID, lw=1.0,
            fs=8.0, color=INK, weight="normal", radius=0.03, ha="left",
            va="center", z=3):
    patch = FancyBboxPatch(
        xy, w, h, boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(patch)
    if text:
        ax.text(xy[0] + (0.018 if ha == "left" else w / 2), xy[1] + h / 2,
                text, ha=ha, va=va, fontsize=fs, color=color,
                fontweight=weight, zorder=z + 1)
    return patch


def arrow(ax, start, end, *, color=BLUE, lw=1.2, dashed=False,
          head=7, rad=0.0, z=2):
    style = "-|>"
    ls = (0, (3, 2)) if dashed else "-"
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=head,
        linewidth=lw, linestyle=ls, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=z,
    ))


def layer(ax, x, y, w, h, title, subtitle, color, fill, nodes, edges,
          *, index_label):
    # A slightly slanted panel gives the layer-stack visual without using a
    # decorative 3-D effect.
    slant = 0.12
    poly = Polygon([(x, y), (x + w, y), (x + w + slant, y + h),
                    (x + slant, y + h)], closed=True,
                   facecolor=fill, edgecolor=color, linewidth=1.0, zorder=1)
    ax.add_patch(poly)
    ax.text(x + 0.20, y + h - 0.23, index_label, color=color,
            fontsize=10, fontweight="bold", va="top", zorder=3)
    ax.text(x + 0.62, y + h - 0.20, title, color=color,
            fontsize=9.0, fontweight="bold", va="top", zorder=3)
    # Keep the panel title unobstructed.  The layer purpose is explained by
    # the bottom legend and the labels inside each panel; a subtitle here
    # would compete with the graph nodes at paper scale.
    for (nx, ny, label, c) in nodes:
        circ = Circle((nx, ny), 0.075, facecolor=c, edgecolor=color,
                      linewidth=0.8, zorder=3)
        ax.add_patch(circ)
        ax.text(nx, ny - 0.14, label, ha="center", va="top",
                fontsize=6.0, color=INK, zorder=4)
    for (a, b) in edges:
        arrow(ax, a, b, color=color, lw=0.8, head=5, z=2)


def main():
    fig, ax = plt.subplots(figsize=(13.2, 6.35), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 6.35)
    ax.axis("off")

    # Outer frame and title.
    rounded(ax, (0.18, 0.20), 12.82, 5.92, fc="white", ec="#9aa7b5",
            lw=0.9, radius=0.08, z=0)
    rounded(ax, (0.46, 5.64), 5.25, 0.32,
            "Question: What changed in Alex's travel plan?",
            fc="#f8fafc", ec=INK, fs=8.6, weight="bold", radius=0.04)
    ax.text(6.10, 5.79, "Deep-Dream: concept addresses, version links, source evidence",
            fontsize=10.5, color=BLUE_DARK, fontweight="bold", ha="left")

    # Left: the agent loop, using compact, explicit operations.
    ax.text(0.52, 5.30, "AGENT-CONTROLLED READ", fontsize=8.2,
            color=BLUE_DARK, fontweight="bold")
    rounded(ax, (0.48, 4.70), 4.92, 0.43, "1  Context tracker   previous episode · current uncertainty",
            fc=BLUE_LIGHT, ec=BLUE, fs=7.2, weight="bold")
    rounded(ax, (0.48, 4.06), 4.92, 0.43, "2  Concept search   Alex · travel · spouse  →  family F₇",
            fc=BLUE_LIGHT, ec=BLUE, fs=7.2, weight="bold")
    rounded(ax, (0.48, 3.42), 4.92, 0.43, "3  Version check   latest compatible v₃  ↔  prior v₂",
            fc=PINK_LIGHT, ec=PINK, fs=7.2, weight="bold")
    rounded(ax, (0.48, 2.78), 4.92, 0.43, "4  Relation walk   spouse(Alex)  →  trip  →  destination",
            fc=TEAL_LIGHT, ec=TEAL, fs=7.2, weight="bold")
    rounded(ax, (0.48, 2.14), 4.92, 0.43, "5  Source read   notes.md [12:28]  +  evidence gate",
            fc=GOLD_LIGHT, ec=GOLD, fs=7.2, weight="bold")
    for y1, y2, col in [(4.70, 4.49, BLUE), (4.06, 3.85, BLUE),
                        (3.42, 3.21, PINK), (2.78, 2.57, TEAL),
                        (2.14, 1.93, GOLD)]:
        arrow(ax, (2.94, y1), (2.94, y2), color=col, lw=1.0, head=6)
    rounded(ax, (0.48, 1.05), 4.92, 0.58,
            "Evidence sufficient?  →  answer\notherwise: expand another layer",
            fc="#f8fafc", ec=BLUE_DARK, fs=7.2, weight="bold")
    arrow(ax, (2.94, 2.14), (2.94, 1.63), color=GOLD, lw=1.0, head=6)
    arrow(ax, (2.94, 1.05), (2.94, 4.70), color=BLUE_DARK,
          lw=0.9, dashed=True, head=6, rad=0.22)

    # Right: document input and linked memory layers.
    rounded(ax, (8.86, 4.74), 3.42, 0.62, fc="#fffaf0", ec=GOLD,
            lw=1.0, radius=0.04)
    ax.text(9.05, 5.17, "SOURCE CORPUS", fontsize=8.0, color=GOLD,
            fontweight="bold")
    for x, lab in [(9.22, "notes.md"), (10.12, "trip.pdf"), (11.02, "chat.log")]:
        rounded(ax, (x, 4.83), 0.68, 0.24, lab, fc="white", ec=GOLD,
                fs=5.8, color=INK, ha="center", radius=0.02)
    arrow(ax, (10.58, 4.74), (10.58, 4.40), color=GOLD, lw=1.0, head=7)
    ax.text(10.66, 4.56, "write / observe", fontsize=6.1, color=GOLD,
            va="center")

    # Concept layer.
    concept_nodes = [
        (9.50, 3.82, "Alex", "#83c5f4"),
        (10.42, 3.65, "spouse", "#83c5f4"),
        (11.36, 3.87, "trip", "#83c5f4"),
        (11.00, 3.40, "travel", "#83c5f4"),
        (9.72, 3.37, "family F₇", "#83c5f4"),
    ]
    layer(ax, 8.42, 3.30, 3.70, 1.00, "Concept layer", "stable handles + relation neighborhoods", BLUE,
          BLUE_LIGHT, concept_nodes,
          [((9.50,3.82),(10.42,3.65)),((10.42,3.65),(11.36,3.87)),
           ((11.36,3.87),(11.00,3.40)),((9.50,3.82),(9.72,3.37))],
          index_label="A")

    # Version layer.
    version_nodes = [
        (9.42, 2.48, "v₁", "#f4a3b9"),
        (10.25, 2.33, "v₂", "#ea7f9f"),
        (11.10, 2.52, "v₃", "#d14b72"),
        (11.64, 2.14, "conflict", "#f4a3b9"),
    ]
    layer(ax, 8.42, 2.02, 3.70, 1.00, "Version layer", "observation time · revisions · conflicts", PINK,
          PINK_LIGHT, version_nodes,
          [((9.42,2.48),(10.25,2.33)),((10.25,2.33),(11.10,2.52)),
           ((11.10,2.52),(11.64,2.14))], index_label="V")

    # Evidence layer.
    evidence_nodes = [
        (9.38, 1.12, "E₁", "#8fd3c7"),
        (10.22, 1.27, "E₂", "#5cb9ad"),
        (11.06, 1.10, "E₃", "#2c9b93"),
        (11.72, 1.34, "span", "#8fd3c7"),
    ]
    layer(ax, 8.42, 0.72, 3.70, 1.00, "Evidence layer", "episode · chunk · exact source span", TEAL,
          TEAL_LIGHT, evidence_nodes,
          [((9.38,1.12),(10.22,1.27)),((10.22,1.27),(11.06,1.10)),
           ((11.06,1.10),(11.72,1.34))], index_label="E")

    # Cross-layer pointers are the key difference from a flat hierarchy.
    arrow(ax, (9.72, 3.34), (9.42, 2.58), color=PINK, lw=1.0, dashed=True, head=6)
    arrow(ax, (10.98, 3.34), (11.10, 2.62), color=PINK, lw=1.0, dashed=True, head=6)
    arrow(ax, (10.25, 2.23), (10.22, 1.37), color=TEAL, lw=1.0, dashed=True, head=6)
    arrow(ax, (11.64, 2.02), (11.72, 1.44), color=TEAL, lw=1.0, dashed=True, head=6)
    ax.text(12.20, 2.74, "concept → version", fontsize=6.1, color=PINK,
            rotation=74, ha="center", va="center")
    ax.text(12.15, 1.63, "version → evidence", fontsize=6.1, color=TEAL,
            rotation=74, ha="center", va="center")

    # Agent connects to all three layers and can stop after any sufficient read.
    arrow(ax, (5.40, 4.27), (8.45, 3.80), color=BLUE, lw=1.0, dashed=True, head=7)
    arrow(ax, (5.40, 3.64), (8.45, 2.47), color=PINK, lw=1.0, dashed=True, head=7)
    arrow(ax, (5.40, 2.35), (8.45, 1.18), color=TEAL, lw=1.0, dashed=True, head=7)
    ax.text(6.35, 4.10, "navigate", fontsize=6.3, color=BLUE, rotation=-9)
    ax.text(6.30, 3.10, "check revision", fontsize=6.3, color=PINK, rotation=-17)
    ax.text(6.25, 1.70, "read evidence", fontsize=6.3, color=TEAL, rotation=-24)

    # Bottom design summary.
    ax.text(6.52, 0.40, "Navigate broadly with concepts and relations  ·  revise through versions  ·  answer from source spans",
            fontsize=7.5, color=BLUE_DARK, ha="center", va="bottom", fontweight="bold")

    plt.savefig(OUT / "fig1_architecture.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.savefig(OUT / "fig1_architecture.png", dpi=260, bbox_inches="tight", pad_inches=0.05)
    print(OUT / "fig1_architecture.pdf")


if __name__ == "__main__":
    main()
