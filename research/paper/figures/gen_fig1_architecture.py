from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from paper_plot_style import COLORS, save


def box(ax, x, y, w, h, text, fill, edge, fontsize=8, lw=1.2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fill, edgecolor=edge, linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, color="#4B5563", style="-|>", lw=1.1, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle=style, mutation_scale=10, linewidth=lw,
            color=color, linestyle="--" if dashed else "-",
            connectionstyle="arc3,rad=0.0",
        )
    )


fig, ax = plt.subplots(figsize=(7.2, 3.25))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Left conceptual contrast.
box(ax, 0.02, 0.57, 0.16, 0.14, "Compressed fact\n/ graph edge", COLORS["light_orange"], COLORS["orange"])
box(ax, 0.02, 0.27, 0.16, 0.14, "Answer authority", "#FDECEC", "#B94A48")
arrow(ax, (0.10, 0.57), (0.10, 0.41), color="#B94A48")
ax.text(0.10, 0.80, "Conflated memory", ha="center", va="center", fontsize=8, color="#7F1D1D")
ax.text(0.10, 0.18, "Extraction errors can\nbecome pseudo-facts", ha="center", va="center", fontsize=7, color="#7F1D1D")

# Separation rule.
ax.plot([0.215, 0.215], [0.08, 0.92], color="#D1D5DB", lw=1.0)

# Source-of-truth layer.
box(ax, 0.27, 0.08, 0.18, 0.14, "Documents & turns\n(source of truth)", COLORS["light_blue"], COLORS["blue"], 8)
box(ax, 0.49, 0.08, 0.18, 0.14, "Versions & timestamps\n(active / historical)", COLORS["light_blue"], COLORS["blue"], 8)
box(ax, 0.71, 0.08, 0.18, 0.14, "Source spans\n& provenance", COLORS["light_blue"], COLORS["blue"], 8)
ax.text(0.58, 0.025, "Authoritative evidence layer", ha="center", va="center", fontsize=8, color=COLORS["blue"])

# Overlay/navigation layer.
box(ax, 0.31, 0.39, 0.16, 0.14, "Stable concept\nfamilies", COLORS["light_green"], COLORS["green"], 8)
box(ax, 0.51, 0.39, 0.16, 0.14, "Versioned\nobservations\n& assertions", COLORS["light_green"], COLORS["green"], 7.3)
box(ax, 0.71, 0.39, 0.16, 0.14, "Relation graph\n& neighborhoods", COLORS["light_green"], COLORS["green"], 8)
ax.text(0.59, 0.34, "Rebuildable navigation overlay", ha="center", va="center", fontsize=8, color="#25613F")

# Query layer.
box(ax, 0.26, 0.72, 0.13, 0.12, "Lexical", COLORS["light_gray"], COLORS["gray"])
box(ax, 0.42, 0.72, 0.13, 0.12, "Semantic", COLORS["light_gray"], COLORS["gray"])
box(ax, 0.58, 0.72, 0.13, 0.12, "Graph /\nprovenance", COLORS["light_gray"], COLORS["gray"])
box(ax, 0.75, 0.72, 0.10, 0.12, "RRF", "#F3ECFA", COLORS["purple"])
box(ax, 0.88, 0.68, 0.105, 0.20, "Scope gate\n→ source read\n→ evidence submit", "#F3ECFA", COLORS["purple"], 7.2)
ax.text(0.57, 0.93, "Source-bounded evidence path", ha="center", va="center", fontsize=8, color="#4C3575")

# Provenance arrows up; source read down.
for x0, x1 in [(0.36, 0.39), (0.58, 0.59), (0.80, 0.79)]:
    arrow(ax, (x0, 0.22), (x1, 0.39), color=COLORS["green"], dashed=True)
for x0, x1 in [(0.39, 0.325), (0.59, 0.485), (0.79, 0.645)]:
    arrow(ax, (x0, 0.53), (x1, 0.72), color=COLORS["gray"])
for x0 in [0.325, 0.485, 0.645]:
    ax.plot([x0, x0], [0.72, 0.665], color=COLORS["purple"], lw=1.0)
ax.plot([0.325, 0.80], [0.665, 0.665], color=COLORS["purple"], lw=1.0)
arrow(ax, (0.80, 0.665), (0.80, 0.72), color=COLORS["purple"])
arrow(ax, (0.85, 0.78), (0.88, 0.78), color=COLORS["purple"])
arrow(ax, (0.93, 0.68), (0.80, 0.22), color=COLORS["blue"], dashed=True)

save(fig, "fig1_architecture")
