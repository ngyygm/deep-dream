"""Figure 1: architecture hero — authority split + three answer paths (rungs).

Static, hand-positioned diagram; the only numbers are the rung Overall
scores (v2 engine, MAB subset, official scorer) kept in sync with
Table tab:track-ladder. Regenerate: python gen_fig1_architecture.py
"""

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


def line(ax, start, end, color, lw=1.1):
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=lw,
            solid_capstyle="round")


fig, ax = plt.subplots(figsize=(7.2, 3.25))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ---- Left: authoritative source layer (blue) -------------------------------
box(ax, 0.02, 0.62, 0.20, 0.15, "Documents & turns\n(source of truth)", COLORS["light_blue"], COLORS["blue"], 8)
box(ax, 0.02, 0.40, 0.20, 0.15, "Versions & timestamps\n(active / historical)", COLORS["light_blue"], COLORS["blue"], 7.5)
box(ax, 0.02, 0.18, 0.20, 0.15, "Source spans\n& provenance", COLORS["light_blue"], COLORS["blue"], 8)
ax.text(0.12, 0.075, "Authoritative sources", ha="center", va="center",
        fontsize=8.5, color=COLORS["blue"], fontweight="bold")

# ---- Middle: non-authoritative navigation overlay (green) ------------------
box(ax, 0.26, 0.62, 0.17, 0.15, "Stable concept\nfamilies", COLORS["light_green"], COLORS["green"], 7.5)
box(ax, 0.26, 0.40, 0.17, 0.15, "Versioned observations\n& assertions", COLORS["light_green"], COLORS["green"], 7)
box(ax, 0.26, 0.18, 0.17, 0.15, "Relation graph\n& neighborhoods", COLORS["light_green"], COLORS["green"], 7.5)
ax.text(0.345, 0.075, "Navigation overlay (derived, rebuildable)", ha="center", va="center",
        fontsize=7.5, color="#25613F")
# Derivation: sources -> overlay (dashed, one-way, in the gutter).
arrow(ax, (0.225, 0.695), (0.255, 0.695), color=COLORS["green"], dashed=True)

# ---- Right: three answer paths (rungs) --------------------------------------
rung_x, rung_w = 0.51, 0.425
box(ax, rung_x, 0.70, rung_w, 0.17,
    "Rung 1  single-shot\nquery $\\rightarrow$ fused retrieval payload $\\rightarrow$ answer",
    "#F3F4F6", COLORS["gray"], 7.6)
ax.text(0.998, 0.785, "0.32", ha="right", va="center", fontsize=7.5, color=COLORS["gray"])
box(ax, rung_x, 0.44, rung_w, 0.17,
    "Rung 2  tool agent\nbounded loop over read-only memory tools",
    "#F3ECFA", COLORS["purple"], 7.6)
ax.text(0.998, 0.525, "0.41", ha="right", va="center", fontsize=7.5, color=COLORS["purple"])
box(ax, rung_x, 0.16, rung_w, 0.20,
    "Rung 3  sandbox\nscope materialization $\\rightarrow$ verify loop\n"
    "with shell tools $\\rightarrow$ evidence gate",
    "#EDE4F7", COLORS["purple"], 7.6, lw=1.6)
ax.text(0.998, 0.26, "0.65", ha="right", va="center", fontsize=7.5,
        color=COLORS["purple"], fontweight="bold")
ax.text(0.73, 0.075, "answer paths (Overall, $v2$)", ha="center", va="center",
        fontsize=8, color="#4C3575")

# Access depth: arrows from each rung toward overlay / sources.
arrow(ax, (0.493, 0.785), (0.435, 0.70), color=COLORS["green"], lw=1.0)    # r1: retrieval payload
arrow(ax, (0.493, 0.525), (0.435, 0.475), color=COLORS["green"], lw=1.2)   # r2: memory tools
# r3: elbow below the overlay column into the source layer (blue).
line(ax, (0.493, 0.20), (0.44, 0.135), color=COLORS["blue"], lw=2.0)
line(ax, (0.44, 0.135), (0.12, 0.135), color=COLORS["blue"], lw=2.0)
arrow(ax, (0.12, 0.135), (0.12, 0.175), color=COLORS["blue"], lw=2.0)      # into source spans

# Authority rule, top.
ax.text(0.5, 0.965, "An overlay error can cause a missed scope, never a fabricated fact",
        ha="center", va="center", fontsize=8, color="#7F1D1D")

save(fig, "fig1_architecture")
