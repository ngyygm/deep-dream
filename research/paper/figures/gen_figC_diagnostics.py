"""Additional quantitative diagnostics from the frozen evidence ledger.

This figure is appendix-only: it exposes type-level LongMemEval scores,
source-expansion trade-offs, and MEME's deletion boundary without adding
new model runs or inventing timestamps.
"""
from pathlib import Path
import json
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
LEDGER = json.loads((ROOT.parent / "results/evidence_ledger.json").read_text())

BLUE = "#24669B"
TEAL = "#18888B"
ROSE = "#B64C72"
GREEN = "#2A8C68"
INK = "#193D55"
MUTED = "#5C7282"
GRID = "#D5E0E6"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.3,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "xtick.labelsize": 6.1,
    "ytick.labelsize": 6.1,
    "legend.fontsize": 6.0,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
    "axes.linewidth": .65,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": None,
    "savefig.dpi": 300,
    "figure.dpi": 150,
})


def style(ax, axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=GRID, lw=.45, alpha=.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, width=.5, pad=2)


def panel_title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", pad=6)


def export(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    sizes = []
    outside = []
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        sizes.append(t.get_fontsize())
        box = t.get_window_extent(renderer)
        if box.x0 < -18 or box.y0 < -18 or box.x1 > bounds.x1 + 18 or box.y1 > bounds.y1 + 18:
            outside.append(t.get_text())
    assert min(sizes) >= 6, (stem, min(sizes))
    assert not outside, (stem, "outside canvas", outside)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(ROOT / f"{stem}.{ext}", dpi=300, bbox_inches=None)
    with Image.open(ROOT / f"{stem}.png") as im:
        ImageOps.grayscale(im).save(ROOT / f"{stem}_gray.png", dpi=(300, 300))
    print(f"{stem}: {fig.get_size_inches()} in; min font={min(sizes):.1f}pt; bounds PASS")
    plt.close(fig)


def diagnostics():
    lme = LEDGER["four_benchmark"]["longmemeval_full_ingest"]
    exp = LEDGER["source_expansion_diagnostic"]
    meme = LEDGER["meme_limitation"]

    fig, axes = plt.subplots(
        1, 3, figsize=(5.5, 2.18),
        gridspec_kw={"width_ratios": [1.20, 1.00, 1.15]},
    )
    fig.subplots_adjust(left=.12, right=.985, top=.78, bottom=.27, wspace=.35)
    a, b, c = axes

    panel_title(a, "a", "LongMemEval type scores")
    types = [
        ("knowledge-update", "Know. upd."),
        ("multi-session", "Multi-sess."),
        ("single-session-assistant", "Sess. asst."),
        ("single-session-preference", "Sess. pref."),
        ("single-session-user", "Sess. user"),
        ("temporal-reasoning", "Temporal"),
    ]
    tracks = [
        ("v1_baseline", "D-B"),
        ("v2_baseline", "D-A"),
        ("v1_skill-agent", "T-B"),
        ("v2_skill-agent", "T-A"),
        ("v1_pi", "S-B"),
        ("v2_pi", "S-A"),
    ]
    vals = np.array([
        [100 * lme[t]["by_type"][k]["score"] for t, _ in tracks]
        for k, _ in types
    ])
    assert vals.shape == (6, 6)
    a.pcolormesh(
        np.arange(7), np.arange(7), vals, cmap="cividis",
        vmin=0, vmax=100, edgecolors="white", linewidth=.5,
    )
    a.set_xlim(0, 6)
    a.set_ylim(6, 0)
    a.set_xticks(np.arange(6) + .5, [label for _, label in tracks])
    a.set_yticks(np.arange(6) + .5, [label for _, label in types])
    a.tick_params(length=0, pad=2)
    a.spines[:].set_visible(False)
    for i in range(6):
        for j in range(6):
            a.text(j + .5, i + .5, f"{vals[i, j]:.0f}",
                    ha="center", va="center", fontsize=6.0,
                    color="white" if vals[i, j] < 45 else INK)

    panel_title(b, "b", "Source expansion")
    names = [
        ("lexical_semantic_span", "span", BLUE),
        ("neighbors_1", "+1", TEAL),
        ("neighbors_2", "+2", ROSE),
        ("legacy_context_3", "ctx3", MUTED),
    ]
    points = []
    for key, label, color in names:
        z = exp[key]
        x = z["mean_response_bytes"] / 1000.0
        points.append((x, z["recall_any_pct"], z["recall_all_pct"], label, color))
    for metric, marker, color in [
        ("any", "o", BLUE),
        ("all", "s", TEAL),
    ]:
        ys = [p[1] if metric == "any" else p[2] for p in points]
        b.plot([p[0] for p in points[:3]], ys[:3], color=color, lw=.9,
               alpha=.55, zorder=1)
        for (x, _, _, label, point_color), y in zip(points, ys):
            b.scatter(x, y, marker=marker, s=17, color=point_color,
                      edgecolor="white", linewidth=.35, zorder=3)
            if metric == "any":
                dy = 9 if label == "ctx3" else (5 if label == "+1" else 6)
                b.annotate(label, (x, y), xytext=(0, dy),
                           textcoords="offset points", ha="center", fontsize=6.0,
                           color=point_color)
    b.set_xlim(2.75, 3.78)
    b.set_ylim(45, 100)
    b.set_xticks([2.8, 3.2, 3.6])
    b.set_xlabel("Response payload (kB)")
    b.set_ylabel("Recall (%)")
    style(b, "y")
    b.legend(handles=[
        Line2D([], [], marker="o", ls="", color=BLUE, label="Any", ms=3),
        Line2D([], [], marker="s", ls="", color=TEAL, label="All", ms=3),
    ], loc="lower right", frameon=False, handletextpad=.3, borderpad=.1)

    panel_title(c, "c", "MEME boundary")
    tasks = [
        ("ER", "ER", 100),
        ("Tr", "Tr", 100),
        ("Agg", "Agg", 100),
        ("Abs", "Abs", 130),
        ("Cas", "Cas", 164),
        ("Del", "Del", 100),
    ]
    y = np.arange(len(tasks))[::-1]
    vals = [meme["task_accuracy_pct"][key] if key in meme["task_accuracy_pct"]
            else meme["task_accuracy_pct"][key + "_real"]
            for key, _, _ in tasks]
    counts = [n for _, _, n in tasks]
    assert vals[-1] == 0.0
    for yy, val, (key, label, _), n in zip(y, vals, tasks, counts):
        col = ROSE if key == "Del" else GREEN
        c.hlines(yy, 0, val, color=col, lw=2.0, alpha=.32)
        c.scatter(val, yy, s=18, marker="D" if key == "Del" else "o",
                  color=col, edgecolor="white", linewidth=.35, zorder=3)
        c.text(val + 3 if val < 80 else val - 2, yy if val < 80 else yy + .18, f"{val:.1f}% (n={n})", ha="left" if val < 80 else "right", va="center", fontsize=6.0, color=INK)
    c.set_yticks(y, [label for _, label, _ in tasks])
    c.set_xlim(0, 125)
    c.set_xticks([0, 50, 100])
    c.set_xlabel("Real accuracy (%)")
    style(c)

    export(fig, "figC_diagnostics")


if __name__ == "__main__":
    diagnostics()
