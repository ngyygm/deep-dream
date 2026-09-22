"""Deterministic geometry QA for fig3: text/text and text/legend overlap +
out-of-canvas checks.  Renders the exact same figure gen_fig3_cost.make_figure()
builds (save is intercepted before export), so positions are ground truth.

Usage: python qa_fig3_geometry.py   (from research/paper/figures)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

import gen_fig3_cost as g

REPORT: list[str] = []


def collect(fig):
    """Collect visible text bboxes.  Tick labels come from the per-axis
    in-view lists (get_xticklabels/get_yticklabels), NOT findobj — findobj
    also returns stale tick instances left over after set_ylim clipped them,
    which produces phantom out-of-canvas/overlap reports."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    items = []
    for i, ax in enumerate(fig.axes):
        texts = list(ax.texts) + list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        texts += [ax.xaxis.label, ax.yaxis.label]
        leg = ax.get_legend()
        if leg is not None:
            texts += [t for t in leg.findobj(mpl.text.Text) if t.get_visible()]
        for t in texts:
            s = t.get_text().strip()
            if s and t.get_visible() and t.get_alpha() != 0:
                items.append((f"[ax{i}] {s.replace(chr(10), '|')[:26]}",
                              t.get_window_extent(renderer)))
    return items


def pairwise_overlaps(fig) -> None:
    boxes = collect(fig)
    fb = fig.bbox
    for label, bb in boxes:
        if not fb.contains(bb.x0, bb.y0) or not fb.contains(bb.x1, bb.y1):
            REPORT.append(f"OUT-OF-CANVAS: '{label}' bbox={bb}")
    n_hit = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            la, ba = boxes[i]
            lb, bb2 = boxes[j]
            if ba.overlaps(bb2.shrunk(0.92, 0.80)):
                REPORT.append(f"TEXT-OVERLAP: '{la}' <-> '{lb}'")
                n_hit += 1
    if n_hit == 0:
        REPORT.append("no text-text overlaps")


def legend_geometry(fig) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(renderer)
        for t in fig.findobj(mpl.text.Text):
            if t.get_text().strip() and t.get_visible() and t not in leg.findobj():
                tb = t.get_window_extent(renderer)
                if lb.overlaps(tb):
                    REPORT.append(
                        f"LEGEND-COVERS-TEXT: {t.get_text()[:24]!r} on ax{id(ax) % 1000}"
                    )


def main() -> None:
    real_save = g.save

    def qa_save(fig, name):
        pairwise_overlaps(fig)
        legend_geometry(fig)
        fig.savefig("_qa_fig3_preview.png", dpi=200)
        plt.close(fig)

    g.save = qa_save
    g.make_figure()
    g.save = real_save
    print("\n".join(REPORT))


if __name__ == "__main__":
    main()
