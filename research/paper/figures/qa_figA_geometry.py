"""Deterministic geometry QA for figA: text/text and text/legend overlap +
out-of-canvas checks.  Renders the exact same figure
gen_figA_alignment.make_figure() builds (save is intercepted before export).

Usage: python qa_figA_geometry.py   (from research/paper/figures)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

import gen_figA_alignment as g

REPORT: list[str] = []


def collect(fig):
    """Collect visible text bboxes.  Tick labels come from the per-axis
    in-view lists (get_xticklabels/get_yticklabels), NOT findobj — findobj
    also returns stale tick instances left over after clipping, which
    produces phantom out-of-canvas/overlap reports."""
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


def main() -> None:
    real_save = g.save

    def qa_save(fig, name):
        pairwise_overlaps(fig)
        fig.savefig("_qa_figA_preview.png", dpi=200)
        plt.close(fig)

    g.save = qa_save
    g.make_figure()
    g.save = real_save
    print("\n".join(REPORT))


if __name__ == "__main__":
    main()
