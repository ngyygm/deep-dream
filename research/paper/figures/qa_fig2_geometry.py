"""Deterministic geometry QA for fig2: text/text and text/legend overlap +
out-of-canvas checks. Renders the exact same figure gen_fig2_ladder.main()
builds (save is intercepted before export), so positions are ground truth.

Usage: python qa_fig2_geometry.py   (from research/paper/figures)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

import gen_fig2_ladder as g

REPORT: list[str] = []


def pairwise_overlaps(fig) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    items: list[tuple[str, mpl.text.Text]] = []
    ax_of = {}
    for ax in fig.axes:
        for t in ax.findobj(mpl.text.Text):
            ax_of[id(t)] = (ax.get_title() or "no-title").split(")")[0] + ")"
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text().strip()
        if s and t.get_visible() and t.get_alpha() != 0:
            tag = ax_of.get(id(t), "fig-level")
            items.append((f"[{tag}] {s.replace(chr(10), '|')[:26]}", t))
    boxes = [(label, t.get_window_extent(renderer)) for label, t in items]

    # any artist outside the canvas?
    fb = fig.bbox
    for label, bb in boxes:
        if not fb.contains(bb.x0, bb.y0) or not fb.contains(bb.x1, bb.y1):
            REPORT.append(f"OUT-OF-CANVAS: '{label}' bbox={bb}")

    # pairwise text intersections (shared-edge touching tolerated via +1px shrink)
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
        ab = ax.get_window_extent(renderer)
        where = "BELOW" if lb.y1 <= ab.y0 else ("ABOVE" if lb.y0 >= ab.y1 else "INSIDE/SIDE")
        REPORT.append(
            f"legend on ax {ax.get_title()[:20]!r}: {where} axes "
            f"(legend y[{lb.y0:.0f},{lb.y1:.0f}] vs axes y[{ab.y0:.0f},{ab.y1:.0f}])"
        )
        # legend must not cover any text artist
        for t in fig.findobj(mpl.text.Text):
            if t.get_text().strip() and t.get_visible() and t not in leg.findobj():
                tb = t.get_window_extent(renderer)
                if lb.overlaps(tb):
                    REPORT.append(
                        f"LEGEND-COVERS-TEXT: {t.get_text()[:24]!r} "
                        f"on {ax.get_title()[:20]!r}"
                    )


def main() -> None:
    real_save = g.save

    def qa_save(fig, name):
        pairwise_overlaps(fig)
        legend_geometry(fig)
        fig.savefig("_qa_fig2_r2_preview.png", dpi=200)
        plt.close(fig)

    g.save = qa_save
    g.main()
    g.save = real_save
    print("\n".join(REPORT))


if __name__ == "__main__":
    main()
