"""Deterministic geometry QA for fig4: text/text overlap + out-of-canvas
checks.  Renders the exact same figure gen_fig4_agent_trace.main() builds
(save is intercepted before export).  The axis is off, so the only text
sources are ax.texts.

Usage: python qa_fig4_geometry.py   (from research/paper/figures)
"""

import matplotlib.pyplot as plt

import gen_fig4_agent_trace as g

REPORT: list[str] = []


def pairwise_overlaps(fig) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for i, ax in enumerate(fig.axes):
        for t in ax.texts:
            s = t.get_text().strip()
            if s and t.get_visible() and t.get_alpha() != 0:
                boxes.append((f"[ax{i}] {s.replace(chr(10), '|')[:30]}",
                              t.get_window_extent(renderer)))
    fb = fig.bbox
    for label, bb in boxes:
        if not fb.contains(bb.x0, bb.y0) or not fb.contains(bb.x1, bb.y1):
            REPORT.append(f"OUT-OF-CANVAS: '{label}' bbox={bb}")
    n_hit = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            la, ba = boxes[i]
            lb, bb2 = boxes[j]
            if ba.overlaps(bb2.shrunk(0.90, 0.75)):
                REPORT.append(f"TEXT-OVERLAP: '{la}' <-> '{lb}'")
                n_hit += 1
    if n_hit == 0:
        REPORT.append("no text-text overlaps")


def main() -> None:
    real_save = g.save

    def qa_save(fig, name):
        pairwise_overlaps(fig)
        fig.savefig("_qa_fig4_preview.png", dpi=200)
        plt.close(fig)

    g.save = qa_save
    g.main()
    g.save = real_save
    print("\n".join(REPORT))


if __name__ == "__main__":
    main()
