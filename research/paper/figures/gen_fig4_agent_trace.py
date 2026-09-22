"""Export the hand-authored Figure 3 SVG; SVG is the source of truth.

Do not replace this diagram with the obsolete Matplotlib reconstruction.
The diagram records LongMemEval question 2b8f3739, including the actual
step-limit ending. Numerical data and diagram semantics live in the SVG.
"""
from pathlib import Path
import subprocess
import tempfile
import cairosvg
from PIL import Image, ImageOps


def main():
    root = Path(__file__).resolve().parent
    source = root / "fig4_agent_trace.svg"
    svg = source.read_bytes()
    assert b"4 episode pointers" in svg
    assert b"Step limit" in svg
    assert b"Recorded sequence" in svg
    with tempfile.TemporaryDirectory(prefix="dd-trace-export-") as tmp:
        pdf = Path(tmp) / "trace.pdf"
        cairosvg.svg2pdf(bytestring=svg, write_to=str(pdf))
        subprocess.run(["gs", "-q", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite",
                        "-dCompatibilityLevel=1.4",
                        "-sOutputFile=" + str(root / "fig4_agent_trace.pdf"), str(pdf)], check=True)
    png = root / "fig4_agent_trace.png"
    cairosvg.svg2png(bytestring=svg, write_to=str(png), output_width=1650, output_height=825)
    with Image.open(png) as im:
        im.save(png, dpi=(300, 300))
        ImageOps.grayscale(im).save(root / "fig4_agent_trace_gray.png", dpi=(300, 300))
    print("Figure 3 restored from authored SVG (5.5 x 2.75 in).")


if __name__ == "__main__":
    main()
