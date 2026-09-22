"""Export the selected ImageGen first-page figure as a lossless PDF wrapper.

Figure 1 was generated with the built-in ImageGen tool using
fig1_architecture.png (paper Figure 2) as the visual reference.
The exact prompt is archived in fig0_motivation.prompt.txt.
This script does not regenerate or alter the artwork. It embeds the selected
PNG without resampling, so LaTeX can include the same artifact consistently.
Requires reportlab. Run: python figures/gen_fig0_motivation.py
"""
from pathlib import Path
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def main():
    folder = Path(__file__).resolve().parent
    source = folder / "fig0_motivation.png"
    target = folder / "fig0_motivation.pdf"
    reader = ImageReader(str(source))
    pixels_w, pixels_h = reader.getSize()
    width = 6.4 * 72
    height = width * pixels_h / pixels_w
    pdf = canvas.Canvas(str(target), pagesize=(width, height))
    pdf.setTitle("Deep-Dream: evidence-path motivation")
    pdf.setSubject("First-page motivation figure; companion style to Figure 2")
    pdf.drawImage(reader, 0, 0, width=width, height=height, mask="auto")
    pdf.showPage()
    pdf.save()
    print(f"Exported {target.name}: {pixels_w} x {pixels_h} pixels, without resampling")


if __name__ == "__main__":
    main()
