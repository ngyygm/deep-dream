"""Embed the approved generated architecture artwork without resampling it.

Run on the authoritative paper host. The PNG is the editable image-generation
asset; the adjacent prompt records the design specification. Revisions should
update the PNG and rerun this script, rather than restoring the older diagram.
"""
from pathlib import Path
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

root = Path(__file__).resolve().parent
source = root / "fig1_architecture.png"
target = root / "fig1_architecture.pdf"
with Image.open(source) as im:
    px_w, px_h = im.size
width = 468.0
height = width * px_h / px_w
pdf = canvas.Canvas(str(target), pagesize=(width, height))
pdf.setTitle("Deep-Dream: adaptive reading and temporal concept memory")
pdf.setAuthor("Deep-Dream")
pdf.drawImage(ImageReader(str(source)), 0, 0, width=width, height=height)
pdf.showPage()
pdf.save()
print(f"Embedded {px_w}x{px_h} image at {px_w / 6.5:.0f} dpi; PDF {width:.1f}x{height:.1f} pt")
