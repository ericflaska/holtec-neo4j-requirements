from pathlib import Path

import fitz
from PIL import Image

from config import MAX_PIXELS


def load_pdf(path: str | Path) -> fitz.Document:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    return fitz.open(path)


def page_to_pil(doc: fitz.Document, page_no: int, scale: float = 2.0) -> Image.Image:
    page = doc[page_no]
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def stitch_two_pages(doc: fitz.Document, first_page: int = 0, scale: float = 2.0, max_pixels: int | None = None) -> Image.Image:
    max_pixels = max_pixels or MAX_PIXELS
    p0 = page_to_pil(doc, first_page, scale=scale)
    p1 = page_to_pil(doc, first_page + 1, scale=scale)
    w = max(p0.width, p1.width)
    h0, h1 = p0.height, p1.height
    if p0.width != w or p1.width != w:
        p0 = p0.resize((w, h0), Image.Resampling.LANCZOS)
        p1 = p1.resize((w, h1), Image.Resampling.LANCZOS)
    stitched = Image.new("RGB", (w, h0 + h1))
    stitched.paste(p0, (0, 0))
    stitched.paste(p1, (0, h0))
    total = stitched.width * stitched.height
    if total > max_pixels:
        ratio = (max_pixels / total) ** 0.5
        stitched = stitched.resize((int(stitched.width * ratio), int(stitched.height * ratio)), Image.Resampling.LANCZOS)
    return stitched


def stitch_pdf_pages(pdf_path: str | Path, first_page: int = 0, scale: float = 2.0) -> tuple[Image.Image, list[int]]:
    doc = load_pdf(pdf_path)
    if first_page + 1 >= len(doc):
        raise ValueError(f"PDF has {len(doc)} pages; need at least {first_page + 2}")
    img = stitch_two_pages(doc, first_page=first_page, scale=scale)
    doc.close()
    return img, [first_page + 1, first_page + 2]
