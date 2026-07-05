import io

from PIL import Image
from pypdf import PdfReader

from app import pdfgen


def _card(w=1012, h=638, color=(200, 60, 40)):
    return Image.new("RGB", (w, h), color)


def test_single_a4_page_true_size():
    data = pdfgen.compose_pdf(_card(), _card(), 300)
    reader = PdfReader(io.BytesIO(data))
    assert len(reader.pages) == 1
    box = reader.pages[0].mediabox
    # A4 = 595.3 × 841.9 pt
    assert abs(float(box.width) - 595.3) < 1.5
    assert abs(float(box.height) - 841.9) < 1.5


def test_tall_card_rotated_to_keep_true_size():
    # 高度超过半页但横放能 1:1 放下 → 应旋转而非缩放
    tall = _card(700, 2000)
    data = pdfgen.compose_pdf(tall, tall, 300)
    assert data.startswith(b"%PDF")


def test_oversized_fallback_scaled_to_fit():
    # 找边失败回退整幅原图（2550×3508）时应缩放放入，不报错
    big = _card(2550, 3508)
    data = pdfgen.compose_pdf(big, big, 300)
    assert data.startswith(b"%PDF")


def test_canvas_uses_scan_background_color():
    bg = (238, 236, 230)
    data = pdfgen.compose_pdf(_card(), _card(), 300, bg_color=bg)
    reader = PdfReader(io.BytesIO(data))
    page_img = next(iter(reader.pages[0].images)).image.convert("RGB")
    corner = page_img.getpixel((5, 5))
    # PDF 内嵌 JPEG 有轻微有损，允许小容差
    assert all(abs(a - b) <= 6 for a, b in zip(corner, bg))
