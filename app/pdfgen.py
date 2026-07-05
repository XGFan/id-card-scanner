"""复印件页合成：A4 纵向 300dpi 画布，正面 1:1 居中上半页、反面下半页。

1:1 的含义：证件图 w×h 像素 @scan_dpi，在纸上渲染为 w/scan_dpi × h/scan_dpi 英寸。
画布 2480×3508 @300dpi，save(resolution=300) 得到 595.3×841.9pt 的标准 A4 页。

页面底色使用原始扫描图的背景采样色（而非纯白），证件图带着 3mm 自然背景
余量贴上来，边界不显突兀，整页观感像一张真实复印件（迭代 2 用户反馈）。
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

CANVAS_DPI = 300
A4_W = 2480  # 210mm @300dpi
A4_H = 3508  # 297mm @300dpi
MARGIN = 59  # 约 5mm 安全边距
FEATHER = 18  # 贴图边缘羽化宽度（约 1.5mm）：色调残差不露接缝，且不侵入证件本体
WHITE = (255, 255, 255)


def compose_pdf(
    front: Image.Image,
    back: Image.Image,
    scan_dpi: int,
    bg_color: tuple[int, int, int] = WHITE,
) -> bytes:
    canvas = Image.new("RGB", (A4_W, A4_H), bg_color)
    _place(canvas, front, scan_dpi, top=0, bg_color=bg_color)
    _place(canvas, back, scan_dpi, top=A4_H // 2, bg_color=bg_color)
    buf = io.BytesIO()
    canvas.save(buf, "PDF", resolution=float(CANVAS_DPI))
    return buf.getvalue()


def _place(
    canvas: Image.Image,
    img: Image.Image,
    scan_dpi: int,
    top: int,
    bg_color: tuple[int, int, int],
) -> None:
    half_h = A4_H // 2
    if scan_dpi != CANVAS_DPI:
        # 统一到画布 DPI，物理尺寸不变
        ratio = CANVAS_DPI / scan_dpi
        img = img.resize(
            (max(1, round(img.width * ratio)), max(1, round(img.height * ratio))),
            Image.LANCZOS,
        )
    fit_w, fit_h = A4_W - 2 * MARGIN, half_h - 2 * MARGIN
    if img.height > fit_h and img.height <= fit_w and img.width <= fit_h:
        # 竖放放不下但横放能 1:1 放下 → 旋转，保住真实尺寸
        img = img.rotate(90, expand=True, fillcolor=bg_color)
    if img.width > fit_w or img.height > fit_h:
        # 兜底（找边失败回退整幅原图时）：缩放放入，此时放弃 1:1
        scale = min(fit_w / img.width, fit_h / img.height)
        img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.LANCZOS,
        )
    x = (A4_W - img.width) // 2
    y = top + (half_h - img.height) // 2
    canvas.paste(img, (x, y), _feather_mask(img.width, img.height))


def _feather_mask(w: int, h: int) -> Image.Image:
    """边缘线性淡出的 alpha 遮罩：贴图外缘 FEATHER 像素渐变融入画布底色。"""
    yy = np.minimum(np.arange(h), np.arange(h)[::-1])
    xx = np.minimum(np.arange(w), np.arange(w)[::-1])
    dist = np.minimum.outer(yy, xx)  # 每个像素到最近边缘的距离
    alpha = np.clip((dist + 1) / FEATHER, 0.0, 1.0) * 255
    return Image.fromarray(alpha.astype(np.uint8), mode="L")
