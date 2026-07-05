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

from .imaging import MARGIN_MM

DEFAULT_MARGIN_PX = round(MARGIN_MM / 25.4 * 300)

CANVAS_DPI = 300
A4_W = 2480  # 210mm @300dpi
A4_H = 3508  # 297mm @300dpi
MARGIN = 59  # 约 5mm 安全边距
FEATHER = 60  # 贴图边缘羽化宽度（约 5mm）：在 12mm 余量的外侧平坦区完成过渡
WHITE = (255, 255, 255)


def compose_canvas(
    front: Image.Image,
    back: Image.Image,
    scan_dpi: int,
    bg_color: tuple[int, int, int] = WHITE,
    background: Image.Image | None = None,
    margins_px: tuple[int, int] | None = None,
) -> Image.Image:
    """复印件页画布：PDF 与页面上的复印件预览共用同一渲染。

    background 是抹掉证件后的真实扫描背景（见 imaging.make_page_background）；
    缺失时回退平色 bg_color。bg_color 同时用于旋转/缩放的填充色。
    margins_px 是两面裁剪的实际余量（画布 DPI 像素），决定羽化宽度与
    「超出半页先裁余量保 1:1」的可裁量；证件贴边扫描时余量会小于默认值。
    """
    if background is not None:
        canvas = background.convert("RGB")
        if canvas.size != (A4_W, A4_H):
            canvas = canvas.resize((A4_W, A4_H), Image.LANCZOS)
    else:
        canvas = Image.new("RGB", (A4_W, A4_H), bg_color)
    front_margin, back_margin = margins_px or (DEFAULT_MARGIN_PX, DEFAULT_MARGIN_PX)
    _place(canvas, front, scan_dpi, top=0, bg_color=bg_color, margin_px=front_margin)
    _place(canvas, back, scan_dpi, top=A4_H // 2, bg_color=bg_color, margin_px=back_margin)
    return canvas


def compose_pdf(
    front: Image.Image,
    back: Image.Image,
    scan_dpi: int,
    bg_color: tuple[int, int, int] = WHITE,
    background: Image.Image | None = None,
    margins_px: tuple[int, int] | None = None,
) -> bytes:
    buf = io.BytesIO()
    canvas = compose_canvas(front, back, scan_dpi, bg_color, background, margins_px)
    canvas.save(buf, "PDF", resolution=float(CANVAS_DPI))
    return buf.getvalue()


def _place(
    canvas: Image.Image,
    img: Image.Image,
    scan_dpi: int,
    top: int,
    bg_color: tuple[int, int, int],
    margin_px: int,
) -> None:
    half_h = A4_H // 2
    if scan_dpi != CANVAS_DPI:
        # 统一到画布 DPI，物理尺寸不变
        ratio = CANVAS_DPI / scan_dpi
        img = img.resize(
            (max(1, round(img.width * ratio)), max(1, round(img.height * ratio))),
            Image.LANCZOS,
        )
        margin_px = round(margin_px * ratio)
    fit_w, fit_h = A4_W - 2 * MARGIN, half_h - 2 * MARGIN
    if img.width > fit_w or img.height > fit_h:
        # 超出半页时先裁掉裁剪余量保 1:1（余量是环境装饰，证件尺寸才是硬指标；
        # 只允许裁掉实际余量，防止裁进证件本体）
        trim = 2 * margin_px
        crop_w = min(img.width, max(fit_w, img.width - trim))
        crop_h = min(img.height, max(fit_h, img.height - trim))
        if (crop_w, crop_h) != (img.width, img.height):
            dx = (img.width - crop_w) // 2
            dy = (img.height - crop_h) // 2
            img = img.crop((dx, dy, dx + crop_w, dy + crop_h))
            margin_px = max(0, margin_px - max(dx, dy))
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
    feather = min(FEATHER, margin_px)  # 羽化只能吃余量，不能把证件边缘变半透明
    if feather > 0:
        canvas.paste(img, (x, y), _feather_mask(img.width, img.height, feather))
    else:
        canvas.paste(img, (x, y))


def _feather_mask(w: int, h: int, feather: int) -> Image.Image:
    """边缘线性淡出的 alpha 遮罩：贴图外缘 feather 像素渐变融入画布底色。"""
    yy = np.minimum(np.arange(h), np.arange(h)[::-1])
    xx = np.minimum(np.arange(w), np.arange(w)[::-1])
    dist = np.minimum.outer(yy, xx)  # 每个像素到最近边缘的距离
    alpha = np.clip((dist + 1) / feather, 0.0, 1.0) * 255
    return Image.fromarray(alpha.astype(np.uint8), mode="L")
