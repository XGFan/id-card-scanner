"""从原始扫描图中定位证件图：找边 → 外扩余量 → 裁剪 → 透视摆正。

已知难点是白色证件放在白色盖板背景上的低对比度场景：
主路用 Canny 边缘 + 膨胀闭合找轮廓；不行再用 Otsu 阈值兜底
（盖板打开的黑背景、深色证件都走这条路很可靠）；
全部失败则返回原图并标记 detected=False，由用户在预览里单面重扫。

裁剪向外保留 MARGIN_MM 的自然背景余量，配合复印件页使用扫描背景采样色
作底，整页观感接近一张真实的复印件（迭代 2 用户反馈）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

MIN_CARD_CM2 = 6.0      # 小于 6cm² 的轮廓不可能是证件
A5_CM2 = 14.8 * 21.0    # 证件面积上限：半张 A4（A5），另留 10% 容差
MIN_SIDE_CM = 2.0       # 证件短边至少 2cm
MIN_FILL_RATIO = 0.72   # 轮廓面积 / 最小外接旋转矩形面积，过滤非实心矩形
GRAD_CLOSE_MM = 5.0     # 梯度兜底：闭合核尺寸（按 mm，随 DPI 缩放）。需足够大
                        # 以桥接白底白卡时稀疏破碎的证件边缘，使填洞后连成整卡
MARGIN_MM = 12.0        # 裁剪余量：足够大，让卡片周围的反光光晕在余量内自然衰减
                        # 到平坦背景，接缝落在平坦区、配合宽羽化即不可见（迭代 9）
EDGE_KEEPOUT_MM = 3.0   # 玻璃板边缘的白色校准带宽度：余量不得伸入该区域
                        # （宽证件≈A4 宽时余量会顶到原图边界，白边即由此而来）


@dataclass
class DetectResult:
    image: np.ndarray                 # BGR：检测成功为摆正后的证件图（含余量），失败为原图
    detected: bool
    bg_color: tuple[int, int, int]    # 扫描背景采样色（BGR 中位数），供复印件页做底色
    quad: np.ndarray | None = None    # 检测到的证件边界四角（原始扫描图坐标，4×2），未检出为 None
    margin_mm: float = 0.0            # 本次裁剪实际使用的余量（证件贴边时会小于 MARGIN_MM）


def detect_card(bgr: np.ndarray, dpi: int) -> DetectResult:
    px_per_cm = dpi / 2.54
    min_area = MIN_CARD_CM2 * px_per_cm**2
    max_area = A5_CM2 * 1.10 * px_per_cm**2
    min_side = MIN_SIDE_CM * px_per_cm

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    for mask in _candidate_masks(blurred, dpi):
        rect = _best_rect(mask, min_area, max_area, min_side)
        if rect is not None:
            margin = _clamp_margin(rect, bgr.shape, dpi)
            expanded = _expand(rect, margin)
            return DetectResult(
                _warp(bgr, expanded),
                True,
                _estimate_bg(bgr, expanded, dpi),
                quad=cv2.boxPoints(rect),
                margin_mm=margin / dpi * 25.4,
            )
    return DetectResult(bgr, False, _estimate_bg(bgr, None, dpi))


def _clamp_margin(rect: cv2.typing.RotatedRect, shape: tuple, dpi: int) -> float:
    """余量按证件到图像边界的距离收缩，且不伸入玻璃板边缘的白色校准带。

    1.42 (≥√2) 是旋转矩形外扩时角点位移的轴向分量上界，保证外扩后仍在界内。
    """
    h, w = shape[:2]
    box = cv2.boxPoints(rect)
    dist = min(
        float(box[:, 0].min()),
        float(box[:, 1].min()),
        float(w - 1 - box[:, 0].max()),
        float(h - 1 - box[:, 1].max()),
    )
    keepout = EDGE_KEEPOUT_MM / 25.4 * dpi
    allowed = max(0.0, (dist - keepout) / 1.42)
    return min(MARGIN_MM / 25.4 * dpi, allowed)


def _candidate_masks(blurred: np.ndarray, dpi: int) -> Iterator[np.ndarray]:
    kernel = np.ones((5, 5), np.uint8)
    # 1) Canny：对白底白卡的弱边缘最敏感
    edges = cv2.Canny(blurred, 30, 120)
    yield cv2.dilate(edges, kernel, iterations=2)
    # 2) Otsu 双向阈值：高对比场景（黑背景 / 深色证件）
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=3)
    yield closed
    yield cv2.bitwise_not(closed)
    # 3) 兜底：梯度幅值 + 闭合填洞。白底白卡放白盖板上边缘极弱，Canny 轮廓
    #    围不成实心（fill 过低被拒）、Otsu 又分不开背景；但证件内部图文/logo/
    #    芯片有丰富梯度，闭合桥接稀疏边缘后填洞，连同无梯度的白边与圆角一起
    #    实心化为完整证件区域（交行 UnionPay 卡回归）。
    yield _gradient_fill_mask(blurred, dpi)


def _gradient_fill_mask(blurred: np.ndarray, dpi: int) -> np.ndarray:
    gx = cv2.Scharr(blurred, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(blurred, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag = (mag / (mag.max() + 1e-6) * 255).astype(np.uint8)  # 相对最强边缘归一
    _, mask = cv2.threshold(mag, 15, 255, cv2.THRESH_BINARY)
    k = int(round(GRAD_CLOSE_MM / 25.4 * dpi)) | 1  # 奇数核，随 DPI 缩放
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=2)
    return _fill_holes(closed)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """从图像角点（必为背景）洪泛，取反并回填，把闭合边界内的空洞实心化。"""
    h, w = mask.shape
    flood = mask.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    return mask | cv2.bitwise_not(flood)


def _best_rect(
    mask: np.ndarray, min_area: float, max_area: float, min_side: float
) -> cv2.typing.RotatedRect | None:
    # RETR_LIST 而非 EXTERNAL：玻璃板边缘阴影可能形成包住证件的外框轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if min(w, h) < min_side:
            continue
        rect_area = w * h
        if rect_area > max_area or area / rect_area < MIN_FILL_RATIO:
            continue
        if rect_area > best_area:
            best, best_area = rect, rect_area
    return best


def _expand(rect: cv2.typing.RotatedRect, margin: float) -> cv2.typing.RotatedRect:
    (cx, cy), (w, h), angle = rect
    return ((cx, cy), (w + 2 * margin, h + 2 * margin), angle)


BG_RING_MM = 4.0  # 底色采样环带宽度：紧贴裁剪边界外侧，匹配接缝处的实际色调


def _estimate_bg(
    bgr: np.ndarray, rect: cv2.typing.RotatedRect | None, dpi: int
) -> tuple[int, int, int]:
    """扫描背景采样色（中位数，降采样加速）。

    有证件时只采裁剪边界外侧 BG_RING_MM 的环带——卡片阴影使近处背景比远处
    亮区偏暗，用全图中位数做画布底色会在贴图边界露出色差接缝。
    """
    small = bgr[::8, ::8]
    if rect is not None:
        inner = np.zeros(bgr.shape[:2], np.uint8)
        cv2.fillPoly(inner, [cv2.boxPoints(rect).astype(np.int32)], 255)
        outer = np.zeros(bgr.shape[:2], np.uint8)
        ring_rect = _expand(rect, margin=BG_RING_MM / 25.4 * dpi)
        cv2.fillPoly(outer, [cv2.boxPoints(ring_rect).astype(np.int32)], 255)
        ring = (outer[::8, ::8] > 0) & (inner[::8, ::8] == 0)
        pixels = small[ring]
        if len(pixels) < 100:  # 证件贴着玻璃板边缘时环带太小，退化为证件外全图
            pixels = small[inner[::8, ::8] == 0]
        if len(pixels) < 100:
            pixels = small.reshape(-1, 3)
    else:
        pixels = small.reshape(-1, 3)
    median = np.median(pixels, axis=0)
    return tuple(int(c) for c in median)


def _warp(bgr: np.ndarray, rect: cv2.typing.RotatedRect) -> np.ndarray:
    box = cv2.boxPoints(rect).astype(np.float32)
    tl, tr, br, bl = _order_corners(box)
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    # 外扩后的区域可能越过图像边界，用边缘复制填充避免黑边
    return cv2.warpPerspective(
        bgr, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
    )


INPAINT_PAD_MM = 8.0  # 抹除区域比检测框外扩的量：盖住裁剪余量与卡片周围的反光光晕
_BLEND_BAND_ROWS = 512  # 羽化混合的分带高度：限制 float32 中间量的瞬时驻留（控内存峰值）


def make_page_background(
    bgr: np.ndarray, quad: np.ndarray, dpi: int, size: tuple[int, int]
) -> np.ndarray:
    """复印件页的真实背景：把原始扫描图上的证件区域 inpaint 抹掉，适配到页面尺寸。

    平色画布与带真实光晕/纹理的裁剪余量放在一起必然露出拼接痕迹；
    用同一次扫描的真实背景（含渐晕、噪点、光晕）做底，贴上去的裁剪
    才像「两面一起放在玻璃板上复印了一次」。
    """
    h, w = bgr.shape[:2]
    pad = INPAINT_PAD_MM / 25.4 * dpi
    rect = cv2.minAreaRect(quad.astype(np.float32))
    hole = cv2.boxPoints(_expand(rect, pad)).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [hole], 255)

    # 逐列线性插值填洞（1/16 分辨率）：扫描背景的结构是纵向条纹（扫描头
    # 纵向走），用洞上/下方的真实像素按列插值，竖向明暗带能自然穿过洞延续。
    # 扩散式 inpaint（Telea）会把洞边界的暗带横向漫进填充区，形成灰色云斑
    small = cv2.resize(bgr, (w // 16, h // 16), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, (w // 16, h // 16), interpolation=cv2.INTER_NEAREST)
    small_mask = cv2.dilate(small_mask, np.ones((3, 3), np.uint8))  # 盖住降采样边缘混色
    filled = _fill_columns(small, small_mask)
    filled = cv2.GaussianBlur(filled, (5, 5), 0)
    filled_up = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)

    # 羽化过渡：抹除区边界不留硬缝。逐带做 alpha 混合——整幅 2550×3508 一次性
    # float32 混合会瞬时驻留几百 MB 中间量，撑爆 512Mi 容器内存（扫完第二面自动
    # 切复印件预览、触发本函数时 OOMKilled 的根因）。每像素独立，分带结果不变。
    alpha = cv2.GaussianBlur(mask, (31, 31), 0)  # uint8 (h, w)，先整幅算好羽化再切带
    out = np.empty_like(bgr)
    for y0 in range(0, h, _BLEND_BAND_ROWS):
        y1 = min(y0 + _BLEND_BAND_ROWS, h)
        a = (alpha[y0:y1].astype(np.float32) / 255)[..., None]
        band = bgr[y0:y1].astype(np.float32) * (1 - a) + filled_up[y0:y1].astype(np.float32) * a
        out[y0:y1] = band.astype(np.uint8)
    return _fit_to(out, *size)


def _fill_columns(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把 mask 区域按列用上/下边界像素线性插值填充（矩形洞每列连续）。"""
    h, w = mask.shape
    out = img.astype(np.float32)
    for x in range(w):
        ys = np.nonzero(mask[:, x])[0]
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max())
        has_top, has_bottom = y0 > 0, y1 + 1 < h
        if has_top and has_bottom:
            top, bottom = out[y0 - 1, x], out[y1 + 1, x]
        elif has_top:
            top = bottom = out[y0 - 1, x]
        elif has_bottom:
            top = bottom = out[y1 + 1, x]
        else:  # 整列都是洞：用行方向邻居兜底
            continue
        t = np.linspace(0.0, 1.0, y1 - y0 + 1)[:, None]
        out[y0 : y1 + 1, x] = top * (1 - t) + bottom * t
    return out.astype(np.uint8)


def _fit_to(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """居中裁剪 / 边缘复制填充到目标尺寸（扫描幅面与 A4 画布略有差异）。"""
    h, w = img.shape[:2]
    if w >= target_w:
        x0 = (w - target_w) // 2
        img = img[:, x0 : x0 + target_w]
    else:
        left = (target_w - w) // 2
        img = cv2.copyMakeBorder(
            img, 0, 0, left, target_w - w - left, cv2.BORDER_REPLICATE
        )
    h, w = img.shape[:2]
    if h >= target_h:
        y0 = (h - target_h) // 2
        img = img[y0 : y0 + target_h]
    else:
        top = (target_h - h) // 2
        img = cv2.copyMakeBorder(
            img, top, target_h - h - top, 0, 0, cv2.BORDER_REPLICATE
        )
    return img


def _order_corners(pts: np.ndarray) -> tuple[np.ndarray, ...]:
    # 按质心极角排序（图像坐标系 y 向下，角度升序即视觉顺时针），
    # 再旋转使 x+y 最小的点为左上角。sum/diff 极值法在 45° 近正方形上会退化出重复角点。
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    return ordered[0], ordered[1], ordered[2], ordered[3]
