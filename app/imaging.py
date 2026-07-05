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
MARGIN_MM = 3.0         # 裁剪向外扩的余量：保留证件周围的自然背景与阴影


@dataclass
class DetectResult:
    image: np.ndarray                 # BGR：检测成功为摆正后的证件图（含余量），失败为原图
    detected: bool
    bg_color: tuple[int, int, int]    # 扫描背景采样色（BGR 中位数），供复印件页做底色
    quad: np.ndarray | None = None    # 检测到的证件边界四角（原始扫描图坐标，4×2），未检出为 None


def detect_card(bgr: np.ndarray, dpi: int) -> DetectResult:
    px_per_cm = dpi / 2.54
    min_area = MIN_CARD_CM2 * px_per_cm**2
    max_area = A5_CM2 * 1.10 * px_per_cm**2
    min_side = MIN_SIDE_CM * px_per_cm

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    for mask in _candidate_masks(blurred):
        rect = _best_rect(mask, min_area, max_area, min_side)
        if rect is not None:
            expanded = _expand(rect, margin=MARGIN_MM / 25.4 * dpi)
            return DetectResult(
                _warp(bgr, expanded),
                True,
                _estimate_bg(bgr, expanded, dpi),
                quad=cv2.boxPoints(rect),
            )
    return DetectResult(bgr, False, _estimate_bg(bgr, None, dpi))


def _candidate_masks(blurred: np.ndarray) -> Iterator[np.ndarray]:
    kernel = np.ones((5, 5), np.uint8)
    # 1) Canny：对白底白卡的弱边缘最敏感
    edges = cv2.Canny(blurred, 30, 120)
    yield cv2.dilate(edges, kernel, iterations=2)
    # 2) Otsu 双向阈值：高对比场景（黑背景 / 深色证件）
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=3)
    yield closed
    yield cv2.bitwise_not(closed)


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

    # 背景是低频内容：1/16 分辨率 inpaint + 强模糊。分辨率高了 Telea 会沿
    # 等照度线扩散出 X 形方向性条纹；低分辨率 + 模糊得到平滑渐变填充
    small = cv2.resize(bgr, (w // 16, h // 16), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, (w // 16, h // 16), interpolation=cv2.INTER_NEAREST)
    small_mask = cv2.dilate(small_mask, np.ones((3, 3), np.uint8))  # 盖住降采样边缘混色
    filled = cv2.inpaint(small, small_mask, 4, cv2.INPAINT_TELEA)
    filled = cv2.GaussianBlur(filled, (9, 9), 0)
    filled_up = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)

    # 羽化过渡：抹除区边界不留硬缝
    alpha = (cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32) / 255)[..., None]
    out = (bgr.astype(np.float32) * (1 - alpha) + filled_up.astype(np.float32) * alpha)
    return _fit_to(out.astype(np.uint8), *size)


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
