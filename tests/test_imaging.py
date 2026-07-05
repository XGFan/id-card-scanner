import cv2
import numpy as np

from app.imaging import detect_card

DPI = 300
# 身份证 85.6×54mm @300dpi ≈ 1012×638px
CARD_W, CARD_H = 1012, 638
EXPAND = round(2 * 3.0 / 25.4 * DPI)  # 裁剪每边向外扩 3mm 余量，两边共 ≈71px
BG = 242


def _platen(color: int) -> np.ndarray:
    return np.full((3508, 2550, 3), color, dtype=np.uint8)


def _draw_card(img, center, angle, color):
    box = cv2.boxPoints((center, (CARD_W, CARD_H), angle)).astype(np.int32)
    cv2.fillPoly(img, [box], color)


def test_detects_and_deskews_rotated_card_on_light_background():
    img = _platen(BG)
    _draw_card(img, (1200, 1600), 12, (140, 90, 60))
    result = detect_card(img, DPI)
    assert result.detected
    dims = sorted(result.image.shape[:2])
    assert abs(dims[0] - (CARD_H + EXPAND)) <= 15
    assert abs(dims[1] - (CARD_W + EXPAND)) <= 15


def test_detects_light_card_on_dark_background():
    # 盖板打开的场景：背景近黑，证件亮
    img = _platen(12)
    _draw_card(img, (1300, 1800), -7, (245, 245, 240))
    result = detect_card(img, DPI)
    assert result.detected
    dims = sorted(result.image.shape[:2])
    assert abs(dims[0] - (CARD_H + EXPAND)) <= 15
    assert abs(dims[1] - (CARD_W + EXPAND)) <= 15


def test_margin_keeps_natural_background_around_card():
    # 余量带应保留扫描背景原色（不漂白、不删除）
    img = _platen(BG)
    _draw_card(img, (1200, 1600), 0, (140, 90, 60))
    result = detect_card(img, DPI)
    assert result.detected
    corners = np.concatenate(
        [result.image[:5, :5].ravel(), result.image[-5:, -5:].ravel()]
    )
    assert abs(int(corners.mean()) - BG) <= 4


def test_bg_color_sampled_from_scan_background():
    img = _platen(BG)
    _draw_card(img, (1200, 1600), 8, (140, 90, 60))
    result = detect_card(img, DPI)
    assert all(abs(c - BG) <= 4 for c in result.bg_color)


def test_square_card_at_45_degrees_not_distorted():
    # 回归：sum/diff 极值法排序角点在 45° 正方形上退化，导致透视变换畸变
    img = _platen(BG)
    box = cv2.boxPoints(((1300, 1700), (800, 800), 45.0)).astype(np.int32)
    cv2.fillPoly(img, [box], (140, 90, 60))
    result = detect_card(img, DPI)
    assert result.detected
    dims = sorted(result.image.shape[:2])
    assert abs(dims[0] - (800 + EXPAND)) <= 15
    assert abs(dims[1] - (800 + EXPAND)) <= 15


def test_blank_scan_returns_original_with_flag():
    img = _platen(BG)
    result = detect_card(img, DPI)
    assert not result.detected
    assert result.image.shape == img.shape
    assert all(abs(c - BG) <= 4 for c in result.bg_color)


def test_ignores_object_larger_than_a5():
    # 大于 A5 上限的整页文档不该被当成证件
    img = _platen(BG)
    box = cv2.boxPoints(((1275, 1754), (2100, 2900), 0)).astype(np.int32)
    cv2.fillPoly(img, [box], (120, 120, 120))
    result = detect_card(img, DPI)
    assert not result.detected
