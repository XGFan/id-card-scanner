"""FastAPI 服务：设备选择 + 扫描接口 + 复印件页 PDF + 前端静态页。

会话状态是内存单会话（front/back 两面的证件图），不落盘留档；
扫描用 asyncio.Lock 串行化——打印机同时只能执行一个扫描任务。
当前设备持久化在 ~/.config/card-scan/settings.json，启动时默认沿用上次设备。
"""

from __future__ import annotations

import asyncio
import io
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel

from . import config, devices, escl, imaging, pdfgen, settings

app = FastAPI(title="证件扫描")

SIDES = ("front", "back")

_scan_lock = asyncio.Lock()
# side -> {"jpeg": bytes, "detected": bool, "bg": (r, g, b), "ts": float}
_sides: dict[str, dict] = {}

_STATIC = Path(__file__).parent / "static"


def _initial_device() -> dict | None:
    saved = settings.load().get("device")
    if saved and saved.get("url"):
        return saved
    if "PRINTER_URL" in os.environ:
        return {"name": "PRINTER_URL 指定设备", "url": config.PRINTER_URL}
    return None


_device: dict | None = _initial_device()


class DeviceIn(BaseModel):
    url: str
    name: str = ""
    model: str = ""
    location: str = ""


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/healthz")
async def healthz() -> dict:
    # K8s 探针用：只反映进程存活，不探测打印机（设备离线不该重启 Pod）
    return {"ok": True}


@app.get("/api/device")
async def get_device() -> dict:
    if _device is None:
        return {"device": None, "available": False}
    return {"device": _device, "available": await devices.is_available(_device["url"])}


@app.get("/api/devices/discover")
async def discover_devices() -> dict:
    return {"devices": await devices.discover()}


@app.post("/api/device")
async def set_device(body: DeviceIn) -> dict:
    global _device
    if not body.url.startswith(("http://", "https://")):
        raise HTTPException(422, "设备地址必须是 http(s) URL")
    _device = body.model_dump()
    settings.save({**settings.load(), "device": _device})
    return {"device": _device, "available": await devices.is_available(_device["url"])}


@app.post("/api/scan/{side}")
async def scan(side: str) -> dict:
    if side not in SIDES:
        raise HTTPException(404, "side 必须是 front 或 back")
    if _device is None:
        raise HTTPException(409, "尚未选择扫描仪，请先选择设备")
    if _scan_lock.locked():
        raise HTTPException(409, "正在扫描中，请等待当前扫描完成")
    async with _scan_lock:
        try:
            raw = await escl.scan_jpeg(_device["url"], config.SCAN_DPI)
        except escl.ScanError as exc:
            raise HTTPException(502, str(exc)) from exc
        jpeg, detected, bg, preview = await asyncio.to_thread(_process, raw)
        _sides[side] = {
            "jpeg": jpeg,
            "preview": preview,
            "detected": detected,
            "bg": bg,
            "ts": time.time(),
        }
    return _state()


PREVIEW_W = 900  # 框选预览图宽度（原图 2550px 降采样，省传输）
_BOX_COLOR = (89, 199, 52)  # BGR，同前端主题绿 #34c759


def _process(raw_jpeg: bytes) -> tuple[bytes, bool, tuple[int, int, int], bytes]:
    array = np.frombuffer(raw_jpeg, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(502, "扫描图像解码失败")
    result = imaging.detect_card(bgr, config.SCAN_DPI)
    ok, encoded = cv2.imencode(".jpg", result.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HTTPException(500, "图像编码失败")
    b, g, r = result.bg_color  # OpenCV 是 BGR，Pillow 用 RGB
    preview = _make_preview(bgr, result.quad)
    return encoded.tobytes(), result.detected, (r, g, b), preview


def _make_preview(bgr: np.ndarray, quad: np.ndarray | None) -> bytes:
    """原始扫描图的框选预览：降采样 + 绿色边界框。

    不压暗框外区域：预览必须呈现真实的扫描背景色，否则用户会拿被压暗的
    预览当参照，误以为复印件页的底色「变白了」（2026-07-05 实际发生过）。
    """
    scale = PREVIEW_W / bgr.shape[1]
    small = cv2.resize(
        bgr, (PREVIEW_W, round(bgr.shape[0] * scale)), interpolation=cv2.INTER_AREA
    )
    if quad is not None:
        pts = (quad * scale).astype(np.int32)
        cv2.polylines(small, [pts], True, _BOX_COLOR, 4, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(500, "预览编码失败")
    return buf.tobytes()


@app.get("/api/image/{side}")
async def image(side: str) -> Response:
    entry = _sides.get(side)
    if entry is None:
        raise HTTPException(404, "这一面还没有扫描")
    return Response(
        entry["jpeg"],
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/preview/{side}")
async def preview(side: str) -> Response:
    """原始扫描图 + 自动框选效果，页面上的主预览。"""
    entry = _sides.get(side)
    if entry is None:
        raise HTTPException(404, "这一面还没有扫描")
    return Response(
        entry["preview"],
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/composite")
async def composite() -> Response:
    """复印件页的图片预览（与 PDF 同源画布），供前端 tab 内统一缩放查看。"""
    entries = {side: _sides.get(side) for side in SIDES}
    if not all(entries.values()):
        raise HTTPException(409, "正反面都扫描完成后才能生成复印件预览")

    def build() -> bytes:
        front = Image.open(io.BytesIO(entries["front"]["jpeg"])).convert("RGB")
        back = Image.open(io.BytesIO(entries["back"]["jpeg"])).convert("RGB")
        canvas = pdfgen.compose_canvas(
            front, back, config.SCAN_DPI, bg_color=entries["front"]["bg"]
        )
        canvas.thumbnail((1400, 1980), Image.LANCZOS)  # 降采样省传输
        buf = io.BytesIO()
        canvas.save(buf, "JPEG", quality=85)
        return buf.getvalue()

    data = await asyncio.to_thread(build)
    return Response(
        data,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/pdf")
async def pdf() -> Response:
    # 先取快照，避免生成期间并发 reset 清空 _sides 导致 KeyError
    entries = {side: _sides.get(side) for side in SIDES}
    if not all(entries.values()):
        raise HTTPException(409, "正反面都扫描完成后才能生成 PDF")

    def build() -> bytes:
        front = Image.open(io.BytesIO(entries["front"]["jpeg"])).convert("RGB")
        back = Image.open(io.BytesIO(entries["back"]["jpeg"])).convert("RGB")
        return pdfgen.compose_pdf(
            front, back, config.SCAN_DPI, bg_color=entries["front"]["bg"]
        )

    data = await asyncio.to_thread(build)
    filename = f"证件-{datetime.now():%Y%m%d-%H%M}.pdf"
    return Response(
        data,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{quote(filename)}",
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/reset")
async def reset() -> dict:
    # 与扫描共用一把锁：扫描在飞行中时 reset 等它落地再清，避免清完又被写回半截状态
    async with _scan_lock:
        _sides.clear()
    return _state()


@app.get("/api/state")
async def state() -> dict:
    return _state()


def _state() -> dict:
    return {
        side: (
            {"detected": _sides[side]["detected"], "ts": _sides[side]["ts"]}
            if side in _sides
            else None
        )
        for side in SIDES
    }
