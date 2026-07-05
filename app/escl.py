"""eSCL (AirScan) 客户端——纯 HTTP 驱动扫描仪，无驱动依赖。

协议时序（见 docs/adr/0001-escl-direct-scan.md）：
GET ScannerStatus 确认 Idle → POST ScanJobs 创建任务（201 + Location）
→ GET {job}/NextDocument 取回 JPEG（未就绪时 503 重试）→ 404 表示无更多页
→ DELETE {job} 释放任务。
"""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit

import httpx

SCAN_NS = "http://schemas.hp.com/imaging/escl/2011/05/03"
PWG_NS = "http://www.pwg.org/schemas/2010/12/sm"

# 玻璃板全幅，单位 1/300 英寸（来自设备 ScannerCapabilities 的 MaxWidth/MaxHeight）
PLATEN_WIDTH = 2550
PLATEN_HEIGHT = 3508

_POLL_INTERVAL = 2.0
_MAX_POLLS = 90  # 最长约 3 分钟

_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)


class ScanError(Exception):
    """扫描失败，message 面向用户展示。"""


def build_scan_settings(dpi: int, width: int = PLATEN_WIDTH, height: int = PLATEN_HEIGHT) -> str:
    # 元素顺序遵循 eSCL XSD 的 sequence 定义，部分设备对顺序敏感
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<scan:ScanSettings xmlns:scan="{SCAN_NS}" xmlns:pwg="{PWG_NS}">
  <pwg:Version>2.63</pwg:Version>
  <scan:Intent>Document</scan:Intent>
  <pwg:ScanRegions pwg:MustHonor="true">
    <pwg:ScanRegion>
      <pwg:ContentRegionUnits>escl:ThreeHundredthsOfInches</pwg:ContentRegionUnits>
      <pwg:Height>{height}</pwg:Height>
      <pwg:Width>{width}</pwg:Width>
      <pwg:XOffset>0</pwg:XOffset>
      <pwg:YOffset>0</pwg:YOffset>
    </pwg:ScanRegion>
  </pwg:ScanRegions>
  <pwg:InputSource>Platen</pwg:InputSource>
  <scan:ColorMode>RGB24</scan:ColorMode>
  <scan:XResolution>{dpi}</scan:XResolution>
  <scan:YResolution>{dpi}</scan:YResolution>
  <pwg:DocumentFormat>image/jpeg</pwg:DocumentFormat>
</scan:ScanSettings>"""


def parse_scanner_state(status_xml: str) -> str:
    root = ET.fromstring(status_xml)
    state = root.find(f"{{{PWG_NS}}}State")
    if state is None or not state.text:
        raise ScanError("无法解析扫描仪状态")
    return state.text.strip()


def job_path(location: str) -> str:
    """Location 头可能是绝对 URL 或路径，统一取路径部分。"""
    return urlsplit(location).path


async def get_scanner_state(base_url: str) -> str:
    async with httpx.AsyncClient(base_url=base_url, timeout=_TIMEOUT) as client:
        resp = await client.get("/eSCL/ScannerStatus")
        resp.raise_for_status()
        return parse_scanner_state(resp.text)


async def scan_jpeg(base_url: str, dpi: int) -> bytes:
    """执行一次平板扫描，返回整幅玻璃板的 JPEG（原始扫描图）。"""
    async with httpx.AsyncClient(base_url=base_url, timeout=_TIMEOUT) as client:
        try:
            return await _scan(client, dpi)
        except httpx.HTTPError as exc:
            # 打印机休眠/断网/超时等网络层错误统一转成面向用户的 ScanError
            raise ScanError(f"与打印机通信失败（{base_url}）：{exc}") from exc


async def _scan(client: httpx.AsyncClient, dpi: int) -> bytes:
    status = await client.get("/eSCL/ScannerStatus")
    status.raise_for_status()
    state = parse_scanner_state(status.text)
    if state != "Idle":
        raise ScanError(f"扫描仪当前状态为 {state}，请稍后再试")

    resp = await client.post(
        "/eSCL/ScanJobs",
        content=build_scan_settings(dpi),
        headers={"Content-Type": "text/xml"},
    )
    if resp.status_code != 201:
        raise ScanError(f"创建扫描任务被拒绝（HTTP {resp.status_code}）：{resp.text[:200]}")
    location = resp.headers.get("Location")
    if not location:
        raise ScanError("设备未返回扫描任务地址（Location 头缺失）")
    job = job_path(location)

    try:
        return await _fetch_document(client, job)
    finally:
        # 释放任务；失败不影响已取回的图像
        try:
            await client.delete(job)
        except httpx.HTTPError:
            pass


async def _fetch_document(client: httpx.AsyncClient, job: str) -> bytes:
    for _ in range(_MAX_POLLS):
        resp = await client.get(f"{job}/NextDocument")
        if resp.status_code == 200:
            if not resp.content:
                raise ScanError("设备返回了空图像")
            return resp.content
        if resp.status_code == 503:
            await asyncio.sleep(_POLL_INTERVAL)
            continue
        if resp.status_code == 404:
            raise ScanError("扫描任务未产出图像（可能被设备取消）")
        raise ScanError(f"取回扫描图像失败（HTTP {resp.status_code}）")
    raise ScanError("等待扫描结果超时")
