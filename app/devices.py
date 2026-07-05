"""局域网 eSCL 扫描设备：mDNS 发现（`_uscan._tcp`）与可用性探测。"""

from __future__ import annotations

import asyncio
import time

import httpx
from zeroconf import ServiceListener, ServiceBrowser, Zeroconf

SERVICE = "_uscan._tcp.local."
DISCOVER_SECONDS = 3.0
PROBE_TIMEOUT = 3.0


def _discover_blocking(timeout: float) -> list[dict]:
    found: dict[str, dict] = {}

    class Listener(ServiceListener):
        def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            info = zc.get_service_info(type_, name, timeout=2000)
            if info is None:
                return
            addresses = info.parsed_addresses()
            addr = next((a for a in addresses if ":" not in a), None)  # 优先 IPv4
            if addr is None:
                return
            txt: dict[str, str] = {}
            for key, value in (info.properties or {}).items():
                try:
                    txt[key.decode()] = value.decode() if value else ""
                except (UnicodeDecodeError, AttributeError):
                    continue
            # 注意：eSCL 资源根路径取自 TXT 的 rs 字段，本项目按主流设备假定为
            # /eSCL（HP 等均如此）；rs 为其他值的设备暂不支持
            display = name[: -len("." + SERVICE)] if name.endswith("." + SERVICE) else name
            found[name] = {
                "name": display,
                "url": f"http://{addr}:{info.port}",
                "model": txt.get("ty", ""),
                "location": txt.get("note", ""),
            }

        def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            pass

        def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            pass

    zc = Zeroconf()
    browser = ServiceBrowser(zc, SERVICE, Listener())
    try:
        time.sleep(timeout)
    finally:
        # cancel() 会同步 join 回调线程：zc.close() 不管 ServiceBrowser 线程，
        # 少了这步每次发现都泄漏一个线程，且 found 存在并发读写窗口
        browser.cancel()
        zc.close()
    return sorted(found.values(), key=lambda d: d["name"])


async def discover(timeout: float = DISCOVER_SECONDS) -> list[dict]:
    return await asyncio.to_thread(_discover_blocking, timeout)


async def is_available(url: str) -> bool:
    """快速探测设备是否在线（eSCL ScannerStatus 可达即视为可用）。"""
    try:
        async with httpx.AsyncClient(base_url=url, timeout=PROBE_TIMEOUT) as client:
            resp = await client.get("/eSCL/ScannerStatus")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False
