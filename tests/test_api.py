import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main


def _fake_platen_jpeg() -> bytes:
    img = np.full((3508, 2550, 3), 242, dtype=np.uint8)
    box = cv2.boxPoints(((1200, 1500), (1012, 638), 8)).astype(np.int32)
    cv2.fillPoly(img, [box], (150, 90, 50))
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def client(monkeypatch):
    main._sides.clear()
    main._device = {"name": "测试扫描仪", "url": "http://fake-printer:8080"}

    async def fake_scan(base_url: str, dpi: int) -> bytes:
        return _fake_platen_jpeg()

    async def fake_available(url: str) -> bool:
        return True

    monkeypatch.setattr(main.escl, "scan_jpeg", fake_scan)
    monkeypatch.setattr(main.devices, "is_available", fake_available)
    monkeypatch.setattr(main.settings, "save", lambda data: None)
    return TestClient(main.app)


def test_full_flow(client):
    resp = client.post("/api/scan/front")
    assert resp.status_code == 200
    assert resp.json()["front"]["detected"] is True

    # 只有一面时不能出 PDF
    assert client.get("/api/pdf").status_code == 409

    assert client.post("/api/scan/back").status_code == 200

    pdf = client.get("/api/pdf")
    assert pdf.status_code == 200
    assert pdf.content.startswith(b"%PDF")
    assert pdf.headers["content-type"] == "application/pdf"

    img = client.get("/api/image/front")
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/jpeg"

    assert client.post("/api/reset").json() == {"front": None, "back": None}


def test_healthz(client):
    assert client.get("/healthz").json() == {"ok": True}


def test_invalid_side_rejected(client):
    assert client.post("/api/scan/top").status_code == 404


def test_image_404_before_scan(client):
    assert client.get("/api/image/front").status_code == 404


def test_scan_error_surfaces_as_502(client, monkeypatch):
    from app import escl

    async def failing_scan(base_url: str, dpi: int) -> bytes:
        raise escl.ScanError("扫描仪当前状态为 Processing，请稍后再试")

    monkeypatch.setattr(main.escl, "scan_jpeg", failing_scan)
    resp = client.post("/api/scan/front")
    assert resp.status_code == 502
    assert "Processing" in resp.json()["detail"]


def test_scan_requires_device(client):
    main._device = None
    resp = client.post("/api/scan/front")
    assert resp.status_code == 409
    assert "选择" in resp.json()["detail"]


def test_get_device_none(client):
    main._device = None
    assert client.get("/api/device").json() == {"device": None, "available": False}


def test_set_device_and_get(client):
    resp = client.post(
        "/api/device",
        json={"name": "HP DeskJet 5135", "url": "http://scanner.local:8080",
              "model": "HP DeskJet 5100 series", "location": "书房"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["device"]["name"] == "HP DeskJet 5135"

    info = client.get("/api/device").json()
    assert info["device"]["url"] == "http://scanner.local:8080"


def test_set_device_rejects_non_http(client):
    resp = client.post("/api/device", json={"name": "x", "url": "ftp://nope"})
    assert resp.status_code == 422


def test_discover_devices(client, monkeypatch):
    async def fake_discover(timeout: float = 3.0):
        return [{"name": "HP DeskJet 5135", "url": "http://10.0.0.5:8080",
                 "model": "HP DeskJet 5100 series", "location": "书房"}]

    monkeypatch.setattr(main.devices, "discover", fake_discover)
    devices = client.get("/api/devices/discover").json()["devices"]
    assert len(devices) == 1
    assert devices[0]["location"] == "书房"
