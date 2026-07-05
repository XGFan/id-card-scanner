from pathlib import Path

from app import settings


def test_roundtrip_and_missing_file(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "settings.json"
    monkeypatch.setattr(settings, "_PATH", path)

    assert settings.load() == {}  # 文件不存在 → 空配置

    settings.save({"device": {"name": "HP", "url": "http://x:8080"}})
    assert settings.load()["device"]["url"] == "http://x:8080"


def test_corrupt_file_returns_empty(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text("{not json")
    monkeypatch.setattr(settings, "_PATH", path)
    assert settings.load() == {}


def test_valid_json_but_not_object_returns_empty(tmp_path, monkeypatch):
    # null/[] 是合法 JSON 但非对象，load 必须归一为 {}，否则服务导入期崩溃
    path = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "_PATH", path)
    for content in ("null", "[]", "123", '"x"'):
        path.write_text(content)
        assert settings.load() == {}
