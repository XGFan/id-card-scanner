"""持久化设置：记住上一次使用的扫描设备。

默认存储在项目内 data/settings.json——本服务面向局域网部署，配置随项目走，
不落在运行用户的 home 目录。可用环境变量 CARD_SCAN_SETTINGS 覆盖路径。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_PATH = Path(
    os.environ.get(
        "CARD_SCAN_SETTINGS",
        Path(__file__).resolve().parent.parent / "data" / "settings.json",
    )
)


def load() -> dict:
    try:
        data = json.loads(_PATH.read_text())
    except (OSError, ValueError):
        return {}
    # 文件内容是合法 JSON 但非对象（null/[]/数字）时同样视为损坏，
    # 否则模块导入期的 load().get(...) 会让服务直接起不来
    return data if isinstance(data, dict) else {}


def save(data: dict) -> None:
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    _PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))
