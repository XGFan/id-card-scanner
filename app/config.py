import os

# 显式覆盖用；正常流程走设备发现 + settings 持久化（见 devices.py / settings.py）
PRINTER_URL = os.environ.get("PRINTER_URL", "")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "5135"))
SCAN_DPI = 300
