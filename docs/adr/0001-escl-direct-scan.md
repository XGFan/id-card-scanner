# 通过 eSCL 协议直接驱动扫描仪，不依赖系统驱动

HP DeskJet 5135 在局域网广播 `_uscan._tcp`（eSCL/AirScan，端点 `http://scanner.local:8080/eSCL`），意味着扫描可以用纯 HTTP 请求完成（POST ScanSettings XML 创建任务 → GET 取回 JPEG）。我们选择直接调 eSCL，而不是走 macOS ImageCaptureCore、SANE 或 HP 官方驱动：这让服务端零驱动依赖、不绑定操作系统，才撑得起「本地 Web 应用、任何设备浏览器都能操作」的产品形态。代价是要自己处理 eSCL 的 XML 协议细节，并且换成不支持 eSCL 的扫描仪时扫描模块需要重写。

## Considered Options

- **macOS ImageCaptureCore**：体验原生，但把服务端绑死在 macOS，且无法从其他设备使用。
- **SANE（sane-airscan）**：多一层本可不要的依赖，底层同样走 eSCL。
- **eSCL 直连（选定）**：设备能力已实测确认（eSCL 2.63、平板、彩色、75–1200 DPI、输出 JPEG/PDF）。
