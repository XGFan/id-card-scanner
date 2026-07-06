# id-card-scanner

把证件放在平板扫描仪（eSCL 协议，如 HP DeskJet 5135）上，「扫正面 → 翻面 →
扫反面」，自动找边、裁剪（含 3mm 自然背景余量）、摆正，合成一页 A4 纵向 PDF
（正面上半页、反面下半页，按 1:1 物理真实尺寸，页面底色取自扫描背景，
观感如真实复印件），浏览器内预览并下载。

术语表见 [CONTEXT.md](./CONTEXT.md)，关键架构决策见 [docs/adr/](./docs/adr/)。

## 运行

需要 [uv](https://docs.astral.sh/uv/)，打印机与本机在同一局域网。

```sh
uv run python -m app
```

然后浏览器打开 <http://127.0.0.1:5135>。

## 配置（环境变量）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PRINTER_URL` | （空） | 显式指定 eSCL 设备地址，跳过设备选择（一般不需要） |
| `HOST` | `127.0.0.1` | 服务监听地址（想给局域网其他设备用改成 `0.0.0.0`） |
| `PORT` | `5135` | 服务端口 |

设备选择：首次使用时页面会自动搜索局域网内的 eSCL 扫描仪（mDNS `_uscan._tcp`），
选定后记住（存于项目内 `data/settings.json`，路径可用 `CARD_SCAN_SETTINGS` 覆盖；
配置随项目走，方便局域网服务器部署），下次启动默认沿用；
上次设备不可用时会提示重新选择。

> ⚠️ 注意：服务无认证。改成 `HOST=0.0.0.0` 后，局域网内任何人都能无认证访问
> 扫描图与 PDF（证件属敏感信息，请知情决定），也能通过 `POST /api/device`
> 让服务端向任意 URL 发起探测请求（盲式内网探测面）。另外服务端只有一份会话，
> 多台设备同时扫描会互相覆盖对方的正反面。

## 使用流程

1. 掀开盖板，把证件放在玻璃板上（位置角度随意，会自动摆正），合上盖板
2. 点「扫描正面」，等约 20 秒
3. 翻面，点「扫描反面」
4. 页面内预览 PDF，点「下载 PDF」（文件名 `证件-YYYYMMDD-HHMM.pdf`）
5. 某一面裁剪不理想（如白色证件贴白盖板识别失败）：重摆后点「重扫正面/反面」即可，
   不用从头再来。识别失败时该面会以整幅原图进入 PDF 并带 ⚠ 提示。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/device` | 当前设备及其可用性 |
| `GET` | `/api/devices/discover` | 搜索局域网内的 eSCL 扫描仪（约 3 秒） |
| `POST` | `/api/device` | 选定设备（`{url, name, model?, location?}`）并持久化 |
| `POST` | `/api/scan/{front\|back}` | 触发一次平板扫描并处理该面 |
| `GET` | `/api/image/{front\|back}` | 该面处理后的证件图（JPEG） |
| `GET` | `/api/preview/{front\|back}` | 该面原始扫描图 + 自动框选效果（页面主预览） |
| `GET` | `/api/pdf` | 复印件页 PDF（两面齐备后可用，否则 409） |
| `GET` | `/api/state` | 当前两面的状态 |
| `POST` | `/api/reset` | 清空重来 |

## 开发

```sh
uv run pytest        # 单元测试（不需要打印机）
```

## 部署

线上跑在私有 k3s 集群，集群入口统一认证，内外网访问都需要登录。对外域名不在
本仓库公开。

- **CI**：push master 触发 Woodpecker：pytest → buildx 构建推
  `docker.test4x.com/xgfan/id-card-scanner`（`sha8` + `latest`）→
  `kubectl set image` 滚动更新。见 `.woodpecker.yaml`。
- **K8s 清单**（source-of-truth）在私有 infra 仓库：Deployment + Service +
  Ingress（挂认证中间件、内外网都要登录）+ Longhorn PVC（挂 `/app/data` 持久化设备选择）。
- **集群内没有 mDNS**：多播不跨 CNI 边界，「搜索设备」在线上会搜不到——初始设备
  由 Deployment 的 `PRINTER_URL`（打印机固定 IP）提供，页面上也可手动输入设备地址。
- 时区 `TZ=Asia/Shanghai` 在镜像里设好（文件名时间戳用）。

扫描协议是标准 eSCL（AirScan）纯 HTTP，无驱动依赖；实现见 `app/escl.py`。
找边裁剪见 `app/imaging.py`（Canny 主路 + Otsu 兜底），PDF 合成见 `app/pdfgen.py`。
