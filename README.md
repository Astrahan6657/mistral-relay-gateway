# Mistral Relay Gateway

> **Unofficial Community Project / 非官方项目声明**：本项目为独立开源工具，与 Mistral AI、OpenAI 或 New API 官方无隶属关系。

**Core Proxy + Built-in Multi-Account Web Console + Optional Tauri v2 Desktop Tray Shell**  
**核心代理 + 内置多账号与额度 Web 控制台 + 可选桌面托盘壳**

A lightweight LLM API compatibility proxy designed for **Mistral API (OpenAPI 3.1.0)**, **New API / One API** deployments, and coding agents (**Codex CLI, ZCode, Cline, Cherry Studio**). Runs either as a local desktop/CLI gateway with a multi-account quota dashboard, or as a Docker Sidecar in front of New API.

```
客户端 (Codex / ZCode / Cherry Studio / OpenAI SDK)
        │  http://127.0.0.1:8790/v1   (或经 Caddy / Cloudflare Tunnel)
        ▼
┌────────────────────────────────────────────────────────────────┐
│  mistral-relay-gateway (FastAPI + 复用 httpx 连接池 + Web UI)   │
│    ├─ 多账号池：优先级轮询 · 401/402/429 自动故障转移           │
│    ├─ 额度可视化：预算 / 月限额 / 实时 TPM 进度条与倒计时       │
│    ├─ OpenAPI 3.1.0 严格白名单清洗 & /openai/v1 分路径路由      │
│    ├─ 历史截图按张数 + 体积自动裁剪 (防多图超时 503)            │
│    ├─ ThinkChunk 结构化数组 content 实时流式拍平               │
│    ├─ /v1/responses 自适应 (直连原生透传 / New API 双向互转)    │
│    └─ 内置 /ui 控制台 (浅/深主题 · 热改配置 · 实时请求监控)     │
└────────────────────────────────────────────────────────────────┘
        │
        ├─ direct_mistral 模式 ──▶ https://api.mistral.ai
        └─ new_api 模式        ──▶ http://new-api:3000
```

---

## 核心特性

1. **多账号池与额度可视化**
   - 支持单个/批量导入多个 Mistral API Key，按优先级轮询调度；遇到 `401` / `402` / `429` 时在**同一次请求内自动切换下一个健康账号**并进入冷却。
   - 自动调用 Mistral OpenAPI 3.1.0 接口（`/v1/whoami`、`/v1/models`、`/v1/billing/subscription`、`/v1/billing/budget`、`/v1/admin/spend-limit`、`/v1/admin/rate-limit`）并实时捕获响应头中的 `x-ratelimit-*`，在 `/ui` 中以进度条与倒计时卡片直观展示剩余额度。
2. **严格对齐 Mistral OpenAPI 3.1.0 & 分路径兼容 `/openai/v1`**
   - 按 `ChatCompletionRequest` / `Message` / `Tool` / `Function` 严格白名单过滤未知字段（防 `422 extra_forbidden` / `union_tag_invalid`），保留 `prompt_cache_key`、`function.strict`、合法 `service_tier` 与 `reasoning_effort`，自动映射 `developer → system`、`max_completion_tokens → max_tokens`。
   - 支持切换 `mistral_chat_endpoint=openai_compat`：仅将 `/v1/chat/completions` 路由至官方 Public Preview 的 `/openai/v1/chat/completions`，而 `/v1/responses`、`/v1/models` 与 `/v1/billing/*` 仍走原生 `/v1/...`（避免 `/openai/v1` 下这些路径返回 404）。
   - **关于默认 `MISTRAL_PREFIXES` 中的 `deepseek-v4` / `zai-glm-` / `glm-5`**：Mistral 平台及部分 New API Mistral 渠道会托管/转售这些第三方模型并沿用相同的严格 schema 与数组型思考块，因此默认纳入前缀列表；您随时可在环境变量 `MISTRAL_PREFIXES` 或 Web 控制台里自由增删。
3. **历史截图裁剪 & `ThinkChunk` 流式拍平 & Responses 互转**
   - 自动裁剪多轮 Coding Agent 堆积的历史截图（Mistral 默认留 4 张 / ≤ 3 MB，其他模型 8 张 / ≤ 10 MB）。
   - 流式与非流式自动将数组型 `content`（`[{"type":"thinking",...}]`）归一化为字符串 `content` + `reasoning_content`。
   - `direct_mistral` 模式下原生透传 `/v1/responses`；`new_api` 模式下自动完成 Responses ↔ Chat Completions 双向流式转换。

---

## 快速开始

### 方式一：本地命令行（Python 3.11+ / `uv`）

```bash
uv sync
uv run mistral-relay-gateway --host 127.0.0.1 --port 8790 --config ./config.json --open
```

浏览器会自动打开 `http://127.0.0.1:8790/ui`；API 端点为 `http://127.0.0.1:8790/v1`。

### 方式二：Docker Compose（服务端 / New API Sidecar）

```bash
cp docker-compose.example.yml docker-compose.yml
# 公网部署请务必设置 GATEWAY_KEY（若启用账号池）与强随机 ADMIN_TOKEN
docker compose up -d
```

### 方式三：桌面托盘应用（Tauri v2，自包含 Sidecar）

无需目标电脑安装 Python；构建脚本与 CI 流水线已内置（详见 [desktop/README.md](desktop/README.md)）：

```bash
# 1. 构建当前平台独立 PyInstaller Sidecar（Windows 用 .\scripts\build_sidecar.ps1）
bash scripts/build_sidecar.sh

# 2. 构建 Tauri 安装包（macOS .dmg/.app, Windows .msi/.exe, Linux .AppImage/.deb）
cd desktop && npm install && npm run build
```

推送 `v*` 标签会自动触发 [release.yml](.github/workflows/release.yml) 跨平台矩阵构建并上传到 GitHub Releases。

---

## 安全说明

详见 [SECURITY.md](SECURITY.md)：
- **防开放中转**：未设置 `GATEWAY_KEY` 时，服务器托管的账号池 / `UPSTREAM_API_KEY` **仅允许本机直连**（`127.0.0.1` / `::1` 且无 `X-Forwarded-For`）；公网或反代部署请配置 `GATEWAY_KEY`。
- **防 DNS 重绑定与本机网页攻击**：未设置 `ADMIN_TOKEN` 时，`/api/admin/*` 严格校验来源 IP、`Host` 头、`Origin` 头及自定义头 `X-MRG-Console: 1`；所有令牌均使用 `hmac.compare_digest` 常数时间比较。
- **兜底透传隔离**：兜底代理仅转发安全白名单请求头（不透传 `Cookie`、`X-Admin-Token` 等敏感头），并对上游连接异常返回标准 `502`/`504`。

## 开发与测试

```bash
uv sync --extra dev
uv run ruff check src tests
uv run pytest -v
```

## License

[MIT](LICENSE)
