# Changelog

## [0.1.0] - 2026-11-19

### Added
- Core FastAPI proxy with shared pooled `httpx.AsyncClient` and transient `502`/`520`/`524` retry.
- Mistral OpenAPI 3.1.0 strict request whitelist (`ChatCompletionRequest`, messages, tools, `prompt_cache_key`, `function.strict`, `service_tier`, `reasoning_effort`).
- Split-path routing for Mistral's Public Preview `/openai/v1/chat/completions` endpoint while preserving `/v1/responses` and `/v1/models`.
- Full `/v1/responses` <-> `/v1/chat/completions` bidirectional translation (sync & SSE streaming) for New API sidecar mode.
- Multi-turn history image compaction (count + max body bytes) and streaming `ThinkChunk` array normalization.
- Multi-account pool with priority/round-robin scheduling, `401`/`402`/`429` automatic failover, live rate-limit header ingestion, and Mistral billing/budget quota visualization.
- Built-in Web Console (`/ui`) with Light/Dark theme and hot-reloadable configuration.
- Optional Tauri v2 desktop tray shell (`desktop/`), standalone PyInstaller Sidecar build scripts (`scripts/build_sidecar.sh`, `scripts/build_sidecar.ps1`), and GitHub Actions CI & Release workflows.

### Security
- Prevented open-relay access to server-managed account pools from non-local hosts without `GATEWAY_KEY`.
- Hardened `/api/admin/*` against DNS rebinding and cross-origin webpage attacks (`Host`, `Origin`, `X-Forwarded-For`, and `X-MRG-Console: 1` checks).
- Switched token validation to constant-time `hmac.compare_digest`.
- Restricted fallback proxy to a safe header allowlist and added graceful `502`/`504` error handling.
