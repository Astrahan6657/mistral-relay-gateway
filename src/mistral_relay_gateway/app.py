from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

from . import __version__
from .accounts import AccountEntry, AccountPool
from .config import GatewayConfig
from .limiter import ModelLimiter
from .telemetry import TelemetryStore
from .transforms import (
    chat_resp_to_responses,
    encode_and_cap_chat,
    encode_and_cap_responses,
    normalize_chat_object,
    normalize_chat_sse_stream,
    responses_req_to_chat,
    sanitize_mistral_chat,
    sanitize_mistral_responses,
    stream_chat_to_responses,
    strip_global_fields,
)


def _resolve_static_dir() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass) / "mistral_relay_gateway" / "static"
        if candidate.is_dir():
            return candidate
    return Path(__file__).parent / "static"


STATIC_DIR = _resolve_static_dir()
LOCAL_CLIENTS = {"127.0.0.1", "::1", "localhost", "testclient"}
LOCAL_HOSTNAMES = {"127.0.0.1", "localhost", "::1", "[::1]"}
SAFE_FALLBACK_HEADERS = {
    "content-type",
    "accept",
    "authorization",
    "x-api-key",
    "anthropic-version",
    "anthropic-beta",
    "openai-beta",
    "user-agent",
}


def _const_eq(a: str, b: str) -> bool:
    return bool(a) and bool(b) and hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def _extract_hostname(host_header: str) -> str:
    raw = (host_header or "").strip().lower()
    if not raw:
        return ""
    if raw.startswith("["):
        end = raw.find("]")
        return raw[: end + 1] if end != -1 else raw
    return raw.split(":", 1)[0]

log = logging.getLogger("mistral-relay-gateway")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )


def _fmt_exc(exc: Exception) -> str:
    msg = str(exc).strip()
    name = type(exc).__name__
    return f"{name}: {msg}" if msg else name


def _build_http_client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=15.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=30, keepalive_expiry=30.0),
    )


def _derive_accounts_path(config: GatewayConfig, accounts_path: str | None = None) -> str:
    if accounts_path is not None:
        return accounts_path
    env_p = os.environ.get("ACCOUNTS_PATH", "").strip()
    if env_p:
        return env_p
    if config.config_path:
        return str(Path(config.config_path).expanduser().with_name("accounts.json"))
    return ""


class GatewayRuntime:
    def __init__(
        self,
        config: GatewayConfig,
        *,
        injected_client: httpx.AsyncClient | None = None,
        accounts_path: str | None = None,
        bind_host: str = "127.0.0.1",
    ):
        self.config = config
        self.bind_host = bind_host
        self.limiter = ModelLimiter(config.model_concurrency_limits, config.concurrency_wait_seconds)
        self.telemetry = TelemetryStore(max_events=250)
        self.accounts = AccountPool(store_path=_derive_accounts_path(config, accounts_path))
        self._injected_client = injected_client
        self._client: httpx.AsyncClient | None = injected_client

    async def start(self) -> None:
        if self._client is None:
            self._client = _build_http_client(self.config.upstream_timeout)
        if self.bind_host not in LOCAL_HOSTNAMES and not self.config.gateway_key and (
            self.accounts.has_enabled() or bool(self.config.upstream_api_key)
        ):
            log.warning(
                "Bound to non-local host '%s' without GATEWAY_KEY: remote unauthenticated requests "
                "to server-managed credentials will be rejected (401).",
                self.bind_host,
            )

    async def close(self) -> None:
        if self._client is not None and self._client is not self._injected_client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = _build_http_client(self.config.upstream_timeout)
        return self._client

    async def apply_config_patch(self, patch: dict[str, Any]) -> bool:
        old_timeout = self.config.upstream_timeout
        self.config.apply_patch(patch)
        self.limiter.reconfigure(
            self.config.model_concurrency_limits,
            self.config.concurrency_wait_seconds,
        )
        if self._injected_client is None and self.config.upstream_timeout != old_timeout:
            old = self._client
            self._client = _build_http_client(self.config.upstream_timeout)
            if old is not None:
                await old.aclose()
        return self.config.save()

    def is_local_client(self, request: Request) -> bool:
        client_host = request.client.host if request.client else ""
        if client_host not in LOCAL_CLIENTS:
            return False
        if request.headers.get("x-forwarded-for"):
            return False
        host_name = _extract_hostname(request.headers.get("host", ""))
        return host_name in LOCAL_HOSTNAMES

    def _matches_gateway_key(self, request: Request) -> bool:
        gk = self.config.gateway_key
        if not gk:
            return False
        hdr = request.headers.get("authorization", "")
        if hdr.startswith("Bearer ") and _const_eq(hdr[7:].strip(), gk):
            return True
        return _const_eq(request.headers.get("x-api-key", "").strip(), gk)

    def _client_supplied_auth(self, request: Request) -> bool:
        return bool(request.headers.get("authorization", "") or request.headers.get("x-api-key", ""))

    def should_use_account_pool(self, request: Request) -> bool:
        if not self.accounts.has_enabled():
            return False
        if not self.config.passthrough_auth:
            return True
        if self._matches_gateway_key(request):
            return True
        return not self._client_supplied_auth(request)

    def auth_ok(self, request: Request) -> bool:
        cfg = self.config
        has_server_creds = bool(cfg.upstream_api_key or self.accounts.has_enabled())
        if cfg.passthrough_auth:
            # Explicit client auth header: allow (either upstream passthrough or matching gateway_key)
            if self._client_supplied_auth(request):
                return True
            # No client auth header -> only allow server-managed credentials from genuine localhost when gateway_key is unset
            if not has_server_creds:
                return False
            if cfg.gateway_key:
                return False
            return self.is_local_client(request)
        # passthrough_auth=False -> requests use server-managed credentials
        if cfg.gateway_key:
            return self._matches_gateway_key(request)
        return self.is_local_client(request)

    def admin_ok(self, request: Request) -> bool:
        token = self.config.admin_token
        if token:
            hdr = request.headers.get("x-admin-token", "").strip()
            if _const_eq(hdr, token):
                return True
            auth = request.headers.get("authorization", "")
            if auth.startswith("Bearer ") and _const_eq(auth[7:].strip(), token):
                return True
            return False
        # Without ADMIN_TOKEN: enforce localhost + Host header + Origin check + custom anti-CSRF header
        if not self.is_local_client(request):
            return False
        origin = request.headers.get("origin", "").strip()
        if origin:
            origin_host = (urlparse(origin).hostname or "").lower()
            if origin_host not in {"127.0.0.1", "localhost", "::1"}:
                return False
        return request.headers.get("x-mrg-console", "").strip() == "1"

    def upstream_headers(
        self,
        request: Request,
        *,
        account: AccountEntry | None = None,
    ) -> dict[str, str]:
        cfg = self.config
        headers = {"Content-Type": "application/json", "Accept-Encoding": "identity"}
        for k in ("anthropic-version", "anthropic-beta", "openai-beta"):
            v = request.headers.get(k)
            if v:
                headers[k] = v

        if account is not None:
            headers["Authorization"] = f"Bearer {account.api_key}"
            return headers

        if cfg.passthrough_auth:
            for k in ("authorization", "x-api-key"):
                v = request.headers.get(k)
                if v:
                    headers[k] = v
            if "authorization" not in headers and "x-api-key" not in headers and cfg.upstream_api_key:
                headers["Authorization"] = f"Bearer {cfg.upstream_api_key}"
        elif cfg.upstream_api_key:
            headers["Authorization"] = f"Bearer {cfg.upstream_api_key}"
        return headers

    async def send_with_retry(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        *,
        stream: bool,
    ) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                if stream:
                    req = self.client.build_request(method, url, content=body, headers=headers)
                    resp = await self.client.send(req, stream=True)
                    if resp.status_code in (502, 520, 524) and attempt == 0:
                        await resp.aclose()
                        await asyncio.sleep(0.5)
                        continue
                    return resp
                else:
                    resp = await self.client.request(method, url, content=body, headers=headers)
                    if resp.status_code in (502, 520, 524) and attempt == 0:
                        await asyncio.sleep(0.5)
                        continue
                    return resp
            except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as exc:
                last_exc = exc
                if attempt == 0:
                    log.warning("transient upstream error (%s), retrying once", _fmt_exc(exc))
                    await asyncio.sleep(0.5)
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    async def dispatch_upstream(
        self,
        request: Request,
        *,
        path_suffix: str,
        body: bytes,
        model: str | None,
        stream: bool,
    ) -> tuple[httpx.Response, AccountEntry | None]:
        """
        Send request to upstream, using the multi-account pool with automatic failover
        on 401/402/429 when pool routing is active.
        """
        if not self.should_use_account_pool(request):
            url = f"{self.config.upstream_base}{path_suffix}"
            headers = self.upstream_headers(request)
            resp = await self.send_with_retry("POST", url, body, headers, stream=stream)
            return resp, None

        exclude: set[str] = set()
        last_resp: httpx.Response | None = None
        last_acc: AccountEntry | None = None

        for _ in range(3):
            acc = self.accounts.select_account(model=model, exclude_ids=exclude)
            if acc is None:
                break
            base = (acc.upstream_base or self.config.upstream_base).rstrip("/")
            if base.endswith("/openai/v1"):
                base = base[: -len("/openai/v1")]
            elif base.endswith("/v1"):
                base = base[: -len("/v1")]
            url = f"{base}{path_suffix}"
            headers = self.upstream_headers(request, account=acc)
            resp = await self.send_with_retry("POST", url, body, headers, stream=stream)
            self.accounts.record_outcome(acc.id, status=resp.status_code, headers=resp.headers)
            last_resp, last_acc = resp, acc

            if resp.status_code in (401, 402, 429):
                exclude.add(acc.id)
                if self.accounts.select_account(model=model, exclude_ids=exclude) is not None:
                    log.warning(
                        "account %s returned %d, failing over to next account",
                        acc.name or acc.id,
                        resp.status_code,
                    )
                    if stream:
                        await resp.aclose()
                    continue
            return resp, acc

        if last_resp is not None:
            return last_resp, last_acc

        # All pool accounts cooling down; fall back to single upstream_api_key if configured
        url = f"{self.config.upstream_base}{path_suffix}"
        headers = self.upstream_headers(request)
        resp = await self.send_with_retry("POST", url, body, headers, stream=stream)
        return resp, None


def _concurrency_exceeded(model: str | None, cap: int) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": (
                    f"concurrency limit exceeded for model '{model}': "
                    f"max {cap} in-flight requests allowed, please retry later"
                ),
                "type": "concurrency_limit_exceeded",
                "code": "concurrency_limit_exceeded",
                "model": model,
                "limit": cap,
            }
        },
        status_code=429,
        headers={"Retry-After": "5"},
    )


def create_app(
    config: GatewayConfig | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
    accounts_path: str | None = None,
    bind_host: str = "127.0.0.1",
) -> FastAPI:
    cfg = config or GatewayConfig.load()
    rt = GatewayRuntime(
        cfg,
        injected_client=http_client,
        accounts_path=accounts_path,
        bind_host=bind_host,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await rt.start()
        try:
            yield
        finally:
            await rt.close()

    app = FastAPI(title="Mistral Relay Gateway", version=__version__, lifespan=lifespan)
    app.state.runtime = rt

    # ---------------------------------------------------------------- Web Console & Health

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/ui", status_code=302)

    @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    async def web_console():
        html_path = STATIC_DIR / "index.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/healthz")
    @app.get("/gw/healthz")
    async def healthz():
        c = rt.config
        return JSONResponse(
            {
                "ok": True,
                "version": __version__,
                "upstream_mode": c.upstream_mode,
                "upstream": c.upstream_base,
                "mistral_chat_endpoint": c.mistral_chat_endpoint,
                "passthrough_auth": c.passthrough_auth,
                "strict_mistral_whitelist": c.strict_mistral_whitelist,
                "responses_mode": c.responses_mode,
                "effective_responses_mode": c.effective_responses_mode(),
                "keep_images_mistral": c.keep_images_mistral,
                "keep_images_other": c.keep_images_other,
                "upstream_timeout": c.upstream_timeout,
                "normalize_content_arrays": c.normalize_content_arrays,
                "accounts": rt.accounts.summary(),
                "concurrency": rt.limiter.stats(),
            },
            headers={"Access-Control-Allow-Origin": "*"},
        )

    # ---------------------------------------------------------------- Admin API

    @app.get("/api/admin/status")
    async def admin_status(request: Request):
        if not rt.admin_ok(request):
            return JSONResponse({"error": "admin access forbidden; set ADMIN_TOKEN or connect locally"}, status_code=403)
        c = rt.config
        return {
            "ok": True,
            "version": __version__,
            "upstream_mode": c.upstream_mode,
            "upstream_base": c.upstream_base,
            "mistral_chat_endpoint": c.mistral_chat_endpoint,
            "effective_responses_mode": c.effective_responses_mode(),
            "telemetry": rt.telemetry.summary(),
            "accounts": rt.accounts.summary(),
            "concurrency": rt.limiter.stats(),
        }

    @app.get("/api/admin/config")
    async def admin_get_config(request: Request):
        if not rt.admin_ok(request):
            return JSONResponse({"error": "admin access forbidden"}, status_code=403)
        return {
            "config": rt.config.to_dict(mask_secrets=True),
            "persisted": bool(rt.config.config_path),
            "config_path": rt.config.config_path,
        }

    @app.put("/api/admin/config")
    async def admin_put_config(request: Request):
        if not rt.admin_ok(request):
            return JSONResponse({"error": "admin access forbidden"}, status_code=403)
        try:
            patch = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        if not isinstance(patch, dict):
            return JSONResponse({"error": "config patch must be an object"}, status_code=400)
        saved = await rt.apply_config_patch(patch)
        return {
            "ok": True,
            "saved": saved,
            "config": rt.config.to_dict(mask_secrets=True),
        }

    @app.get("/api/admin/logs")
    async def admin_logs(request: Request, limit: int = 80):
        if not rt.admin_ok(request):
            return JSONResponse({"error": "admin access forbidden"}, status_code=403)
        return {"events": rt.telemetry.recent(limit=max(1, min(limit, 250)))}

    # ---------------------------------------------------------------- Multi-Account Pool Admin API

    @app.get("/api/admin/accounts")
    async def admin_list_accounts(request: Request):
        if not rt.admin_ok(request):
            return JSONResponse({"error": "admin access forbidden"}, status_code=403)
        return {
            "accounts": [a.to_dict(mask_secret=True) for a in rt.accounts.list_entries()],
            "summary": rt.accounts.summary(),
            "persisted": bool(rt.accounts.store_path),
            "passthrough_auth": rt.config.passthrough_auth,
        }

    @app.post("/api/admin/accounts")
    async def admin_add_account(request: Request):
        if not rt.admin_ok(request):
            return JSONResponse({"error": "admin access forbidden"}, status_code=403)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)

        auto_refresh = bool(body.get("auto_refresh", True))
        created: list[AccountEntry] = []

        # Support batch import via newline-separated keys (`key` or `name|key`)
        batch_text = str(body.get("batch_keys") or "").strip()
        if batch_text:
            for line in batch_text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "|" in line:
                    n, _, k = line.partition("|")
                    acc = rt.accounts.add(
                        api_key=k.strip(),
                        name=n.strip(),
                        priority=int(body.get("priority", 50)),
                        upstream_base=str(body.get("upstream_base") or ""),
                    )
                else:
                    acc = rt.accounts.add(
                        api_key=line,
                        priority=int(body.get("priority", 50)),
                        upstream_base=str(body.get("upstream_base") or ""),
                    )
                created.append(acc)
        else:
            api_key = str(body.get("api_key") or "").strip()
            if not api_key:
                return JSONResponse({"error": "api_key is required"}, status_code=400)
            tags_raw = body.get("tags") or []
            tags = [x.strip() for x in tags_raw.split(",")] if isinstance(tags_raw, str) else list(tags_raw)
            acc = rt.accounts.add(
                api_key=api_key,
                name=str(body.get("name") or ""),
                email=str(body.get("email") or ""),
                priority=int(body.get("priority", 50)),
                upstream_base=str(body.get("upstream_base") or ""),
                plan=str(body.get("plan") or "API"),
                tags=tags,
                enabled=bool(body.get("enabled", True)),
            )
            created.append(acc)

        if auto_refresh and created:
            await asyncio.gather(
                *[rt.accounts.refresh_account(a.id, rt.client, rt.config.upstream_base) for a in created]
            )

        return {
            "ok": True,
            "created": [a.to_dict(mask_secret=True) for a in created],
            "summary": rt.accounts.summary(),
        }

    @app.patch("/api/admin/accounts/{account_id}")
    async def admin_update_account(account_id: str, request: Request):
        if not rt.admin_ok(request):
            return JSONResponse({"error": "admin access forbidden"}, status_code=403)
        try:
            patch = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        acc = rt.accounts.update(account_id, patch)
        if not acc:
            return JSONResponse({"error": "account not found"}, status_code=404)
        return {"ok": True, "account": acc.to_dict(mask_secret=True)}

    @app.delete("/api/admin/accounts/{account_id}")
    async def admin_delete_account(account_id: str, request: Request):
        if not rt.admin_ok(request):
            return JSONResponse({"error": "admin access forbidden"}, status_code=403)
        if not rt.accounts.delete(account_id):
            return JSONResponse({"error": "account not found"}, status_code=404)
        return {"ok": True, "summary": rt.accounts.summary()}

    @app.post("/api/admin/accounts/{account_id}/activate")
    async def admin_activate_account(account_id: str, request: Request):
        if not rt.admin_ok(request):
            return JSONResponse({"error": "admin access forbidden"}, status_code=403)
        acc = rt.accounts.activate(account_id)
        if not acc:
            return JSONResponse({"error": "account not found"}, status_code=404)
        return {"ok": True, "account": acc.to_dict(mask_secret=True)}

    @app.post("/api/admin/accounts/{account_id}/refresh")
    async def admin_refresh_account(account_id: str, request: Request):
        if not rt.admin_ok(request):
            return JSONResponse({"error": "admin access forbidden"}, status_code=403)
        acc = await rt.accounts.refresh_account(account_id, rt.client, rt.config.upstream_base)
        if not acc:
            return JSONResponse({"error": "account not found"}, status_code=404)
        return {"ok": True, "account": acc.to_dict(mask_secret=True)}

    @app.post("/api/admin/accounts/refresh-all")
    async def admin_refresh_all_accounts(request: Request):
        if not rt.admin_ok(request):
            return JSONResponse({"error": "admin access forbidden"}, status_code=403)
        entries = rt.accounts.list_entries()
        if entries:
            await asyncio.gather(
                *[rt.accounts.refresh_account(a.id, rt.client, rt.config.upstream_base) for a in entries]
            )
        return {
            "ok": True,
            "accounts": [a.to_dict(mask_secret=True) for a in rt.accounts.list_entries()],
            "summary": rt.accounts.summary(),
        }

    @app.get("/api/admin/accounts/{account_id}/raw")
    async def admin_account_raw(account_id: str, request: Request):
        if not rt.admin_ok(request):
            return JSONResponse({"error": "admin access forbidden"}, status_code=403)
        acc = rt.accounts.get(account_id)
        if not acc:
            return JSONResponse({"error": "account not found"}, status_code=404)
        return {"account": acc.to_dict(mask_secret=True, include_raw=True)}

    @app.get("/api/admin/mistral-account")
    async def admin_mistral_account(request: Request):
        if not rt.admin_ok(request):
            return JSONResponse({"error": "admin access forbidden"}, status_code=403)
        api_key = rt.config.upstream_api_key
        if not api_key:
            sel = rt.accounts.select_account()
            if sel:
                api_key = sel.api_key
        if not api_key:
            api_key = request.headers.get("x-mistral-key", "").strip()
        if not api_key:
            return JSONResponse(
                {"error": "No API key configured. Add an account in Accounts tab or set upstream_api_key."},
                status_code=400,
            )

        base = rt.config.upstream_base.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

        async def _probe(path: str) -> dict[str, Any]:
            try:
                r = await rt.client.get(f"{base}{path}", headers=headers, timeout=15.0)
                if r.status_code == 200:
                    return {"ok": True, "status": 200, "data": r.json()}
                return {"ok": False, "status": r.status_code, "detail": r.text[:300]}
            except Exception as exc:
                return {"ok": False, "status": 0, "detail": _fmt_exc(exc)}

        models_res, sub_res, budget_res = await asyncio.gather(
            _probe("/v1/models"),
            _probe("/v1/billing/subscription"),
            _probe("/v1/billing/budget"),
        )
        models_list = []
        if models_res.get("ok") and isinstance(models_res.get("data"), dict):
            models_list = [m.get("id") for m in models_res["data"].get("data", []) if isinstance(m, dict)]

        return {
            "upstream": base,
            "models_count": len(models_list),
            "models": models_list[:60],
            "subscription": sub_res,
            "budget": budget_res,
        }

    # ---------------------------------------------------------------- Proxy: /v1/chat/completions

    @app.post("/v1/chat/completions")
    @app.post("/v1/chat/completions/")
    @app.post("/gw/v1/chat/completions")
    @app.post("/gw/v1/chat/completions/")
    async def chat_completions(request: Request):
        t0 = time.perf_counter()
        if not rt.auth_ok(request):
            return JSONResponse({"error": {"message": "invalid api key"}}, status_code=401)

        raw = await request.body()
        try:
            payload = json.loads(raw)
        except Exception:
            return JSONResponse({"error": {"message": "invalid json"}}, status_code=400)

        cfg = rt.config
        model = payload.get("model")
        mistral = cfg.is_mistral_model(model)

        slot_key, cap, sem = await rt.limiter.acquire_slot(model)
        if cap is None and slot_key is not None:
            rt.telemetry.record(
                method="POST",
                path=request.url.path,
                model=str(model or ""),
                mistral=mistral,
                route_mode="chat",
                stream=bool(payload.get("stream")),
                status=429,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                body_kb=len(raw) // 1024,
                error="concurrency_limit_exceeded",
            )
            return _concurrency_exceeded(model, rt.limiter.limit_for(model) or 0)

        handed_off = False
        try:
            use_openai_compat = (
                mistral and cfg.upstream_mode == "direct_mistral" and cfg.mistral_chat_endpoint == "openai_compat"
            )
            if use_openai_compat:
                payload, stripped = strip_global_fields(payload)
                body, total_imgs, dropped = encode_and_cap_chat(
                    payload, cfg.keep_images_mistral, cfg.max_body_bytes_mistral
                )
            elif mistral:
                payload, stripped = sanitize_mistral_chat(
                    payload,
                    strict_whitelist=cfg.strict_mistral_whitelist,
                    keep_stream_options_for_new_api=(cfg.upstream_mode == "new_api"),
                )
                body, total_imgs, dropped = encode_and_cap_chat(
                    payload, cfg.keep_images_mistral, cfg.max_body_bytes_mistral
                )
            else:
                payload, stripped = strip_global_fields(payload)
                body, total_imgs, dropped = encode_and_cap_chat(
                    payload, cfg.keep_images_other, cfg.max_body_bytes_other
                )

            is_stream = bool(payload.get("stream"))
            path_suffix = "/openai/v1/chat/completions" if use_openai_compat else "/v1/chat/completions"
            chat_route_mode = "chat_openai_compat" if use_openai_compat else "chat"

            try:
                upstream, used_acc = await rt.dispatch_upstream(
                    request,
                    path_suffix=path_suffix,
                    body=body,
                    model=model,
                    stream=is_stream,
                )
            except httpx.HTTPError as exc:
                err_detail = _fmt_exc(exc)
                status = 504 if isinstance(exc, httpx.TimeoutException) else 502
                rt.telemetry.record(
                    method="POST",
                    path=request.url.path,
                    model=str(model or ""),
                    mistral=mistral,
                    route_mode=chat_route_mode,
                    stream=is_stream,
                    status=status,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    body_kb=len(body) // 1024,
                    images_total=total_imgs,
                    images_dropped=dropped,
                    stripped_fields=stripped,
                    error=err_detail,
                )
                return JSONResponse(
                    {"error": {"message": f"upstream connection error: {err_detail}", "type": "gateway_error"}},
                    status_code=status,
                )

            acc_label = (used_acc.email or used_acc.name) if used_acc else ""

            if is_stream:
                norm_counter = [0]
                stream_err = [""]

                async def gen():
                    try:
                        async for chunk in normalize_chat_sse_stream(
                            upstream,
                            enable_normalize=cfg.normalize_content_arrays,
                            counter=norm_counter,
                        ):
                            yield chunk
                    except httpx.HTTPError as exc:
                        stream_err[0] = _fmt_exc(exc)
                        log.error("upstream stream error model=%s: %s", model, stream_err[0])
                    finally:
                        await upstream.aclose()
                        rt.limiter.release_slot(slot_key, sem)
                        rt.telemetry.record(
                            method="POST",
                            path=request.url.path,
                            model=str(model or ""),
                            mistral=mistral,
                            route_mode=chat_route_mode,
                            stream=True,
                            status=upstream.status_code,
                            latency_ms=int((time.perf_counter() - t0) * 1000),
                            body_kb=len(body) // 1024,
                            images_total=total_imgs,
                            images_dropped=dropped,
                            stripped_fields=stripped,
                            normalized_chunks=norm_counter[0],
                            account=acc_label,
                            error=stream_err[0],
                        )

                handed_off = True
                return StreamingResponse(
                    gen(),
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "text/event-stream"),
                )

            content = upstream.content
            ctype = upstream.headers.get("content-type", "application/json")
            norm_count = 0
            if cfg.normalize_content_arrays and upstream.status_code == 200 and "json" in (ctype or ""):
                try:
                    obj = json.loads(content)
                    if normalize_chat_object(obj):
                        norm_count = 1
                        content = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                except Exception:
                    pass

            rt.telemetry.record(
                method="POST",
                path=request.url.path,
                model=str(model or ""),
                mistral=mistral,
                route_mode=chat_route_mode,
                stream=False,
                status=upstream.status_code,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                body_kb=len(body) // 1024,
                images_total=total_imgs,
                images_dropped=dropped,
                stripped_fields=stripped,
                normalized_chunks=norm_count,
                account=acc_label,
                error="" if upstream.status_code < 400 else content[:200].decode("utf-8", errors="ignore"),
            )
            return Response(content=content, status_code=upstream.status_code, media_type=ctype)
        finally:
            if not handed_off:
                rt.limiter.release_slot(slot_key, sem)

    # ---------------------------------------------------------------- Proxy: /v1/responses

    @app.post("/v1/responses")
    @app.post("/v1/responses/")
    @app.post("/gw/v1/responses")
    @app.post("/gw/v1/responses/")
    async def responses_endpoint(request: Request):
        t0 = time.perf_counter()
        if not rt.auth_ok(request):
            return JSONResponse({"error": {"message": "invalid api key"}}, status_code=401)

        raw = await request.body()
        try:
            payload = json.loads(raw)
        except Exception:
            return JSONResponse({"error": {"message": "invalid json"}}, status_code=400)

        cfg = rt.config
        model = payload.get("model")
        mistral = cfg.is_mistral_model(model)

        slot_key, cap, sem = await rt.limiter.acquire_slot(model)
        if cap is None and slot_key is not None:
            rt.telemetry.record(
                method="POST",
                path=request.url.path,
                model=str(model or ""),
                mistral=mistral,
                route_mode="responses",
                stream=bool(payload.get("stream")),
                status=429,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                body_kb=len(raw) // 1024,
                error="concurrency_limit_exceeded",
            )
            return _concurrency_exceeded(model, rt.limiter.limit_for(model) or 0)

        handed_off = False
        try:
            should_convert = mistral and cfg.effective_responses_mode() == "convert"

            if not should_convert:
                if mistral:
                    payload, stripped = sanitize_mistral_responses(payload)
                    body, total_imgs, dropped = encode_and_cap_responses(
                        payload, cfg.keep_images_mistral, cfg.max_body_bytes_mistral
                    )
                else:
                    payload, stripped = strip_global_fields(payload)
                    body, total_imgs, dropped = encode_and_cap_responses(
                        payload, cfg.keep_images_other, cfg.max_body_bytes_other
                    )
                is_stream = bool(payload.get("stream"))
                try:
                    upstream, used_acc = await rt.dispatch_upstream(
                        request,
                        path_suffix="/v1/responses",
                        body=body,
                        model=model,
                        stream=is_stream,
                    )
                except httpx.HTTPError as exc:
                    err_detail = _fmt_exc(exc)
                    status = 504 if isinstance(exc, httpx.TimeoutException) else 502
                    rt.telemetry.record(
                        method="POST",
                        path=request.url.path,
                        model=str(model or ""),
                        mistral=mistral,
                        route_mode="responses_native",
                        stream=is_stream,
                        status=status,
                        latency_ms=int((time.perf_counter() - t0) * 1000),
                        body_kb=len(body) // 1024,
                        images_total=total_imgs,
                        images_dropped=dropped,
                        stripped_fields=stripped,
                        error=err_detail,
                    )
                    return JSONResponse(
                        {"error": {"message": f"upstream connection error: {err_detail}", "type": "gateway_error"}},
                        status_code=status,
                    )

                acc_label = (used_acc.email or used_acc.name) if used_acc else ""

                if is_stream:
                    stream_err = [""]

                    async def gen_native():
                        try:
                            async for chunk in upstream.aiter_bytes():
                                yield chunk
                        except httpx.HTTPError as exc:
                            stream_err[0] = _fmt_exc(exc)
                        finally:
                            await upstream.aclose()
                            rt.limiter.release_slot(slot_key, sem)
                            rt.telemetry.record(
                                method="POST",
                                path=request.url.path,
                                model=str(model or ""),
                                mistral=mistral,
                                route_mode="responses_native",
                                stream=True,
                                status=upstream.status_code,
                                latency_ms=int((time.perf_counter() - t0) * 1000),
                                body_kb=len(body) // 1024,
                                images_total=total_imgs,
                                images_dropped=dropped,
                                stripped_fields=stripped,
                                account=acc_label,
                                error=stream_err[0],
                            )

                    handed_off = True
                    return StreamingResponse(
                        gen_native(),
                        status_code=upstream.status_code,
                        media_type=upstream.headers.get("content-type", "text/event-stream"),
                    )

                content = upstream.content
                rt.telemetry.record(
                    method="POST",
                    path=request.url.path,
                    model=str(model or ""),
                    mistral=mistral,
                    route_mode="responses_native",
                    stream=False,
                    status=upstream.status_code,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    body_kb=len(body) // 1024,
                    images_total=total_imgs,
                    images_dropped=dropped,
                    stripped_fields=stripped,
                    account=acc_label,
                    error="" if upstream.status_code < 400 else content[:200].decode("utf-8", errors="ignore"),
                )
                return Response(
                    content=content,
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "application/json"),
                )

            # Convert Responses API <-> Chat Completions
            chat_payload = responses_req_to_chat(payload, preserve_strict=True)
            chat_payload, stripped = sanitize_mistral_chat(
                chat_payload,
                strict_whitelist=cfg.strict_mistral_whitelist,
                keep_stream_options_for_new_api=(cfg.upstream_mode == "new_api"),
            )
            if chat_payload.get("stream") and cfg.upstream_mode == "new_api":
                chat_payload.setdefault("stream_options", {})["include_usage"] = True
            body, total_imgs, dropped = encode_and_cap_chat(
                chat_payload, cfg.keep_images_mistral, cfg.max_body_bytes_mistral
            )
            is_stream = bool(chat_payload.get("stream"))

            try:
                upstream, used_acc = await rt.dispatch_upstream(
                    request,
                    path_suffix="/v1/chat/completions",
                    body=body,
                    model=model,
                    stream=is_stream,
                )
            except httpx.HTTPError as exc:
                err_detail = _fmt_exc(exc)
                status = 504 if isinstance(exc, httpx.TimeoutException) else 502
                rt.telemetry.record(
                    method="POST",
                    path=request.url.path,
                    model=str(model or ""),
                    mistral=mistral,
                    route_mode="responses_convert",
                    stream=is_stream,
                    status=status,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    body_kb=len(body) // 1024,
                    images_total=total_imgs,
                    images_dropped=dropped,
                    stripped_fields=stripped,
                    error=err_detail,
                )
                return JSONResponse(
                    {"error": {"message": f"upstream connection error: {err_detail}", "type": "gateway_error"}},
                    status_code=status,
                )

            acc_label = (used_acc.email or used_acc.name) if used_acc else ""

            if is_stream:
                if upstream.status_code >= 400:
                    err_body = await upstream.aread()
                    ct = upstream.headers.get("content-type", "application/json")
                    await upstream.aclose()
                    rt.telemetry.record(
                        method="POST",
                        path=request.url.path,
                        model=str(model or ""),
                        mistral=mistral,
                        route_mode="responses_convert",
                        stream=True,
                        status=upstream.status_code,
                        latency_ms=int((time.perf_counter() - t0) * 1000),
                        body_kb=len(body) // 1024,
                        images_total=total_imgs,
                        images_dropped=dropped,
                        stripped_fields=stripped,
                        account=acc_label,
                        error=err_body[:200].decode("utf-8", errors="ignore"),
                    )
                    return Response(content=err_body, status_code=upstream.status_code, media_type=ct)

                norm_counter = [0]
                stream_err = [""]

                async def gen_resp_sse():
                    try:
                        async for ev in stream_chat_to_responses(upstream, model, counter=norm_counter):
                            yield ev
                    except httpx.HTTPError as exc:
                        stream_err[0] = _fmt_exc(exc)
                    finally:
                        await upstream.aclose()
                        rt.limiter.release_slot(slot_key, sem)
                        rt.telemetry.record(
                            method="POST",
                            path=request.url.path,
                            model=str(model or ""),
                            mistral=mistral,
                            route_mode="responses_convert",
                            stream=True,
                            status=200,
                            latency_ms=int((time.perf_counter() - t0) * 1000),
                            body_kb=len(body) // 1024,
                            images_total=total_imgs,
                            images_dropped=dropped,
                            stripped_fields=stripped,
                            normalized_chunks=norm_counter[0],
                            account=acc_label,
                            error=stream_err[0],
                        )

                handed_off = True
                return StreamingResponse(gen_resp_sse(), status_code=200, media_type="text/event-stream")

            if upstream.status_code >= 400:
                err_content = upstream.content
                rt.telemetry.record(
                    method="POST",
                    path=request.url.path,
                    model=str(model or ""),
                    mistral=mistral,
                    route_mode="responses_convert",
                    stream=False,
                    status=upstream.status_code,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    body_kb=len(body) // 1024,
                    images_total=total_imgs,
                    images_dropped=dropped,
                    stripped_fields=stripped,
                    account=acc_label,
                    error=err_content[:200].decode("utf-8", errors="ignore"),
                )
                return Response(
                    content=err_content,
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "application/json"),
                )

            chat_json = upstream.json()
            resp_json = chat_resp_to_responses(chat_json, model)
            rt.telemetry.record(
                method="POST",
                path=request.url.path,
                model=str(model or ""),
                mistral=mistral,
                route_mode="responses_convert",
                stream=False,
                status=200,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                body_kb=len(body) // 1024,
                images_total=total_imgs,
                images_dropped=dropped,
                stripped_fields=stripped,
                account=acc_label,
            )
            return JSONResponse(content=resp_json, status_code=200)
        finally:
            if not handed_off:
                rt.limiter.release_slot(slot_key, sem)

    # ---------------------------------------------------------------- Fallback passthrough

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    async def fallback_proxy(request: Request, full_path: str):
        if full_path.startswith("api/admin"):
            return JSONResponse({"error": "not found"}, status_code=404)
        if not rt.auth_ok(request):
            return JSONResponse({"error": {"message": "invalid api key"}}, status_code=401)

        clean_path = full_path[3:] if full_path.startswith("gw/") else full_path
        qs = request.url.query
        body = await request.body()

        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() in SAFE_FALLBACK_HEADERS
        }
        headers["Accept-Encoding"] = "identity"

        base = rt.config.upstream_base
        if rt.should_use_account_pool(request):
            acc = rt.accounts.select_account()
            if acc:
                base = (acc.upstream_base or base).rstrip("/")
                headers.pop("authorization", None)
                headers.pop("Authorization", None)
                headers.pop("x-api-key", None)
                headers["Authorization"] = f"Bearer {acc.api_key}"
        elif not rt.config.passthrough_auth and rt.config.upstream_api_key:
            headers.pop("authorization", None)
            headers.pop("Authorization", None)
            headers.pop("x-api-key", None)
            headers["Authorization"] = f"Bearer {rt.config.upstream_api_key}"
        elif (
            rt.config.passthrough_auth
            and "authorization" not in {k.lower() for k in headers}
            and "x-api-key" not in {k.lower() for k in headers}
            and rt.config.upstream_api_key
        ):
            headers["Authorization"] = f"Bearer {rt.config.upstream_api_key}"

        url = f"{base}/{clean_path}" + (f"?{qs}" if qs else "")
        try:
            req = rt.client.build_request(request.method, url, content=body or None, headers=headers)
            upstream = await rt.client.send(req, stream=True)
        except httpx.HTTPError as exc:
            err_detail = _fmt_exc(exc)
            status = 504 if isinstance(exc, httpx.TimeoutException) else 502
            log.error("fallback upstream error path=%s status=%d: %s", clean_path, status, err_detail)
            return JSONResponse(
                {"error": {"message": f"upstream connection error: {err_detail}", "type": "gateway_error"}},
                status_code=status,
            )

        async def gen():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            except httpx.HTTPError as exc:
                log.error("fallback stream error path=%s: %s", clean_path, _fmt_exc(exc))
            finally:
                await upstream.aclose()

        return StreamingResponse(
            gen(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )

    return app
