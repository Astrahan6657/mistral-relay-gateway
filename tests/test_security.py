import httpx
import pytest

from mistral_relay_gateway.app import create_app
from mistral_relay_gateway.config import GatewayConfig


@pytest.mark.asyncio
async def test_no_open_relay_for_non_local_clients():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cfg = GatewayConfig(
        upstream_mode="direct_mistral",
        upstream_base="https://api.mistral.ai",
        passthrough_auth=True,
        upstream_api_key="sk-server-owned",
    )
    app = create_app(cfg, http_client=mock_client, bind_host="0.0.0.0")
    app.state.runtime.accounts.add(api_key="sk-pool-secret", name="pool-1")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        # 1) Proxied / remote client (X-Forwarded-For) with NO key -> MUST be rejected (401), never open relay
        r_remote = await client.post(
            "/v1/chat/completions",
            headers={"X-Forwarded-For": "203.0.113.9"},
            json={"model": "mistral-large-latest", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r_remote.status_code == 401

        # 2) Non-local Host header with NO key -> MUST be rejected (401)
        r_host = await client.post(
            "/v1/chat/completions",
            headers={"Host": "public.example.com"},
            json={"model": "mistral-large-latest", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r_host.status_code == 401

        # 3) Direct localhost with NO key -> allowed for local personal desktop use
        r_local = await client.post(
            "/v1/chat/completions",
            json={"model": "mistral-large-latest", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r_local.status_code == 200

    # 4) When GATEWAY_KEY is configured, remote request MUST supply matching GATEWAY_KEY
    cfg.gateway_key = "gw-secret-token"
    cfg.passthrough_auth = False
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        r_bad = await client.post(
            "/v1/chat/completions",
            headers={"Host": "public.example.com", "Authorization": "Bearer wrong-key"},
            json={"model": "mistral-large-latest", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r_bad.status_code == 401

        r_ok = await client.post(
            "/v1/chat/completions",
            headers={"Host": "public.example.com", "Authorization": "Bearer gw-secret-token"},
            json={"model": "mistral-large-latest", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r_ok.status_code == 200

    await mock_client.aclose()


@pytest.mark.asyncio
async def test_admin_dns_rebinding_and_csrf_protection():
    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200)))
    cfg = GatewayConfig(upstream_mode="direct_mistral")
    app = create_app(cfg, http_client=mock_client)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        # Missing X-MRG-Console custom header -> 403
        r_no_hdr = await client.get("/api/admin/status")
        assert r_no_hdr.status_code == 403

        # DNS rebinding Host: evil.com even with custom header -> 403
        r_rebind = await client.get(
            "/api/admin/status",
            headers={"Host": "evil.com:8790", "X-MRG-Console": "1"},
        )
        assert r_rebind.status_code == 403

        # Malicious browser Origin -> 403
        r_origin = await client.get(
            "/api/admin/status",
            headers={"Origin": "https://evil.com", "X-MRG-Console": "1"},
        )
        assert r_origin.status_code == 403

        # Reverse-proxied X-Forwarded-For without ADMIN_TOKEN -> 403
        r_xff = await client.get(
            "/api/admin/status",
            headers={"X-Forwarded-For": "198.51.100.1", "X-MRG-Console": "1"},
        )
        assert r_xff.status_code == 403

        # Genuine local Web Console call -> 200
        r_ok = await client.get(
            "/api/admin/status",
            headers={"Origin": "http://127.0.0.1:8790", "X-MRG-Console": "1"},
        )
        assert r_ok.status_code == 200

    await mock_client.aclose()


@pytest.mark.asyncio
async def test_fallback_proxy_scrubs_sensitive_headers_and_handles_errors():
    seen_headers: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append({k.lower(): v for k, v in request.headers.items()})
        if request.url.path == "/v1/boom":
            raise httpx.ConnectError("upstream down")
        if request.url.path == "/v1/slow":
            raise httpx.ReadTimeout("upstream slow")
        return httpx.Response(200, json={"data": []})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cfg = GatewayConfig(
        upstream_mode="direct_mistral",
        upstream_base="https://api.mistral.ai",
        passthrough_auth=False,
        upstream_api_key="sk-upstream",
    )
    app = create_app(cfg, http_client=mock_client)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        r = await client.get(
            "/v1/models",
            headers={
                "Cookie": "session=supersecret",
                "X-Admin-Token": "admin-secret",
                "X-MRG-Console": "1",
                "X-Custom-Leak": "should-not-forward",
            },
        )
        assert r.status_code == 200
        forwarded = seen_headers[-1]
        assert "cookie" not in forwarded
        assert "x-admin-token" not in forwarded
        assert "x-mrg-console" not in forwarded
        assert "x-custom-leak" not in forwarded
        assert forwarded["authorization"] == "Bearer sk-upstream"

        # Upstream connection error returns 502 (not 500)
        r_err = await client.get("/v1/boom")
        assert r_err.status_code == 502
        assert r_err.json()["error"]["type"] == "gateway_error"

        # Upstream timeout returns 504 (not 500)
        r_to = await client.get("/v1/slow")
        assert r_to.status_code == 504

        # Non-local unauthenticated fallback call -> 401
        r_unauth = await client.get("/v1/models", headers={"X-Forwarded-For": "203.0.113.5"})
        assert r_unauth.status_code == 401

    await mock_client.aclose()
