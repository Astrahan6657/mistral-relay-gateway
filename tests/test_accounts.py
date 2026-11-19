from pathlib import Path

import httpx
import pytest

from mistral_relay_gateway.app import create_app
from mistral_relay_gateway.config import GatewayConfig


@pytest.mark.asyncio
async def test_multi_account_probe_quotas_and_failover(tmp_path: Path):
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        calls.append((request.url.path, auth))

        if request.url.path == "/v1/whoami":
            email = "acct1@example.com" if "key-1" in auth else "acct2@example.com"
            return httpx.Response(
                200,
                json={
                    "principal": {"principal_type": "user", "id": "u1", "email": email},
                    "organization": {"id": "o1", "name": "DevOrg", "org_tier": "pro"},
                    "workspace": {"id": "w1", "name": "ProdWS"},
                },
            )
        if request.url.path == "/v1/users/me":
            return httpx.Response(404)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "zai-glm-5-3"},
                        {"id": "deepseek-v4-1-flash-1017"},
                        {"id": "mistral-large-latest"},
                    ]
                },
            )
        if request.url.path == "/v1/billing/subscription":
            return httpx.Response(
                200,
                json={"plan_name": "pro", "payg_enabled": True, "vibe_payg_enabled": True, "enterprise": False},
            )
        if request.url.path == "/v1/billing/budget":
            return httpx.Response(
                200,
                json={
                    "api_budget": {
                        "usage_percentage": 1.0,
                        "initial_budget": 100.0,
                        "currency": "USD",
                        "reset_at": "2026-12-01T00:00:00Z",
                        "payg_enabled": True,
                    },
                    "vibe_budget": {
                        "usage_percentage": 6.0,
                        "initial_budget": 50.0,
                        "currency": "USD",
                        "reset_at": "2026-12-01T00:00:00Z",
                        "payg_enabled": False,
                    },
                },
            )
        if request.url.path == "/v1/admin/spend-limit":
            return httpx.Response(
                200,
                json={
                    "limits": {
                        "completion": {
                            "no_monthly_limit": False,
                            "monthly_limit_reached": False,
                            "total_usage": 25.0,
                            "usage_limit": 100.0,
                        },
                        "currency": "USD",
                    }
                },
            )
        if request.url.path == "/v1/admin/rate-limit":
            return httpx.Response(
                200,
                json={"requests_per_second": 5, "tokens_limits_by_model": {"zai-glm-5-3": 500000}},
            )
        if request.url.path == "/v1/chat/completions":
            if "key-1" in auth:
                return httpx.Response(402, json={"error": "quota exhausted"})
            return httpx.Response(
                200,
                headers={
                    "x-ratelimit-remaining-tokens-minute": "420000",
                    "x-ratelimit-limit-tokens-minute": "500000",
                },
                json={"choices": [{"message": {"role": "assistant", "content": "served by acct2"}}]},
            )
        return httpx.Response(404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cfg = GatewayConfig(
        upstream_mode="direct_mistral",
        upstream_base="https://api.mistral.ai",
        passthrough_auth=False,
        config_path=str(tmp_path / "config.json"),
    )
    app = create_app(cfg, http_client=mock_client)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
        headers={"X-MRG-Console": "1"},
    ) as client:
        # 1) Add two accounts via single + batch
        r1 = await client.post(
            "/api/admin/accounts",
            json={"api_key": "sk-key-1-aaaaaaaa", "priority": 80, "auto_refresh": True},
        )
        assert r1.status_code == 200
        a1 = r1.json()["created"][0]
        assert a1["email"] == "acct1@example.com"
        assert a1["plan"] == "PRO"
        assert "GLM 5.3" in a1["tags"]
        assert "DeepSeek V4" in a1["tags"]
        quotas_by_id = {q["id"]: q for q in a1["quotas"]}
        assert quotas_by_id["api_budget"]["remaining_pct"] == 99.0
        assert quotas_by_id["vibe_budget"]["remaining_pct"] == 94.0
        assert quotas_by_id["monthly_spend"]["remaining_pct"] == 75.0
        assert quotas_by_id["rate_limits"]["wide"] is True
        assert a1["api_key"] == "••••••••"

        r2 = await client.post(
            "/api/admin/accounts",
            json={"batch_keys": "backup|sk-key-2-bbbbbbbb", "priority": 50, "auto_refresh": True},
        )
        assert r2.status_code == 200
        a2 = r2.json()["created"][0]
        assert a2["email"] == "acct2@example.com"
        assert (tmp_path / "accounts.json").is_file()

        # 2) Chat request hits priority-80 acct1 -> gets 402 -> automatically fails over to acct2 -> returns 200!
        chat = await client.post(
            "/v1/chat/completions",
            json={"model": "zai-glm-5-3", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert chat.status_code == 200
        assert chat.json()["choices"][0]["message"]["content"] == "served by acct2"

        # 3) Verify acct1 is cooling (402) and acct2 ingested live TPM header (84.0%)
        listed = (await client.get("/api/admin/accounts")).json()
        by_id = {x["id"]: x for x in listed["accounts"]}
        assert by_id[a1["id"]]["last_status"] == "402"
        assert by_id[a1["id"]]["cooldown_until"] > 0
        a2_quotas = {q["id"]: q for q in by_id[a2["id"]]["quotas"]}
        assert a2_quotas["live_tpm"]["remaining_pct"] == 84.0

        # 4) Verify telemetry recorded the failover account
        ev = (await client.get("/api/admin/logs")).json()["events"][0]
        assert ev["account"] == "acct2@example.com"

        # 5) Activate / raw / delete endpoints
        act = await client.post(f"/api/admin/accounts/{a2['id']}/activate")
        assert act.status_code == 200 and act.json()["account"]["pinned"] is True

        raw = await client.get(f"/api/admin/accounts/{a1['id']}/raw")
        assert raw.status_code == 200 and raw.json()["account"]["raw_snapshot"]["budget"]["status"] == 200

        dele = await client.delete(f"/api/admin/accounts/{a1['id']}")
        assert dele.status_code == 200 and dele.json()["summary"]["total"] == 1

    await mock_client.aclose()
