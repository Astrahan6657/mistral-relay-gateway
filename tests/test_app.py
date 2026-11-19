import json
from pathlib import Path

import httpx
import pytest

from mistral_relay_gateway.app import create_app
from mistral_relay_gateway.config import GatewayConfig


@pytest.mark.asyncio
async def test_direct_mistral_chat_retry_and_thinkchunk_flatten():
    captured: list[tuple[str, dict]] = []
    attempts = {"chat": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            attempts["chat"] += 1
            body = json.loads(request.content)
            captured.append((request.url.path, body))
            if attempts["chat"] == 1:
                return httpx.Response(502, json={"error": "bad gateway"})
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-1",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": [
                                    {"type": "thinking", "thinking": [{"type": "text", "text": "reason"}]},
                                    {"type": "text", "text": "final answer"},
                                ],
                            }
                        }
                    ],
                },
            )
        return httpx.Response(404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cfg = GatewayConfig(
        upstream_mode="direct_mistral",
        upstream_base="https://api.mistral.ai",
        passthrough_auth=False,
        upstream_api_key="test-mistral-key",
    )
    app = create_app(cfg, http_client=mock_client)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        res = await client.post(
            "/v1/chat/completions",
            json={
                "model": "zai-glm-5-3",
                "store": False,
                "prompt_cache_key": "keep-me",
                "messages": [{"role": "developer", "content": "sys"}, {"role": "user", "content": "hi"}],
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["choices"][0]["message"]["content"] == "final answer"
        assert data["choices"][0]["message"]["reasoning_content"] == "reason"
        assert attempts["chat"] == 2
        sent_body = captured[-1][1]
        assert "store" not in sent_body
        assert sent_body["prompt_cache_key"] == "keep-me"
        assert sent_body["messages"][0]["role"] == "system"

        logs = await client.get("/api/admin/logs", headers={"X-MRG-Console": "1"})
        assert logs.status_code == 200
        ev = logs.json()["events"][0]
        assert ev["model"] == "zai-glm-5-3"
        assert ev["normalized_chunks"] == 1
        assert "store" in ev["stripped_fields"]

    await mock_client.aclose()


@pytest.mark.asyncio
async def test_responses_auto_routing_direct_vs_new_api():
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        body = json.loads(request.content)
        if request.url.path == "/v1/responses":
            assert "previous_response_id" not in body
            return httpx.Response(200, json={"id": "resp_native", "object": "response", "output": []})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "converted ok"}}]},
            )
        return httpx.Response(404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    # 1) direct_mistral -> native /v1/responses
    cfg_direct = GatewayConfig(
        upstream_mode="direct_mistral",
        upstream_base="https://api.mistral.ai",
        passthrough_auth=False,
        upstream_api_key="k",
    )
    app1 = create_app(cfg_direct, http_client=mock_client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app1), base_url="http://127.0.0.1") as client:
        r1 = await client.post(
            "/v1/responses",
            json={"model": "mistral-large-latest", "input": "hi", "previous_response_id": "old"},
        )
        assert r1.status_code == 200
        assert r1.json()["id"] == "resp_native"
        assert seen_paths[-1] == "/v1/responses"

    # 2) new_api -> Mistral model converts to /v1/chat/completions; non-Mistral stays native /v1/responses
    cfg_newapi = GatewayConfig(
        upstream_mode="new_api",
        upstream_base="http://new-api:3000",
        passthrough_auth=True,
    )
    app2 = create_app(cfg_newapi, http_client=mock_client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app2), base_url="http://127.0.0.1") as client:
        r2 = await client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer sk-user"},
            json={"model": "zai-glm-5-3", "input": "hi"},
        )
        assert r2.status_code == 200
        assert seen_paths[-1] == "/v1/chat/completions"
        assert r2.json()["output"][0]["content"][0]["text"] == "converted ok"

        r3 = await client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer sk-user"},
            json={"model": "gpt-6-luna", "input": "hi"},
        )
        assert r3.status_code == 200
        assert seen_paths[-1] == "/v1/responses"

    await mock_client.aclose()


@pytest.mark.asyncio
async def test_web_ui_admin_config_hot_reload_and_mistral_account(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "mistral-large-latest"}, {"id": "zai-glm-5-3"}]})
        if request.url.path == "/v1/billing/subscription":
            return httpx.Response(200, json={"plan": "pro", "status": "active"})
        if request.url.path == "/v1/billing/budget":
            return httpx.Response(200, json={"remaining_usd": 42.5})
        return httpx.Response(404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cfg_file = tmp_path / "config.json"
    cfg = GatewayConfig(
        upstream_mode="direct_mistral",
        upstream_base="https://api.mistral.ai",
        admin_token="secret-admin",
        config_path=str(cfg_file),
    )
    app = create_app(cfg, http_client=mock_client)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # UI & healthz
        ui = await client.get("/ui")
        assert ui.status_code == 200
        assert "Mistral Relay" in ui.text
        hz = await client.get("/gw/healthz")
        assert hz.status_code == 200
        assert hz.json()["ok"] is True

        # Admin auth enforced when ADMIN_TOKEN is set
        denied = await client.get("/api/admin/config")
        assert denied.status_code == 403

        hdr = {"X-Admin-Token": "secret-admin"}
        upd = await client.put(
            "/api/admin/config",
            headers=hdr,
            json={
                "upstream_api_key": "sk-mistral-real",
                "keep_images_mistral": 2,
                "model_concurrency_limits": "zai-glm-5-3=6",
            },
        )
        assert upd.status_code == 200
        body = upd.json()
        assert body["saved"] is True
        assert body["config"]["upstream_api_key"] == "••••••••"
        assert body["config"]["has_upstream_api_key"] is True
        assert body["config"]["keep_images_mistral"] == 2
        assert cfg_file.is_file()

        # Re-read status confirms limiter hot-reconfigured
        st = await client.get("/api/admin/status", headers=hdr)
        assert st.json()["concurrency"]["model_limits"] == {"zai-glm-5-3": 6}

        # Probe Mistral OpenAPI 3.1.0 account endpoints
        acct = await client.get("/api/admin/mistral-account", headers=hdr)
        assert acct.status_code == 200
        ad = acct.json()
        assert ad["models_count"] == 2
        assert ad["subscription"]["data"]["plan"] == "pro"
        assert ad["budget"]["data"]["remaining_usd"] == 42.5

    await mock_client.aclose()


@pytest.mark.asyncio
async def test_mistral_openai_compat_split_routing():
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path == "/openai/v1/chat/completions":
            body = json.loads(request.content)
            assert body["messages"][0]["role"] == "developer"
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "compat ok"}}]})
        if request.url.path == "/v1/responses":
            return httpx.Response(200, json={"id": "resp_1", "object": "response", "output": []})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "mistral-large-latest"}]})
        return httpx.Response(404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cfg = GatewayConfig(
        upstream_mode="direct_mistral",
        passthrough_auth=False,
        upstream_api_key="k",
    )
    # Pasting /openai/v1 into upstream_base should normalize base & enable openai_compat chat endpoint
    cfg.apply_patch({"upstream_base": "https://api.mistral.ai/openai/v1"})
    assert cfg.upstream_base == "https://api.mistral.ai"
    assert cfg.mistral_chat_endpoint == "openai_compat"

    app = create_app(cfg, http_client=mock_client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        r_chat = await client.post(
            "/v1/chat/completions",
            json={"model": "mistral-large-latest", "messages": [{"role": "developer", "content": "sys"}, {"role": "user", "content": "hi"}]},
        )
        assert r_chat.status_code == 200
        assert r_chat.json()["choices"][0]["message"]["content"] == "compat ok"

        r_resp = await client.post("/v1/responses", json={"model": "mistral-large-latest", "input": "hi"})
        assert r_resp.status_code == 200

        r_models = await client.get("/v1/models")
        assert r_models.status_code == 200

    assert seen == [
        ("POST", "/openai/v1/chat/completions"),
        ("POST", "/v1/responses"),
        ("GET", "/v1/models"),
    ]
    await mock_client.aclose()


@pytest.mark.asyncio
async def test_native_responses_zcode_input_type_and_reasoning_sse():
    captured_body: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/responses":
            captured_body.append(json.loads(request.content))
            sse = (
                'event: response.content_part.added\n'
                'data: {"type":"response.content_part.added","item_id":"rs_1","output_index":0,"content_index":0,"part":{"type":"reasoning_text","text":""}}\n\n'
                'event: response.reasoning_text.delta\n'
                'data: {"type":"response.reasoning_text.delta","item_id":"rs_1","output_index":0,"content_index":0,"delta":"thinking..."}\n\n'
                'event: response.output_text.delta\n'
                'data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":1,"content_index":0,"delta":"done"}\n\n'
            ).encode()
            return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cfg = GatewayConfig(
        upstream_mode="direct_mistral",
        upstream_base="https://api.mistral.ai",
        passthrough_auth=False,
        upstream_api_key="k",
        rewrite_responses_reasoning=True,
    )
    app = create_app(cfg, http_client=mock_client)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        r = await client.post(
            "/v1/responses",
            json={
                "model": "mistral-medium-latest",
                "stream": True,
                "input": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
                ],
            },
        )
        assert r.status_code == 200
        text = r.text
        assert "event: response.reasoning_summary_part.added" in text
        assert "event: response.reasoning_summary_text.delta" in text
        assert '"summary_index":0' in text
        assert '"delta":"thinking..."' in text
        assert "event: response.output_text.delta" in text
        assert "reasoning_text" not in text

        # Verify upstream received injected "type": "message"
        assert captured_body[0]["input"][0]["type"] == "message"
        assert captured_body[0]["input"][1]["type"] == "message"

        # Verify telemetry recorded normalized_chunks and stripped_fields
        logs = await client.get("/api/admin/logs", headers={"X-MRG-Console": "1"})
        ev = logs.json()["events"][0]
        assert ev["normalized_chunks"] == 2
        assert "input[].type=message" in ev["stripped_fields"]

    await mock_client.aclose()
