import pytest

from mistral_relay_gateway.limiter import ModelLimiter


@pytest.mark.asyncio
async def test_model_limiter_acquire_timeout_and_reconfigure():
    limiter = ModelLimiter({"gemini-4-carbon-high": 1}, wait_seconds=0.05)

    # Unconfigured model passes straight through
    key, cap, sem = await limiter.acquire_slot("zai-glm-5-3")
    assert key is None and cap is None and sem is None

    # Configured model acquires slot
    k1, c1, s1 = await limiter.acquire_slot("gemini-4-carbon-high")
    assert k1 == "gemini-4-carbon-high" and c1 == 1 and s1 is not None
    assert limiter.inflight("gemini-4-carbon-high") == 1

    # Second concurrent request times out
    k2, c2, s2 = await limiter.acquire_slot("gemini-4-carbon-high")
    assert k2 == "gemini-4-carbon-high" and c2 is None and s2 is None
    assert limiter.stats()["total_rejected"] == 1

    # Release and hot-reconfigure cap to 2
    limiter.release_slot(k1, s1)
    assert limiter.inflight("gemini-4-carbon-high") == 0
    limiter.reconfigure({"gemini-4-carbon-high": 2}, wait_seconds=0.1)

    a = await limiter.acquire_slot("gemini-4-carbon-high")
    b = await limiter.acquire_slot("gemini-4-carbon-high")
    assert a[1] == 2 and b[1] == 2
    assert limiter.inflight("gemini-4-carbon-high") == 2
    limiter.release_slot(a[0], a[2])
    limiter.release_slot(b[0], b[2])
