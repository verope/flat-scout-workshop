"""`with_backoff` is what stands between a corpus-wide run and one 429.

Every retry sleeps through the `sleep` it is handed, so these tests run in
milliseconds and can read the delays back.
"""

from __future__ import annotations

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError

from flat_scout.backoff import with_backoff


def http_error(status: int, headers: dict[str, str] | None = None) -> ModelHTTPError:
    return ModelHTTPError(status_code=status, model_name="m", body=None, headers=headers)


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def failing_then(succeed_after: int, exc: Exception):
    calls = {"n": 0}

    async def call():
        calls["n"] += 1
        if calls["n"] <= succeed_after:
            raise exc
        return "ok"

    return call, calls


@pytest.mark.asyncio
async def test_a_rate_limit_is_retried_until_it_clears():
    call, calls = failing_then(2, http_error(429))
    assert await with_backoff(call, sleep=Sleeps(), rng=lambda: 1.0) == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_the_delay_grows_and_is_jittered():
    """Full jitter: each wait is uniform on [0, min(cap, base * 2**n)]."""
    call, _ = failing_then(3, http_error(503))
    sleeps = Sleeps()
    await with_backoff(call, sleep=sleeps, rng=lambda: 0.5, base=1.0, cap=30.0)
    assert sleeps.delays == [0.5, 1.0, 2.0]


@pytest.mark.asyncio
async def test_the_delay_is_capped():
    call, _ = failing_then(5, http_error(502))
    sleeps = Sleeps()
    await with_backoff(call, sleep=sleeps, rng=lambda: 1.0, base=1.0, cap=4.0, attempts=6)
    assert sleeps.delays == [1.0, 2.0, 4.0, 4.0, 4.0]


@pytest.mark.asyncio
async def test_it_gives_up_after_the_last_attempt():
    call, calls = failing_then(10, http_error(429))
    with pytest.raises(ModelHTTPError):
        await with_backoff(call, sleep=Sleeps(), rng=lambda: 0.0, attempts=3)
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_a_client_error_is_not_retried():
    """A 400 will be a 400 again; retrying it only spends the budget."""
    call, calls = failing_then(10, http_error(400))
    with pytest.raises(ModelHTTPError):
        await with_backoff(call, sleep=Sleeps(), rng=lambda: 0.0)
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_a_transport_error_is_retried():
    call, calls = failing_then(1, httpx.ReadTimeout("slow"))
    assert await with_backoff(call, sleep=Sleeps(), rng=lambda: 0.0) == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_retry_after_is_honoured_when_the_provider_names_a_wait():
    call, _ = failing_then(1, http_error(429, {"retry-after": "7"}))
    sleeps = Sleeps()
    await with_backoff(call, sleep=sleeps, rng=lambda: 0.0, base=1.0)
    assert sleeps.delays == [7.0]


@pytest.mark.asyncio
async def test_retry_after_is_still_capped():
    call, _ = failing_then(1, http_error(429, {"retry-after": "600"}))
    sleeps = Sleeps()
    await with_backoff(call, sleep=sleeps, rng=lambda: 0.0, cap=30.0)
    assert sleeps.delays == [30.0]
