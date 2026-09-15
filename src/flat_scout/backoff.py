"""One retry policy for every model call.

A corpus-wide run is a few hundred calls fired a handful at a time, and a
provider answers some fraction of any burst with a 429 or a 5xx. Without this,
one such answer costs a whole Criterion - `by_model` treats any exception as
"the model could not be reached" - and the coverage figure quietly drops on
Listings that were fine.

Exponential backoff with full jitter: the n-th wait is uniform on
[0, min(cap, base * 2**n)]. Jitter matters more than the curve here, because
the calls that hit a limit hit it together, and a fixed schedule would send
them all back together too. A `Retry-After` header wins over the curve when
the provider sends one, capped the same way so a spiteful value cannot park
the run.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from pydantic_ai.exceptions import ModelHTTPError

log = logging.getLogger(__name__)

T = TypeVar("T")

# What is worth asking again. A 4xx outside this set is the request's fault
# and will be the same 4xx next time.
RETRYABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Module attributes rather than default arguments, so a test of a caller that
# never passes them can still pin the schedule with one monkeypatch.
_sleep = asyncio.sleep
_rng = random.random


def retryable(exc: BaseException) -> bool:
    if isinstance(exc, ModelHTTPError):
        return exc.status_code in RETRYABLE_STATUSES
    return isinstance(exc, (httpx.TransportError, asyncio.TimeoutError))


def _retry_after(exc: BaseException) -> float | None:
    if not isinstance(exc, ModelHTTPError) or not exc.headers:
        return None
    value = exc.headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        # An HTTP-date. Not worth parsing: the curve is a fine answer.
        return None


async def with_backoff(
    call: Callable[[], Awaitable[T]],
    *,
    attempts: int = 6,
    base: float = 1.0,
    cap: float = 30.0,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    rng: Callable[[], float] | None = None,
) -> T:
    """Await `call()` until it returns, retrying what is worth retrying.

    `sleep` and `rng` are parameters so a test can read the schedule back in
    milliseconds; production never passes them.
    """
    sleep = sleep or _sleep
    rng = rng or _rng
    for attempt in range(attempts):
        try:
            return await call()
        except Exception as exc:  # noqa: BLE001 - classified below
            last = attempt == attempts - 1
            if last or not retryable(exc):
                raise
            named = _retry_after(exc)
            if named is not None:
                delay = min(cap, named)
            else:
                delay = rng() * min(cap, base * 2**attempt)
            log.warning(
                "model call failed (%s), attempt %d of %d, waiting %.1fs",
                exc, attempt + 1, attempts, delay,
            )
            await sleep(delay)
    raise AssertionError("unreachable: the loop returns or raises")
