"""Token-bucket rate limiter for async LLM calls.

Usage::

    limiter = RateLimiter(calls_per_minute=30)
    await limiter.acquire()   # blocks until a token is available
    response = await llm.ainvoke(...)
"""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Async token-bucket rate limiter.

    Allows up to *calls_per_minute* calls per minute on average, with short
    bursts up to the full bucket capacity.  Calls beyond the rate block
    (``await limiter.acquire()``) until a token refills.

    Pass ``calls_per_minute=0`` to disable rate limiting entirely.
    """

    def __init__(self, calls_per_minute: int) -> None:
        if calls_per_minute < 0:
            raise ValueError("calls_per_minute must be >= 0")
        self.calls_per_minute = calls_per_minute
        if calls_per_minute == 0:
            # Disabled — acquire() returns immediately.
            self._disabled = True
            return

        self._disabled = False
        self._interval: float = 60.0 / calls_per_minute  # seconds per token
        self._capacity: float = float(calls_per_minute)
        self._tokens: float = float(calls_per_minute)     # start full
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until a token is available, then consume one token."""
        if self._disabled:
            return

        async with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            # Calculate exactly how long to wait for 1 token to refill.
            deficit = 1.0 - self._tokens
            wait_seconds = deficit * self._interval

        # Sleep *outside* the lock so other coroutines can proceed.
        await asyncio.sleep(wait_seconds)

        async with self._lock:
            self._refill()
            self._tokens = max(0.0, self._tokens - 1.0)

    def _refill(self) -> None:
        """Add tokens proportional to elapsed time (called under lock)."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed / self._interval)
        self._last_refill = now
