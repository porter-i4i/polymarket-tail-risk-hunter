"""
Async token bucket rate limiter.
Each instance tracks its own token count independently.
Tokens refill at a steady rate (max_per_minute / 60 per second).
If no tokens available, acquire() awaits until one refills.
"""
import asyncio
import time


class RateLimiter:
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.tokens = float(max_per_minute)
        self.max_tokens = float(max_per_minute)
        self.refill_rate = max_per_minute / 60.0  # tokens per second
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume one."""
        while True:
            async with self._lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
            # No token available — wait for partial refill
            wait_time = (1.0 - self.tokens) / self.refill_rate if self.refill_rate > 0 else 1.0
            await asyncio.sleep(min(wait_time, 1.0))

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)

    @property
    def available(self) -> float:
        """Current token count (approximate, for logging)."""
        return self.tokens
