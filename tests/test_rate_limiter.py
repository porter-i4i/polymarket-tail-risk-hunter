"""Tests for modules/rate_limiter.py — Token bucket rate limiter."""

import asyncio
import time

import pytest

from modules.rate_limiter import RateLimiter


class TestRateLimiterInit:
    def test_initial_tokens_equal_max(self):
        rl = RateLimiter(max_per_minute=60)
        assert rl.tokens == 60.0
        assert rl.max_tokens == 60.0

    def test_refill_rate_calculated(self):
        rl = RateLimiter(max_per_minute=120)
        assert rl.refill_rate == 2.0  # 120 / 60

    def test_available_property(self):
        rl = RateLimiter(max_per_minute=30)
        assert rl.available == 30.0


class TestAcquire:
    @pytest.mark.asyncio
    async def test_single_acquire(self):
        rl = RateLimiter(max_per_minute=60)
        await rl.acquire()
        assert rl.tokens < 60.0

    @pytest.mark.asyncio
    async def test_multiple_acquires_decrement(self):
        rl = RateLimiter(max_per_minute=60)
        await rl.acquire()
        await rl.acquire()
        await rl.acquire()
        # Should have consumed ~3 tokens (minus any tiny refill during execution)
        assert rl.tokens < 58.0

    @pytest.mark.asyncio
    async def test_acquire_all_tokens(self):
        """Consuming all tokens should work without error."""
        rl = RateLimiter(max_per_minute=5)
        for _ in range(5):
            await rl.acquire()
        # After consuming all 5, tokens should be near 0 (plus tiny refill)
        assert rl.tokens < 1.0

    @pytest.mark.asyncio
    async def test_acquire_waits_when_empty(self):
        """When tokens are exhausted, acquire should wait for refill."""
        rl = RateLimiter(max_per_minute=600)  # 10 tokens/sec => fast refill
        # Drain all tokens
        rl.tokens = 0.0
        start = time.monotonic()
        await rl.acquire()
        elapsed = time.monotonic() - start
        # Should have waited some time for a token to refill
        assert elapsed > 0.05  # at least 50ms
        assert elapsed < 2.0   # but not too long


class TestRefill:
    @pytest.mark.asyncio
    async def test_tokens_refill_over_time(self):
        rl = RateLimiter(max_per_minute=600)  # 10 tokens/sec
        await rl.acquire()
        tokens_after_acquire = rl.tokens
        await asyncio.sleep(0.2)  # Wait 200ms => ~2 tokens refill
        # Trigger refill by acquiring
        async with rl._lock:
            rl._refill()
        assert rl.tokens > tokens_after_acquire

    def test_refill_does_not_exceed_max(self):
        rl = RateLimiter(max_per_minute=60)
        # Simulate time passing
        rl._last_refill = time.monotonic() - 120  # 2 minutes ago
        rl._refill()
        assert rl.tokens == rl.max_tokens

    def test_refill_adds_correct_amount(self):
        rl = RateLimiter(max_per_minute=60)  # 1 token/sec
        rl.tokens = 0.0
        rl._last_refill = time.monotonic() - 1.0  # 1 second ago
        rl._refill()
        # Should have refilled ~1 token
        assert 0.9 < rl.tokens < 1.2


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_acquires(self):
        """Multiple concurrent acquires should not over-consume."""
        rl = RateLimiter(max_per_minute=600)  # 10 tokens/sec
        initial = rl.tokens
        tasks = [rl.acquire() for _ in range(10)]
        await asyncio.gather(*tasks)
        # All 10 should have succeeded; tokens should be roughly initial - 10 + refill
        assert rl.tokens < initial
