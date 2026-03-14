"""Tests for modules/scoring_cache.py — adaptive TTL, eviction, cleanup, hit rate."""

import time

import pytest

from modules.scoring_cache import ScoringCache
from modules.types import AdjustedScore, RawAIScore, PricePoint


def _raw_score(**overrides) -> RawAIScore:
    defaults = dict(
        base_rate=5, p10=2, p50=5, p90=10, confidence=60,
        mispricing="underpriced", edge_pct=3.0, key_factor="test",
        reasoning="Test", evidence_for=["a"], evidence_against=["b"],
        recommendation="BUY_YES",
    )
    defaults.update(overrides)
    return RawAIScore(**defaults)


def _adjusted_score(**overrides) -> AdjustedScore:
    defaults = dict(
        raw=_raw_score(),
        adjusted_p10=0.02, adjusted_p50=0.05, adjusted_p90=0.10,
        adjusted_probability=0.05, expected_value=0.01,
        recommendation="BUY_YES", ci_width=0.08, debiased_p50=0.05,
    )
    defaults.update(overrides)
    return AdjustedScore(**defaults)


def _make_history(prices: list[float]) -> list[PricePoint]:
    return [
        PricePoint(timestamp=f"2026-03-13T{i:02d}:00:00Z", price=p, volume=100)
        for i, p in enumerate(prices)
    ]


# === GET / SET / MISS ===


class TestGetSet:
    def test_miss_returns_none(self):
        cache = ScoringCache()
        assert cache.get("unknown") is None

    def test_set_then_get(self):
        cache = ScoringCache()
        score = _adjusted_score()
        cache.set("abc", score)
        assert cache.get("abc") is score

    def test_expired_entry_returns_none(self):
        cache = ScoringCache(default_ttl=0)
        cache.set("abc", _adjusted_score())
        time.sleep(0.01)
        assert cache.get("abc") is None

    def test_custom_ttl(self):
        cache = ScoringCache(default_ttl=3600)
        score = _adjusted_score()
        # Manually inject an already-expired entry
        cache._cache["abc"] = (score, time.time() - 1)
        assert cache.get("abc") is None


# === HIT RATE ===


class TestHitRate:
    def test_zero_total(self):
        cache = ScoringCache()
        assert cache.hit_rate == 0.0

    def test_all_hits(self):
        cache = ScoringCache()
        cache.set("a", _adjusted_score())
        cache.get("a")
        cache.get("a")
        assert cache.hit_rate == pytest.approx(1.0)

    def test_mixed(self):
        cache = ScoringCache()
        cache.set("a", _adjusted_score())
        cache.get("a")   # hit
        cache.get("b")   # miss
        assert cache.hits == 1
        assert cache.misses == 1
        assert cache.hit_rate == pytest.approx(0.5)


# === EVICTION ===


class TestEviction:
    def test_evicts_at_capacity(self):
        cache = ScoringCache(max_entries=2)
        cache.set("a", _adjusted_score())
        cache.set("b", _adjusted_score())
        cache.set("c", _adjusted_score())
        assert cache.size == 2
        # "a" had earliest expiry → evicted
        assert cache.get("a") is None
        assert cache.get("c") is not None

    def test_evicts_oldest_expiry(self):
        cache = ScoringCache(max_entries=2)
        cache.set("short", _adjusted_score(), ttl=10)
        cache.set("long", _adjusted_score(), ttl=9999)
        # Add third — "short" has earlier expiry
        cache.set("new", _adjusted_score(), ttl=500)
        assert cache.get("short") is None
        assert cache.get("long") is not None


# === CLEANUP ===


class TestCleanup:
    def test_removes_expired(self):
        cache = ScoringCache(default_ttl=0)
        cache.set("a", _adjusted_score())
        cache.set("b", _adjusted_score())
        time.sleep(0.01)
        removed = cache.cleanup()
        assert removed == 2
        assert cache.size == 0

    def test_keeps_valid(self):
        cache = ScoringCache(default_ttl=3600)
        cache.set("a", _adjusted_score())
        removed = cache.cleanup()
        assert removed == 0
        assert cache.size == 1


# === ADAPTIVE TTL ===


class TestAdaptiveTTL:
    def test_no_history_returns_default(self):
        cache = ScoringCache(default_ttl=1800)
        assert cache.get_ttl({}, []) == 1800

    def test_single_point_returns_default(self):
        cache = ScoringCache(default_ttl=1800)
        hist = _make_history([0.03])
        assert cache.get_ttl({}, hist) == 1800

    def test_volatile_returns_300(self):
        hist = _make_history([0.02, 0.10, 0.04])  # max change 0.08 > 0.05
        cache = ScoringCache()
        assert cache.get_ttl({}, hist) == 300

    def test_moderate_returns_900(self):
        hist = _make_history([0.03, 0.06, 0.05])  # max change 0.03 > 0.02
        cache = ScoringCache()
        assert cache.get_ttl({}, hist) == 900

    def test_mild_returns_1800(self):
        hist = _make_history([0.03, 0.04, 0.035])  # max change 0.01 > 0.005
        cache = ScoringCache()
        assert cache.get_ttl({}, hist) == 1800

    def test_stable_returns_3600(self):
        hist = _make_history([0.03, 0.03, 0.031])  # max change 0.001 <= 0.005
        cache = ScoringCache()
        assert cache.get_ttl({}, hist) == 3600


# === SIZE PROPERTY ===


class TestSize:
    def test_empty(self):
        assert ScoringCache().size == 0

    def test_after_inserts(self):
        cache = ScoringCache()
        cache.set("a", _adjusted_score())
        cache.set("b", _adjusted_score())
        assert cache.size == 2
