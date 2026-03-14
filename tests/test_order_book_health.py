"""Tests for modules/order_book_health.py."""

import pytest

from modules.order_book_health import OrderBookHealthChecker
from modules.types import OrderBook


def _healthy_ob() -> OrderBook:
    """A reasonably healthy order book."""
    bids = [
        (0.040, 1000),
        (0.039, 800),
        (0.038, 600),
        (0.037, 500),
        (0.036, 400),
        (0.035, 300),
    ]
    asks = [
        (0.042, 1000),
        (0.043, 800),
        (0.044, 600),
        (0.045, 500),
        (0.046, 400),
    ]
    return OrderBook(bids=bids, asks=asks, spread=0.002, midpoint=0.041)


class TestHealthyBook:
    def test_healthy_book_returns_is_healthy(self):
        checker = OrderBookHealthChecker()
        result = checker.assess(_healthy_ob(), entry_price=0.04)
        assert result.is_healthy is True
        assert result.score >= 0.5


class TestThinBook:
    def test_thin_book_unhealthy_with_depth_issue(self):
        """Very thin book should fail depth check."""
        ob = OrderBook(
            bids=[(0.04, 1), (0.03, 1)],
            asks=[(0.06, 1)],
            spread=0.02,
            midpoint=0.05,
        )
        checker = OrderBookHealthChecker({"MIN_BOOK_DEPTH": 200})
        result = checker.assess(ob, entry_price=0.04)
        assert result.is_healthy is False
        depth_issues = [i for i in result.issues if "depth" in i.lower() or "thin" in i.lower()]
        assert len(depth_issues) > 0


class TestWideSpread:
    def test_wide_spread_hurts_score(self):
        """Wide spread should reduce the score."""
        checker = OrderBookHealthChecker()
        narrow = OrderBook(
            bids=[(0.049, 1000)] * 5,
            asks=[(0.051, 1000)] * 5,
            spread=0.002,
            midpoint=0.05,
        )
        wide = OrderBook(
            bids=[(0.03, 1000)] * 5,
            asks=[(0.07, 1000)] * 5,
            spread=0.04,
            midpoint=0.05,
        )
        score_narrow = checker.assess(narrow, entry_price=0.05).score
        score_wide = checker.assess(wide, entry_price=0.05).score
        assert score_narrow > score_wide


class TestGapDetection:
    def test_gap_hurts_score(self):
        """Large gaps between bid levels should reduce health score."""
        checker = OrderBookHealthChecker()
        # No gaps
        no_gap_bids = [(0.050, 500), (0.049, 500), (0.048, 500), (0.047, 500), (0.046, 500)]
        no_gap_ob = OrderBook(bids=no_gap_bids, asks=[(0.052, 500)] * 5, spread=0.002, midpoint=0.051)

        # Big gap
        gap_bids = [(0.050, 500), (0.040, 500), (0.030, 500), (0.020, 500), (0.010, 500)]
        gap_ob = OrderBook(bids=gap_bids, asks=[(0.052, 500)] * 5, spread=0.002, midpoint=0.051)

        score_no_gap = checker.assess(no_gap_ob, entry_price=0.05).score
        score_gap = checker.assess(gap_ob, entry_price=0.05).score
        assert score_no_gap > score_gap


class TestInsufficientLevels:
    def test_insufficient_levels_hurts_score(self):
        """Fewer bid levels than MIN_LEVELS should degrade score."""
        checker = OrderBookHealthChecker()
        # Only 2 levels
        thin = OrderBook(
            bids=[(0.04, 5000), (0.039, 5000)],
            asks=[(0.042, 5000), (0.043, 5000)],
            spread=0.002,
            midpoint=0.041,
        )
        result = checker.assess(thin, entry_price=0.04)
        level_issues = [i for i in result.issues if "level" in i.lower()]
        assert len(level_issues) > 0

    def test_many_levels_no_level_issue(self):
        checker = OrderBookHealthChecker()
        result = checker.assess(_healthy_ob(), entry_price=0.04)
        level_issues = [i for i in result.issues if "level" in i.lower()]
        assert len(level_issues) == 0
