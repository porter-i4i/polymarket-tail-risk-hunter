"""
Order Book Health Checker: evaluates order book quality for safe entry.

5 weighted checks:
  - depth        0.30
  - spread       0.25
  - nearby_bids  0.20
  - no_gaps      0.15
  - levels       0.10

is_healthy if composite >= 0.5.
"""

import logging
from modules.types import OrderBook, BookHealth

logger = logging.getLogger(__name__)

# Defaults
_DEFAULT_MIN_DEPTH = 200.0     # USD total depth
_DEFAULT_MAX_SPREAD_PCT = 0.50  # 50% of midpoint
_DEFAULT_NEARBY_RANGE = 0.05    # within 5 cents of entry
_DEFAULT_NEARBY_MIN_DEPTH = 50.0  # USD nearby depth
_DEFAULT_MAX_GAP_PCT = 0.10     # 10% price gap between levels
_DEFAULT_MIN_LEVELS = 5         # minimum distinct price levels

# Check weights
_W_DEPTH = 0.30
_W_SPREAD = 0.25
_W_NEARBY = 0.20
_W_GAPS = 0.15
_W_LEVELS = 0.10


class OrderBookHealthChecker:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.min_depth = self.cfg.get("MIN_BOOK_DEPTH", _DEFAULT_MIN_DEPTH)
        self.max_spread_pct = self.cfg.get("MAX_SPREAD_PCT", _DEFAULT_MAX_SPREAD_PCT)

    def assess(self, ob: OrderBook, entry_price: float) -> BookHealth:
        """Run all 5 health checks and produce composite score."""
        issues: list[str] = []

        depth_score = self._check_depth(ob, issues)
        spread_score = self._check_spread(ob, issues)
        nearby_score = self._check_nearby_bids(ob, entry_price, issues)
        gap_score = self._check_gaps(ob, issues)
        level_score = self._check_levels(ob, issues)

        composite = (
            _W_DEPTH * depth_score
            + _W_SPREAD * spread_score
            + _W_NEARBY * nearby_score
            + _W_GAPS * gap_score
            + _W_LEVELS * level_score
        )

        return BookHealth(
            score=round(composite, 4),
            is_healthy=composite >= 0.5,
            issues=issues,
        )

    # ------------------------------------------------------------------
    # Individual checks — each returns 0-1 score and appends issues
    # ------------------------------------------------------------------

    def _check_depth(self, ob: OrderBook, issues: list[str]) -> float:
        """Total bid+ask depth in USD."""
        bid_depth = sum(p * s for p, s in ob.bids) if ob.bids else 0.0
        ask_depth = sum(p * s for p, s in ob.asks) if ob.asks else 0.0
        total = bid_depth + ask_depth

        if total >= self.min_depth:
            return 1.0
        if total <= 0:
            issues.append(f"No book depth (need ${self.min_depth:.0f})")
            return 0.0
        ratio = total / self.min_depth
        issues.append(f"Thin book: ${total:.0f} depth (need ${self.min_depth:.0f})")
        return ratio

    def _check_spread(self, ob: OrderBook, issues: list[str]) -> float:
        """Spread relative to midpoint."""
        if ob.midpoint <= 0:
            issues.append("No midpoint")
            return 0.0

        spread_pct = ob.spread / ob.midpoint
        if spread_pct <= 0.01:
            return 1.0  # very tight
        if spread_pct >= self.max_spread_pct:
            issues.append(f"Wide spread: {spread_pct:.1%} (max {self.max_spread_pct:.0%})")
            return 0.0

        # Linear interpolation: 1% -> 1.0, max_spread -> 0.0
        return 1.0 - (spread_pct - 0.01) / (self.max_spread_pct - 0.01)

    def _check_nearby_bids(self, ob: OrderBook, entry_price: float, issues: list[str]) -> float:
        """Bid depth within NEARBY_RANGE of entry price."""
        nearby_depth = 0.0
        for price, size in ob.bids:
            if abs(price - entry_price) <= _DEFAULT_NEARBY_RANGE:
                nearby_depth += price * size

        if nearby_depth >= _DEFAULT_NEARBY_MIN_DEPTH:
            return 1.0
        if nearby_depth <= 0:
            issues.append("No nearby bid support")
            return 0.0
        ratio = nearby_depth / _DEFAULT_NEARBY_MIN_DEPTH
        issues.append(f"Thin nearby bids: ${nearby_depth:.0f} (need ${_DEFAULT_NEARBY_MIN_DEPTH:.0f})")
        return ratio

    def _check_gaps(self, ob: OrderBook, issues: list[str]) -> float:
        """Detect large gaps between adjacent bid levels."""
        if len(ob.bids) < 2:
            issues.append("Insufficient bid levels for gap analysis")
            return 0.0

        prices = sorted([p for p, _ in ob.bids], reverse=True)
        max_gap = 0.0
        for i in range(len(prices) - 1):
            gap = prices[i] - prices[i + 1]
            gap_pct = gap / prices[i] if prices[i] > 0 else 0.0
            max_gap = max(max_gap, gap_pct)

        if max_gap <= 0.02:
            return 1.0
        if max_gap >= _DEFAULT_MAX_GAP_PCT:
            issues.append(f"Large price gap: {max_gap:.1%}")
            return 0.0
        return 1.0 - (max_gap - 0.02) / (_DEFAULT_MAX_GAP_PCT - 0.02)

    def _check_levels(self, ob: OrderBook, issues: list[str]) -> float:
        """Count distinct price levels on bid side."""
        n = len(ob.bids)
        if n >= _DEFAULT_MIN_LEVELS:
            return 1.0
        if n == 0:
            issues.append(f"No bid levels (need {_DEFAULT_MIN_LEVELS})")
            return 0.0
        ratio = n / _DEFAULT_MIN_LEVELS
        issues.append(f"Only {n} bid levels (need {_DEFAULT_MIN_LEVELS})")
        return ratio
