"""
Tail Event Debiaser: shrinks AI tail probabilities toward market price
using explicit price-bucket shrinkage factors.

Formula: adjusted = market + (ai - market) * factor
Lower market price => more shrinkage (lower factor).
"""

# Price-bucket shrinkage table: (lower_bound, upper_bound) -> factor
_SHRINKAGE_TABLE: list[tuple[tuple[float, float], float]] = [
    ((0.000, 0.010), 0.40),
    ((0.010, 0.020), 0.50),
    ((0.020, 0.030), 0.60),
    ((0.030, 0.050), 0.70),
    ((0.050, 0.100), 0.80),
    ((0.100, 1.000), 0.90),
]


def _get_shrinkage_factor(market_price: float) -> float:
    """Return shrinkage factor for the given market price bucket."""
    for (lo, hi), factor in _SHRINKAGE_TABLE:
        if lo <= market_price < hi:
            return factor
    # market_price >= 1.0 edge case — use highest bucket
    return 0.90


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class TailEventDebiaser:
    """Shrinks AI tail probabilities toward market price."""

    @staticmethod
    def debias(raw_probability: float, market_price: float) -> float:
        """
        Debias a single probability.
        adjusted = market + (ai - market) * factor
        """
        factor = _get_shrinkage_factor(market_price)
        adjusted = market_price + (raw_probability - market_price) * factor
        return _clamp(adjusted)

    @staticmethod
    def debias_distribution(
        p10: float, p50: float, p90: float, market_price: float
    ) -> dict[str, float]:
        """
        Debias a full distribution (p10, p50, p90).
        Preserves monotonicity: p10 <= p50 <= p90 after debiasing.
        """
        d10 = TailEventDebiaser.debias(p10, market_price)
        d50 = TailEventDebiaser.debias(p50, market_price)
        d90 = TailEventDebiaser.debias(p90, market_price)

        # Enforce monotonicity
        d50 = max(d50, d10)
        d90 = max(d90, d50)

        return {
            "p10": _clamp(d10),
            "p50": _clamp(d50),
            "p90": _clamp(d90),
        }
