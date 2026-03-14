"""Tests for modules/tail_event_debiaser.py."""

import pytest

from modules.tail_event_debiaser import TailEventDebiaser


class TestDebias:
    def test_lower_price_causes_more_shrinkage(self):
        """Lower market price bucket => smaller factor => more shrinkage toward market."""
        raw = 0.08
        # Market at 0.005 (bucket 0.40) should shrink more than market at 0.06 (bucket 0.80)
        adjusted_low = TailEventDebiaser.debias(raw, market_price=0.005)
        adjusted_high = TailEventDebiaser.debias(raw, market_price=0.06)
        # Both should be less than raw, but low-price result is closer to its market
        gap_low = abs(adjusted_low - 0.005)
        gap_high = abs(adjusted_high - 0.06)
        # Low-price gap should be smaller relative to the original gap
        shrinkage_low = gap_low / abs(raw - 0.005)
        shrinkage_high = gap_high / abs(raw - 0.06)
        assert shrinkage_low < shrinkage_high

    def test_debiased_does_not_exceed_raw_in_tail(self):
        """In typical tail scenarios (AI > market), debiased should not exceed raw."""
        raw = 0.04
        for market_price in [0.005, 0.015, 0.025, 0.04, 0.07]:
            adjusted = TailEventDebiaser.debias(raw, market_price)
            if raw >= market_price:
                assert adjusted <= raw + 1e-10, (
                    f"adjusted {adjusted} > raw {raw} at market {market_price}"
                )

    def test_identity_when_ai_equals_market(self):
        """If AI equals market, output should equal market."""
        for price in [0.005, 0.015, 0.025, 0.04, 0.07, 0.15]:
            result = TailEventDebiaser.debias(price, price)
            assert abs(result - price) < 1e-10

    def test_output_clamped(self):
        """Outputs should always be in [0.0, 1.0]."""
        assert TailEventDebiaser.debias(0.0, 0.0) >= 0.0
        assert TailEventDebiaser.debias(1.0, 1.0) <= 1.0
        assert TailEventDebiaser.debias(1.5, 0.5) <= 1.0
        assert TailEventDebiaser.debias(-0.1, 0.5) >= 0.0


class TestDebiasDistribution:
    def test_monotonicity_preserved(self):
        """After debiasing, p10 <= p50 <= p90 must hold."""
        result = TailEventDebiaser.debias_distribution(
            p10=0.01, p50=0.03, p90=0.08, market_price=0.02
        )
        assert result["p10"] <= result["p50"] <= result["p90"]

    def test_distribution_identity_at_market(self):
        """When all AI estimates equal market, output equals market."""
        result = TailEventDebiaser.debias_distribution(
            p10=0.03, p50=0.03, p90=0.03, market_price=0.03
        )
        assert abs(result["p10"] - 0.03) < 1e-10
        assert abs(result["p50"] - 0.03) < 1e-10
        assert abs(result["p90"] - 0.03) < 1e-10

    def test_distribution_all_clamped(self):
        """All distribution outputs should be in [0.0, 1.0]."""
        result = TailEventDebiaser.debias_distribution(
            p10=0.0, p50=0.5, p90=1.0, market_price=0.5
        )
        for key in ("p10", "p50", "p90"):
            assert 0.0 <= result[key] <= 1.0

    def test_monotonicity_with_close_values(self):
        """Test monotonicity when debiased values could potentially cross."""
        result = TailEventDebiaser.debias_distribution(
            p10=0.02, p50=0.021, p90=0.022, market_price=0.025
        )
        assert result["p10"] <= result["p50"] <= result["p90"]
