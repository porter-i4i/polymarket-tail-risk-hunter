"""Tests for modules/fee_calculator.py."""

import pytest

from modules.fee_calculator import FeeCalculator, TAKER_FEE_BPS


class TestEVWithFees:
    def test_ev_positive_when_ai_above_market(self):
        """High AI probability relative to market price => positive EV."""
        fc = FeeCalculator()
        # AI says 0.15, market at 0.03 => big edge
        ev = fc.calc_ev_with_fees(ai_probability=0.15, entry_price=0.03)
        assert ev > 0

    def test_ev_negative_when_no_edge(self):
        """AI probability at or below market price => negative EV."""
        fc = FeeCalculator()
        ev = fc.calc_ev_with_fees(ai_probability=0.03, entry_price=0.03)
        assert ev < 0

    def test_ev_negative_for_fair_price(self):
        """At fair value, fees make EV negative."""
        fc = FeeCalculator()
        ev = fc.calc_ev_with_fees(ai_probability=0.50, entry_price=0.50)
        assert ev < 0

    def test_ev_with_zero_fee(self):
        """With 0 fee, fair value should give ~0 EV."""
        fc = FeeCalculator(fee_bps=0)
        ev = fc.calc_ev_with_fees(ai_probability=0.50, entry_price=0.50)
        assert abs(ev) < 0.001

    def test_invalid_entry_price(self):
        fc = FeeCalculator()
        assert fc.calc_ev_with_fees(0.5, 0.0) == -1.0
        assert fc.calc_ev_with_fees(0.5, 1.0) == -1.0


class TestMinProbability:
    def test_monotonic_with_price(self):
        """Higher entry price should require higher min probability."""
        fc = FeeCalculator()
        prices = [0.01, 0.03, 0.05, 0.10, 0.20, 0.50]
        min_probs = [fc.min_probability_for_positive_ev(p) for p in prices]
        for i in range(len(min_probs) - 1):
            assert min_probs[i] <= min_probs[i + 1]

    def test_min_prob_exceeds_price(self):
        """Min probability should exceed entry price due to fees."""
        fc = FeeCalculator()
        for p in [0.03, 0.05, 0.10]:
            min_p = fc.min_probability_for_positive_ev(p)
            assert min_p > p

    def test_invalid_entry_returns_one(self):
        fc = FeeCalculator()
        assert fc.min_probability_for_positive_ev(0.0) == 1.0
        assert fc.min_probability_for_positive_ev(1.0) == 1.0


class TestBreakEvenTable:
    def test_returns_expected_keys(self):
        fc = FeeCalculator()
        table = fc.break_even_table()
        assert 0.01 in table
        assert 0.05 in table
        assert 0.10 in table
        assert 0.50 in table

    def test_approximate_values(self):
        """Break-even probabilities should be entry_price * (1 + fee_rate)."""
        fc = FeeCalculator()
        table = fc.break_even_table()
        fee_rate = TAKER_FEE_BPS / 10_000
        for price, min_prob in table.items():
            expected = price * (1 + fee_rate)
            assert abs(min_prob - expected) < 0.01, f"At price {price}: {min_prob} vs {expected}"

    def test_table_values_monotonic(self):
        fc = FeeCalculator()
        table = fc.break_even_table()
        sorted_prices = sorted(table.keys())
        for i in range(len(sorted_prices) - 1):
            assert table[sorted_prices[i]] <= table[sorted_prices[i + 1]]


class TestMinBet:
    def test_min_bet_is_practical_minimum(self):
        fc = FeeCalculator()
        assert fc.min_bet_for_positive_ev(0.05) == 0.50


class TestConstants:
    def test_taker_fee_bps(self):
        assert TAKER_FEE_BPS == 200
