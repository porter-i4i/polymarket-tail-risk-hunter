"""
Fee Calculator: models Polymarket taker fees and computes fee-adjusted EV.

TAKER_FEE_BPS = 200  (2%)

For a YES buy at price p:
  cost     = p
  fee      = p * fee_rate
  net_cost = p + fee = p * (1 + fee_rate)
  payout   = 1.0 (if YES resolves)
  profit   = 1.0 - net_cost
  EV       = ai_prob * profit - (1 - ai_prob) * net_cost
"""

import logging

logger = logging.getLogger(__name__)

TAKER_FEE_BPS = 200


class FeeCalculator:
    def __init__(self, fee_bps: int = TAKER_FEE_BPS):
        self.fee_bps = fee_bps
        self.fee_rate = fee_bps / 10_000  # e.g. 0.02

    def calc_ev_with_fees(self, ai_probability: float, entry_price: float) -> float:
        """
        Expected value per dollar risked, accounting for taker fee.

        ai_probability: 0-1
        entry_price: 0-1 (market YES price)

        Returns EV (can be negative).
        """
        if entry_price <= 0 or entry_price >= 1:
            return -1.0

        net_cost = entry_price * (1 + self.fee_rate)
        profit_if_yes = 1.0 - net_cost
        loss_if_no = net_cost

        ev = ai_probability * profit_if_yes - (1 - ai_probability) * loss_if_no
        return round(ev, 6)

    def min_probability_for_positive_ev(self, entry_price: float) -> float:
        """
        Minimum AI probability needed for EV > 0 at given entry price.

        Derived from: p * (1 - net_cost) - (1 - p) * net_cost > 0
                   => p > net_cost
        """
        if entry_price <= 0 or entry_price >= 1:
            return 1.0  # impossible

        net_cost = entry_price * (1 + self.fee_rate)
        return min(1.0, net_cost)

    def min_bet_for_positive_ev(self, entry_price: float) -> float:
        """
        Minimum bet size (USD) for the fee structure to allow positive EV.
        On Polymarket the fee is proportional, so any positive-EV opportunity
        is positive at any size. Returns 0.50 as the practical minimum bet.
        """
        return 0.50

    def break_even_table(self) -> dict[float, float]:
        """
        Return {entry_price: min_probability} for common price points.
        """
        prices = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
        return {p: round(self.min_probability_for_positive_ev(p), 4) for p in prices}
