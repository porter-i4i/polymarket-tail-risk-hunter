"""
Adverse Selection Detector: identifies toxic flow conditions in order books and price history.

Uses 4 weighted signals:
  - volume spike     (0.35)
  - book imbalance   (0.30)
  - price momentum   (0.20)
  - spread compress  (0.15)

Composite >= TOXICITY_THRESHOLD => should_skip.
"""

import logging
from modules.types import OrderBook, PricePoint, FlowToxicity

logger = logging.getLogger(__name__)

# Default threshold — overridable via config
_DEFAULT_TOXICITY_THRESHOLD = 0.50

# Signal weights
_WEIGHT_VOLUME_SPIKE = 0.35
_WEIGHT_BOOK_IMBALANCE = 0.30
_WEIGHT_PRICE_MOMENTUM = 0.20
_WEIGHT_SPREAD_COMPRESSION = 0.15


class AdverseSelectionDetector:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.toxicity_threshold = self.cfg.get("TOXICITY_THRESHOLD", _DEFAULT_TOXICITY_THRESHOLD)
        self.volume_spike_threshold = self.cfg.get("VOLUME_SPIKE_THRESHOLD", 3.0)

    def assess_toxicity(
        self,
        market: dict,
        ob: OrderBook,
        price_history: list[PricePoint],
    ) -> FlowToxicity:
        """Compute composite toxicity score from 4 signals."""
        vol_score = self._volume_spike_score(price_history)
        imb_score = self._book_imbalance_score(ob)
        mom_score = self._price_momentum_score(price_history)
        spr_score = self._spread_compression_score(ob)

        composite = (
            _WEIGHT_VOLUME_SPIKE * vol_score
            + _WEIGHT_BOOK_IMBALANCE * imb_score
            + _WEIGHT_PRICE_MOMENTUM * mom_score
            + _WEIGHT_SPREAD_COMPRESSION * spr_score
        )

        signals: list[str] = []
        if vol_score >= 0.5:
            signals.append("volume_spike")
        if imb_score >= 0.5:
            signals.append("sell_imbalance" if self._asks_dominate(ob) else "buy_imbalance")
        if mom_score >= 0.5:
            signals.append("price_momentum")
        if spr_score >= 0.5:
            signals.append("spread_compression")

        should_skip = composite >= self.toxicity_threshold
        reason = ", ".join(signals) if signals else "Clean flow"

        return FlowToxicity(
            score=round(composite, 4),
            reason=reason,
            should_skip=should_skip,
            signals=signals,
        )

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------

    def _volume_spike_score(self, history: list[PricePoint]) -> float:
        """Score 0-1 based on how much recent volume exceeds the rolling average."""
        if len(history) < 2:
            return 0.0

        volumes = [p.volume for p in history]
        recent = volumes[-1]
        avg = sum(volumes[:-1]) / len(volumes[:-1])

        if avg <= 0:
            return 1.0 if recent > 0 else 0.0

        ratio = recent / avg
        # Map ratio to 0-1 score:
        #   ratio <= 1   -> 0
        #   ratio == spike_threshold -> 0.5
        #   ratio >= 2*spike_threshold -> 1.0
        if ratio <= 1.0:
            return 0.0
        elif ratio >= 2 * self.volume_spike_threshold:
            return 1.0
        else:
            return min(1.0, (ratio - 1.0) / (2 * self.volume_spike_threshold - 1.0))

    def _book_imbalance_score(self, ob: OrderBook) -> float:
        """Score 0-1 based on bid/ask volume imbalance."""
        bid_vol = sum(size for _, size in ob.bids) if ob.bids else 0.0
        ask_vol = sum(size for _, size in ob.asks) if ob.asks else 0.0
        total = bid_vol + ask_vol

        if total == 0:
            return 0.5  # no info

        imbalance = abs(bid_vol - ask_vol) / total  # 0-1
        # Map: imbalance 0 -> 0, imbalance 0.7 -> 0.5, imbalance 1 -> 1
        return min(1.0, imbalance / 0.7) * 0.5 + (max(0.0, imbalance - 0.7) / 0.3) * 0.5

    def _price_momentum_score(self, history: list[PricePoint]) -> float:
        """Score 0-1 based on recent directional price movement."""
        if len(history) < 3:
            return 0.0

        prices = [p.price for p in history]
        recent = prices[-3:]
        # Compute simple slope over last 3 points
        avg_price = sum(prices) / len(prices)
        if avg_price == 0:
            return 0.0

        move = abs(recent[-1] - recent[0]) / avg_price
        # Moves > 20% of avg price score high
        return min(1.0, move / 0.20)

    def _spread_compression_score(self, ob: OrderBook) -> float:
        """Score 0-1: tight spread relative to midpoint suggests informed trading."""
        if ob.midpoint <= 0 or ob.spread < 0:
            return 0.0

        spread_pct = ob.spread / ob.midpoint
        # Very tight spread (< 1%) = suspicious = high score
        # Normal spread (> 5%) = low score
        if spread_pct >= 0.05:
            return 0.0
        elif spread_pct <= 0.005:
            return 1.0
        else:
            return 1.0 - (spread_pct - 0.005) / (0.05 - 0.005)

    def _asks_dominate(self, ob: OrderBook) -> bool:
        """Helper: True if ask volume exceeds bid volume."""
        bid_vol = sum(size for _, size in ob.bids) if ob.bids else 0.0
        ask_vol = sum(size for _, size in ob.asks) if ob.asks else 0.0
        return ask_vol > bid_vol
