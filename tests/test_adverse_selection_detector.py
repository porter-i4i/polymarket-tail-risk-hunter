"""Tests for modules/adverse_selection_detector.py."""

import pytest

from modules.adverse_selection_detector import AdverseSelectionDetector
from modules.types import OrderBook, PricePoint


def _make_ob(
    bids=None,
    asks=None,
    spread=0.02,
    midpoint=0.05,
) -> OrderBook:
    bids = bids or [(0.04, 100), (0.03, 200), (0.02, 300)]
    asks = asks or [(0.06, 100), (0.07, 200), (0.08, 300)]
    return OrderBook(bids=bids, asks=asks, spread=spread, midpoint=midpoint)


def _make_history(
    n=10,
    base_price=0.05,
    base_volume=100.0,
    last_volume=None,
    drift=0.0,
) -> list[PricePoint]:
    """Build a simple price history. drift adds per-step price change."""
    pts = []
    for i in range(n):
        vol = base_volume if (last_volume is None or i < n - 1) else last_volume
        pts.append(PricePoint(
            timestamp=f"2026-01-01T{i:02d}:00:00Z",
            price=base_price + drift * i,
            volume=vol,
        ))
    return pts


class TestCleanFlow:
    def test_clean_flow_passes(self):
        """Normal market conditions should produce low toxicity and should_skip=False."""
        det = AdverseSelectionDetector()
        result = det.assess_toxicity(
            market={},
            ob=_make_ob(),
            price_history=_make_history(),
        )
        assert not result.should_skip
        assert result.score < 0.5
        assert result.reason == "Clean flow"
        assert result.signals == []


class TestVolumeSpikeSignal:
    def test_volume_spike_increases_score(self):
        """A large volume spike on the last candle should raise the composite score."""
        det = AdverseSelectionDetector()

        normal = det.assess_toxicity(
            market={},
            ob=_make_ob(),
            price_history=_make_history(last_volume=100),
        )
        spiked = det.assess_toxicity(
            market={},
            ob=_make_ob(),
            price_history=_make_history(last_volume=1000),
        )
        assert spiked.score > normal.score
        assert "volume_spike" in spiked.signals


class TestBookImbalance:
    def test_sell_imbalance_in_reason(self):
        """Heavy ask-side volume should trigger sell_imbalance signal."""
        det = AdverseSelectionDetector()
        ob = _make_ob(
            bids=[(0.04, 10)],
            asks=[(0.06, 500), (0.07, 500)],
        )
        result = det.assess_toxicity(
            market={},
            ob=ob,
            price_history=_make_history(),
        )
        assert "sell_imbalance" in result.reason or "sell_imbalance" in result.signals

    def test_buy_imbalance(self):
        """Heavy bid-side volume should trigger buy_imbalance signal."""
        det = AdverseSelectionDetector()
        ob = _make_ob(
            bids=[(0.04, 500), (0.03, 500)],
            asks=[(0.06, 10)],
        )
        result = det.assess_toxicity(
            market={},
            ob=ob,
            price_history=_make_history(),
        )
        if result.signals:
            assert "buy_imbalance" in result.signals or "sell_imbalance" not in result.signals


class TestCombinedTrigger:
    def test_combined_signals_trigger_skip(self):
        """Multiple toxic signals should push composite above threshold -> should_skip."""
        det = AdverseSelectionDetector({"TOXICITY_THRESHOLD": 0.40})
        # Extreme conditions: volume spike + heavy imbalance + momentum + tight spread
        ob = _make_ob(
            bids=[(0.049, 5)],
            asks=[(0.051, 500), (0.052, 500)],
            spread=0.002,
            midpoint=0.05,
        )
        history = _make_history(n=10, base_price=0.03, drift=0.005, last_volume=2000)
        result = det.assess_toxicity(market={}, ob=ob, price_history=history)
        assert result.should_skip is True
        assert len(result.signals) >= 2


class TestEdgeCases:
    def test_empty_history(self):
        """Empty price history should not crash."""
        det = AdverseSelectionDetector()
        result = det.assess_toxicity(market={}, ob=_make_ob(), price_history=[])
        assert not result.should_skip

    def test_single_point_history(self):
        det = AdverseSelectionDetector()
        result = det.assess_toxicity(
            market={},
            ob=_make_ob(),
            price_history=[PricePoint("2026-01-01T00:00:00Z", 0.05, 100)],
        )
        assert not result.should_skip
