"""Tests for modules/signal_aggregator.py — 5-signal composite aggregation."""

import pytest

from modules.signal_aggregator import SignalAggregator, SIGNAL_WEIGHTS
from modules.types import MarketCandidate, NewsArticle, OrderBook, PricePoint, CompositeSignal


def _make_candidate(**overrides) -> MarketCandidate:
    defaults = dict(
        condition_id="abc123",
        question="Will X happen?",
        category="Politics",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        yes_price=0.03,
        no_price=0.97,
        volume_24h=500,
        liquidity=1000,
        end_date="",
        tags=["Politics"],
        market_age_hours=100.0,
    )
    defaults.update(overrides)
    return MarketCandidate(**defaults)


def _make_article(
    sentiment_label="positive",
    sentiment_score=0.8,
    age_hours=2.0,
    source="reuters",
) -> NewsArticle:
    return NewsArticle(
        title="Test",
        description="desc",
        source=source,
        published_at="2026-03-13T00:00:00Z",
        age_hours=age_hours,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
    )


def _make_order_book(bids=None, asks=None) -> OrderBook:
    bids = bids or [(0.03, 100), (0.02, 200)]
    asks = asks or [(0.04, 100), (0.05, 200)]
    return OrderBook(
        bids=bids,
        asks=asks,
        spread=asks[0][0] - bids[0][0],
        midpoint=(asks[0][0] + bids[0][0]) / 2,
    )


def _make_history(prices: list[float]) -> list[PricePoint]:
    return [
        PricePoint(timestamp=f"2026-03-13T{i:02d}:00:00Z", price=p, volume=100)
        for i, p in enumerate(prices)
    ]


# === SIGNAL WEIGHTS ===


class TestSignalWeights:
    def test_weights_sum_to_one(self):
        assert sum(SIGNAL_WEIGHTS.values()) == pytest.approx(1.0)

    def test_all_five_signals_present(self):
        assert set(SIGNAL_WEIGHTS.keys()) == {
            "news", "momentum", "volume", "calendar", "cross_market"
        }


# === NEWS SIGNAL ===


class TestNewsSignal:
    def test_no_news_returns_zero(self):
        score, conf = SignalAggregator._news_signal([])
        assert score == 0.0
        assert conf == 0.0

    def test_positive_news_positive_score(self):
        articles = [_make_article("positive", 0.9, age_hours=2.0)]
        score, conf = SignalAggregator._news_signal(articles)
        assert score > 0
        assert conf > 0

    def test_negative_news_negative_score(self):
        articles = [_make_article("negative", 0.8, age_hours=2.0)]
        score, conf = SignalAggregator._news_signal(articles)
        assert score < 0

    def test_neutral_news_zero_score(self):
        articles = [_make_article("neutral", 0.5, age_hours=2.0)]
        score, _ = SignalAggregator._news_signal(articles)
        assert score == 0.0

    def test_fresh_news_higher_confidence(self):
        fresh = [_make_article(age_hours=2.0)]
        stale = [_make_article(age_hours=30.0)]
        _, conf_fresh = SignalAggregator._news_signal(fresh)
        _, conf_stale = SignalAggregator._news_signal(stale)
        assert conf_fresh >= conf_stale

    def test_more_articles_higher_confidence(self):
        one = [_make_article()]
        five = [_make_article() for _ in range(5)]
        _, conf1 = SignalAggregator._news_signal(one)
        _, conf5 = SignalAggregator._news_signal(five)
        assert conf5 >= conf1


# === MOMENTUM SIGNAL ===


class TestMomentumSignal:
    def test_no_history_returns_zero(self):
        score, conf = SignalAggregator._momentum_signal([])
        assert score == 0.0

    def test_single_point_returns_zero(self):
        hist = _make_history([0.03])
        score, conf = SignalAggregator._momentum_signal(hist)
        assert score == 0.0

    def test_upward_momentum_positive(self):
        hist = _make_history([0.02, 0.03, 0.04, 0.05, 0.06])
        score, conf = SignalAggregator._momentum_signal(hist)
        assert score > 0

    def test_downward_momentum_negative(self):
        hist = _make_history([0.06, 0.05, 0.04, 0.03, 0.02])
        score, conf = SignalAggregator._momentum_signal(hist)
        assert score < 0

    def test_flat_prices_zero_score(self):
        hist = _make_history([0.03, 0.03, 0.03, 0.03])
        score, conf = SignalAggregator._momentum_signal(hist)
        assert score == 0.0
        assert conf == 0.0


# === VOLUME SIGNAL ===


class TestVolumeSignal:
    def test_zero_volume(self):
        m = _make_candidate(volume_24h=0)
        score, conf = SignalAggregator._volume_signal(m)
        assert score == 0.0
        assert conf == 0.0

    def test_high_volume_positive(self):
        m = _make_candidate(yes_price=0.03, volume_24h=1000)
        score, conf = SignalAggregator._volume_signal(m)
        assert score > 0  # 1000 / 200 expected = 5x → positive

    def test_low_volume_sub5pct(self):
        m = _make_candidate(yes_price=0.03, volume_24h=100)
        score, conf = SignalAggregator._volume_signal(m)
        assert score < 0  # below expected 200


# === CALENDAR SIGNAL ===


class TestCalendarSignal:
    def test_new_market_positive(self):
        m = _make_candidate(market_age_hours=24)
        score, conf = SignalAggregator._calendar_signal(m)
        assert score > 0
        assert conf >= 0.6

    def test_old_market_baseline(self):
        m = _make_candidate(market_age_hours=500, end_date="")
        score, conf = SignalAggregator._calendar_signal(m)
        assert score == 0.0
        assert conf == 0.3  # baseline

    def test_past_end_date_negative(self):
        m = _make_candidate(end_date="2020-01-01T00:00:00Z")
        score, conf = SignalAggregator._calendar_signal(m)
        assert score < 0


# === CROSS MARKET SIGNAL ===


class TestCrossMarketSignal:
    def test_none_order_book(self):
        score, conf = SignalAggregator._cross_market_signal(None)
        assert score == 0.0
        assert conf == 0.0

    def test_tight_spread_positive(self):
        ob = _make_order_book(
            bids=[(0.035, 100)],
            asks=[(0.04, 100)],
        )
        score, conf = SignalAggregator._cross_market_signal(ob)
        assert score >= 0  # spread 0.005 → tight

    def test_wide_spread_negative(self):
        ob = _make_order_book(
            bids=[(0.02, 50)],
            asks=[(0.10, 50)],
        )
        score, conf = SignalAggregator._cross_market_signal(ob)
        assert score < 0  # spread 0.08 → wide

    def test_bid_heavy_imbalance_positive(self):
        ob = _make_order_book(
            bids=[(0.03, 500)],
            asks=[(0.04, 50)],
        )
        score, conf = SignalAggregator._cross_market_signal(ob)
        # Tight-ish spread (0.01) + heavy bid imbalance → positive
        assert score > 0


# === FULL AGGREGATION ===


class TestAggregate:
    def test_all_neutral_near_zero(self):
        agg = SignalAggregator({})
        m = _make_candidate(market_age_hours=500, volume_24h=200)
        result = agg.aggregate(m, [], None, [])
        assert isinstance(result, CompositeSignal)
        assert -0.3 < result.composite_score < 0.3

    def test_should_score_threshold(self):
        agg = SignalAggregator({})
        m = _make_candidate(market_age_hours=500, volume_24h=200)
        result = agg.aggregate(m, [], None, [])
        # With neutral data, score should be small
        if abs(result.composite_score) <= 0.2:
            assert result.should_score_with_ai is False
        else:
            assert result.should_score_with_ai is True

    def test_positive_signals_increase_composite(self):
        agg = SignalAggregator({})
        m = _make_candidate(market_age_hours=24, volume_24h=1000)
        news = [_make_article("positive", 0.9, age_hours=1.0) for _ in range(5)]
        hist = _make_history([0.02, 0.03, 0.04, 0.05, 0.06])
        ob = _make_order_book()
        result = agg.aggregate(m, news, ob, hist)
        assert result.composite_score > 0.2

    def test_composite_clamped(self):
        agg = SignalAggregator({})
        m = _make_candidate()
        result = agg.aggregate(m, [], None, [])
        assert -1.0 <= result.composite_score <= 1.0

    def test_signals_dict_has_all_five(self):
        agg = SignalAggregator({})
        m = _make_candidate()
        result = agg.aggregate(m, [], None, [])
        assert set(result.signals.keys()) == {
            "news", "momentum", "volume", "calendar", "cross_market"
        }
