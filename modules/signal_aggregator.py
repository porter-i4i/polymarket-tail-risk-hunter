"""
Signal Aggregator: Multi-source signal aggregation for market opportunities.

Combines five independent signals into a CompositeSignal:
- news (0.25): sentiment-based from recent articles
- momentum (0.20): price momentum from history
- volume (0.20): volume analysis relative to price tier
- calendar (0.20): time-based signals (market age, end date proximity)
- cross-market (0.15): order book spread/depth analysis

Each signal returns (score: float -1..+1, confidence: float 0..1).
"""
import logging
import math
from datetime import datetime, timezone
from typing import Optional

from modules.types import (
    CompositeSignal,
    MarketCandidate,
    NewsArticle,
    OrderBook,
    PricePoint,
)

logger = logging.getLogger(__name__)

# Signal weights (must sum to 1.0)
SIGNAL_WEIGHTS = {
    "news": 0.25,
    "momentum": 0.20,
    "volume": 0.20,
    "calendar": 0.20,
    "cross_market": 0.15,
}


class SignalAggregator:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def aggregate(
        self,
        market: MarketCandidate,
        news: list[NewsArticle],
        order_book: Optional[OrderBook],
        price_history: list[PricePoint],
    ) -> CompositeSignal:
        """
        Aggregate all five signals into a CompositeSignal.
        Each signal returns (score: -1..+1, confidence: 0..1).
        Composite = weighted sum of (score * confidence * weight).
        """
        signals: dict[str, tuple[float, float]] = {}

        signals["news"] = self._news_signal(news)
        signals["momentum"] = self._momentum_signal(price_history)
        signals["volume"] = self._volume_signal(market)
        signals["calendar"] = self._calendar_signal(market)
        signals["cross_market"] = self._cross_market_signal(order_book)

        # Weighted composite
        composite = 0.0
        for name, (score, confidence) in signals.items():
            weight = SIGNAL_WEIGHTS.get(name, 0.0)
            composite += score * confidence * weight

        # Clamp to [-1, 1]
        composite = max(-1.0, min(1.0, composite))

        return CompositeSignal(
            composite_score=round(composite, 4),
            should_score_with_ai=abs(composite) > 0.2,
            signals=signals,
        )

    @staticmethod
    def _news_signal(news: list[NewsArticle]) -> tuple[float, float]:
        """
        News signal: sentiment direction and freshness.
        Score: positive sentiment → +1, negative → -1, neutral → 0
        Confidence: based on article count and freshness.
        """
        if not news:
            return (0.0, 0.0)

        fresh = [a for a in news if a.age_hours <= 24]
        if not fresh:
            # Only stale news → low confidence
            fresh = news

        # Aggregate sentiment scores
        sentiment_sum = 0.0
        for article in fresh:
            if article.sentiment_label == "positive":
                sentiment_sum += article.sentiment_score
            elif article.sentiment_label == "negative":
                sentiment_sum -= article.sentiment_score
            # neutral contributes 0

        avg_sentiment = sentiment_sum / len(fresh) if fresh else 0.0
        score = max(-1.0, min(1.0, avg_sentiment))

        # Confidence: more articles + fresher = higher confidence
        article_conf = min(1.0, len(fresh) / 5.0)  # Saturates at 5 articles
        freshness_conf = 1.0 if any(a.age_hours <= 6 for a in fresh) else 0.5
        confidence = article_conf * freshness_conf

        return (round(score, 4), round(confidence, 4))

    @staticmethod
    def _momentum_signal(price_history: list[PricePoint]) -> tuple[float, float]:
        """
        Momentum signal: direction and magnitude of recent price changes.
        Score: price going up → positive, down → negative.
        Confidence: based on consistency and magnitude.
        """
        if not price_history or len(price_history) < 2:
            return (0.0, 0.0)

        recent = price_history[-5:]  # Last 5 data points
        prices = [p.price for p in recent]

        # Price change from oldest to newest
        total_change = prices[-1] - prices[0]

        # Direction consistency: count ups vs downs
        ups = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i - 1])
        downs = sum(1 for i in range(1, len(prices)) if prices[i] < prices[i - 1])
        total_moves = ups + downs

        if total_moves == 0:
            return (0.0, 0.0)

        # Score: normalized price change, capped at ±1
        # For tail events (price 0.01-0.05), a 1pp move is significant
        score = total_change * 20  # 5pp full-range → ±1
        score = max(-1.0, min(1.0, score))

        # Confidence: consistency of direction
        direction_ratio = max(ups, downs) / total_moves if total_moves > 0 else 0
        magnitude = abs(total_change)
        magnitude_conf = min(1.0, magnitude / 0.03)  # Saturates at 3pp change
        confidence = direction_ratio * magnitude_conf

        return (round(score, 4), round(confidence, 4))

    @staticmethod
    def _volume_signal(market: MarketCandidate) -> tuple[float, float]:
        """
        Volume signal: unusual volume relative to price tier.
        Score: high volume at low price → positive (interest signal).
        Confidence: based on absolute volume.
        """
        volume = market.volume_24h
        price = market.yes_price

        if volume <= 0:
            return (0.0, 0.0)

        # Expected volume for price tier
        # Sub-5% markets typically have low volume; high volume is noteworthy
        if price <= 0.05:
            expected_volume = 200
        elif price <= 0.10:
            expected_volume = 500
        else:
            expected_volume = 1000

        volume_ratio = volume / expected_volume
        # Score: above-average volume → positive signal
        score = max(-1.0, min(1.0, (volume_ratio - 1.0) * 0.5))

        # Confidence: scales with absolute volume
        confidence = min(1.0, volume / 1000.0)

        return (round(score, 4), round(confidence, 4))

    @staticmethod
    def _calendar_signal(market: MarketCandidate) -> tuple[float, float]:
        """
        Calendar signal: time-based factors.
        Score: new markets and approaching end dates → positive (action needed).
        Confidence: based on data availability.
        """
        score = 0.0
        confidence = 0.0

        # New market bonus
        if market.market_age_hours < 48:
            score += 0.5
            confidence = max(confidence, 0.6)

        # End date proximity
        if market.end_date:
            try:
                end_dt = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                hours_remaining = (end_dt - now).total_seconds() / 3600
                if 0 < hours_remaining <= 72:
                    # Approaching resolution → higher urgency signal
                    proximity = 1.0 - (hours_remaining / 72.0)
                    score += proximity * 0.5
                    confidence = max(confidence, 0.7)
                elif hours_remaining <= 0:
                    # Past end date → negative signal
                    score = -0.5
                    confidence = 0.8
            except (ValueError, TypeError):
                pass

        score = max(-1.0, min(1.0, score))
        if confidence == 0.0:
            confidence = 0.3  # Baseline confidence for calendar

        return (round(score, 4), round(confidence, 4))

    @staticmethod
    def _cross_market_signal(order_book: Optional[OrderBook]) -> tuple[float, float]:
        """
        Cross-market signal: order book structure analysis.
        Score: tight spread + good depth → positive, wide spread → negative.
        Confidence: based on book depth.
        """
        if order_book is None:
            return (0.0, 0.0)

        # Spread analysis
        spread = order_book.spread
        if spread <= 0.01:
            spread_score = 0.5  # Very tight spread → good
        elif spread <= 0.03:
            spread_score = 0.2
        elif spread <= 0.05:
            spread_score = 0.0
        else:
            spread_score = -0.5  # Wide spread → poor liquidity signal

        # Depth analysis
        bid_depth = sum(size for _, size in order_book.bids) if order_book.bids else 0
        ask_depth = sum(size for _, size in order_book.asks) if order_book.asks else 0
        total_depth = bid_depth + ask_depth

        if total_depth > 0 and bid_depth > 0 and ask_depth > 0:
            # Imbalance: more bids than asks → buying pressure → positive
            imbalance = (bid_depth - ask_depth) / total_depth
            depth_score = imbalance * 0.5
        else:
            depth_score = 0.0

        score = max(-1.0, min(1.0, spread_score + depth_score))

        # Confidence: based on total depth
        confidence = min(1.0, total_depth / 500.0) if total_depth > 0 else 0.0

        return (round(score, 4), round(confidence, 4))
