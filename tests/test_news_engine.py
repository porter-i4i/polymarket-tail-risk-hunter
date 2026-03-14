"""Tests for modules/news_engine.py — NewsData.io + FinBERT sentiment + pre-filter."""

import pytest

from modules.news_engine import NewsEngine
from modules.rate_limiter import RateLimiter
from modules.types import NewsArticle, PricePoint


def _make_article(
    title="Test",
    source="reuters",
    age_hours=2.0,
    sentiment_label="neutral",
    sentiment_score=0.0,
) -> NewsArticle:
    return NewsArticle(
        title=title,
        description="desc",
        source=source,
        published_at="2026-03-13T00:00:00Z",
        age_hours=age_hours,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
    )


def _make_engine(api_key="") -> NewsEngine:
    limiter = RateLimiter(max_per_minute=60)
    cfg = {"NEWS_API_KEY": api_key}
    engine = NewsEngine(limiter, cfg)
    # Force FinBERT off for predictable test behaviour
    engine._finbert_available = False
    engine._finbert_pipeline = None
    return engine


# === EXTRACT QUERY ===


class TestExtractQuery:
    def test_removes_will_prefix(self):
        assert NewsEngine._extract_query("Will Biden resign?") == "Biden resign"

    def test_removes_is_prefix(self):
        assert NewsEngine._extract_query("Is inflation above 5%?") == "inflation above 5%"

    def test_no_prefix(self):
        assert NewsEngine._extract_query("Bitcoin hits $100k") == "Bitcoin hits $100k"

    def test_truncates_long_query(self):
        long = "A" * 200
        assert len(NewsEngine._extract_query(long)) == 80


# === AGGREGATE SENTIMENT ===


class TestAggregateSentiment:
    def test_empty_news(self):
        engine = _make_engine()
        result = engine.aggregate_sentiment([])
        assert result["article_count"] == 0
        assert result["dominant_label"] == "neutral"

    def test_single_positive(self):
        engine = _make_engine()
        articles = [_make_article(sentiment_label="positive", sentiment_score=0.9)]
        result = engine.aggregate_sentiment(articles)
        assert result["dominant_label"] == "positive"
        assert result["article_count"] == 1
        assert result["avg_score"] == pytest.approx(0.9, abs=0.01)

    def test_mixed_sentiment_dominant(self):
        engine = _make_engine()
        articles = [
            _make_article(sentiment_label="positive", sentiment_score=0.8),
            _make_article(sentiment_label="negative", sentiment_score=0.6),
            _make_article(sentiment_label="positive", sentiment_score=0.7),
        ]
        result = engine.aggregate_sentiment(articles)
        assert result["dominant_label"] == "positive"
        assert result["article_count"] == 3

    def test_fresh_count(self):
        engine = _make_engine()
        articles = [
            _make_article(age_hours=2.0),
            _make_article(age_hours=48.0),
            _make_article(age_hours=12.0),
        ]
        result = engine.aggregate_sentiment(articles)
        assert result["fresh_count"] == 2  # 2 and 12 are <= 24


# === SENTIMENT SCORING ===


class TestScoreSentiment:
    def test_finbert_unavailable_returns_neutral(self):
        engine = _make_engine()
        label, score = engine._score_sentiment("This is good news")
        assert label == "neutral"
        assert score == 0.0


# === PRE-FILTER ===


class TestShouldCallAI:
    def _market(self, **overrides):
        base = {
            "yes_price": 0.03,
            "no_price": 0.97,
            "volume_24h": 200,
            "market_age_hours": 200,
        }
        base.update(overrides)
        return base

    def test_rule1_fresh_news(self):
        engine = _make_engine()
        news = [_make_article(age_hours=6.0)]
        assert engine.should_call_ai(self._market(), news, {}, []) is True

    def test_rule2_strong_sentiment_finbert_off(self):
        """When FinBERT unavailable and news exists, pass by default."""
        engine = _make_engine()
        engine._finbert_available = False
        news = [_make_article(age_hours=30.0)]  # Not fresh (>24h)
        assert engine.should_call_ai(self._market(), news, {"avg_score": 0.9}, []) is True

    def test_rule2_strong_sentiment_finbert_on(self):
        """When FinBERT available, need avg_score > 0.6."""
        engine = _make_engine()
        engine._finbert_available = True
        news = [_make_article(age_hours=30.0)]
        sentiment = {"avg_score": 0.8}
        assert engine.should_call_ai(self._market(), news, sentiment, []) is True

    def test_rule2_weak_sentiment_finbert_on(self):
        """When FinBERT available and score < 0.6, this rule doesn't trigger."""
        engine = _make_engine()
        engine._finbert_available = True
        # No fresh news, weak sentiment, high market age, low volume → all rules fail
        assert engine.should_call_ai(
            self._market(volume_24h=100, market_age_hours=200),
            [_make_article(age_hours=30.0)],
            {"avg_score": 0.3},
            [],
        ) is False

    def test_rule3_price_momentum(self):
        engine = _make_engine()
        hist = [
            PricePoint("2026-03-13T00:00:00Z", 0.02, 100),
            PricePoint("2026-03-13T01:00:00Z", 0.04, 200),
            PricePoint("2026-03-13T02:00:00Z", 0.05, 150),
        ]
        assert engine.should_call_ai(
            self._market(volume_24h=100, market_age_hours=200),
            [], {"avg_score": 0.0}, hist
        ) is True

    def test_rule4_high_volume(self):
        engine = _make_engine()
        engine._finbert_available = True
        mkt = self._market(yes_price=0.03, volume_24h=600, market_age_hours=200)
        assert engine.should_call_ai(mkt, [], {"avg_score": 0.0}, []) is True

    def test_rule5_new_market(self):
        engine = _make_engine()
        engine._finbert_available = True
        mkt = self._market(market_age_hours=24, volume_24h=100)
        assert engine.should_call_ai(mkt, [], {"avg_score": 0.0}, []) is True

    def test_rule6_multiple_sources(self):
        engine = _make_engine()
        engine._finbert_available = True
        news = [
            _make_article(source="reuters", age_hours=30.0),
            _make_article(source="bbc", age_hours=30.0),
            _make_article(source="cnn", age_hours=30.0),
        ]
        assert engine.should_call_ai(
            self._market(volume_24h=100, market_age_hours=200),
            news, {"avg_score": 0.0}, []
        ) is True

    def test_no_rule_triggers_returns_false(self):
        engine = _make_engine()
        engine._finbert_available = True
        # No news, no momentum, low volume, old market
        assert engine.should_call_ai(
            self._market(volume_24h=100, market_age_hours=200),
            [], {"avg_score": 0.0}, []
        ) is False


# === FIND RELEVANT NEWS ===


class TestFindRelevantNews:
    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty(self):
        engine = _make_engine(api_key="")
        result = await engine.find_relevant_news("Will X happen?")
        assert result == []
