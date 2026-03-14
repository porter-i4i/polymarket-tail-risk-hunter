"""
News Engine: NewsData.io integration with FinBERT sentiment and pre-filter logic.

Fetches news articles relevant to market questions, scores sentiment,
and provides an enhanced pre-filter (should_call_ai) with 6 rules to save
Claude API costs.
"""
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from modules.rate_limiter import RateLimiter
from modules.types import NewsArticle, PricePoint

logger = logging.getLogger(__name__)

# FinBERT model name for sentiment analysis
FINBERT_MODEL = "ProsusAI/finbert"


class NewsEngine:
    def __init__(self, news_limiter: RateLimiter, cfg: dict):
        self.limiter = news_limiter
        self.cfg = cfg
        self.api_key = cfg.get("NEWS_API_KEY", os.getenv("NEWS_API_KEY", ""))
        self._finbert_pipeline = None
        self._finbert_available = False
        self._init_finbert()

    def _init_finbert(self) -> None:
        """Load FinBERT once at init. If load fails, log WARNING and continue."""
        try:
            from transformers import pipeline
            self._finbert_pipeline = pipeline(
                "sentiment-analysis",
                model=FINBERT_MODEL,
                tokenizer=FINBERT_MODEL,
                top_k=None,
            )
            self._finbert_available = True
            logger.info("FinBERT loaded successfully")
        except Exception as e:
            logger.warning("FinBERT load failed, pre-filter will return True when sentiment needed: %s", e)
            self._finbert_available = False

    async def find_relevant_news(self, question: str) -> list[NewsArticle]:
        """
        Query NewsData.io for articles relevant to the market question.
        Returns list of NewsArticle with sentiment labels.
        """
        if not self.api_key:
            logger.debug("No NEWS_API_KEY configured, returning empty news")
            return []

        try:
            await self.limiter.acquire()
            articles = await self._fetch_newsdata(question)
        except Exception as e:
            logger.warning("News fetch failed for '%s': %s", question[:50], e)
            return []

        # Filter stale articles (>72h old)
        fresh = [a for a in articles if a.age_hours <= 72]

        # Score sentiment for each article
        scored = []
        for article in fresh:
            sentiment_label, sentiment_score = self._score_sentiment(
                f"{article.title} {article.description}"
            )
            scored.append(NewsArticle(
                title=article.title,
                description=article.description,
                source=article.source,
                published_at=article.published_at,
                age_hours=article.age_hours,
                sentiment_label=sentiment_label,
                sentiment_score=sentiment_score,
            ))
        return scored

    async def _fetch_newsdata(self, question: str) -> list[NewsArticle]:
        """Fetch from NewsData.io API. Returns raw articles with neutral sentiment."""
        import aiohttp

        # Extract key terms from question for search
        query = self._extract_query(question)
        url = "https://newsdata.io/api/1/news"
        params = {
            "apikey": self.api_key,
            "q": query,
            "language": "en",
            "size": 10,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    logger.warning("NewsData API returned status %d", resp.status)
                    return []
                data = await resp.json()

        results = data.get("results", [])
        now = datetime.now(timezone.utc)
        articles = []
        for item in results:
            pub_str = item.get("pubDate", "")
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                age_hours = (now - pub_dt).total_seconds() / 3600
            except (ValueError, TypeError):
                age_hours = 999.0

            articles.append(NewsArticle(
                title=item.get("title", ""),
                description=item.get("description", "") or "",
                source=item.get("source_id", "unknown"),
                published_at=pub_str,
                age_hours=round(age_hours, 1),
                sentiment_label="neutral",  # Will be scored later
                sentiment_score=0.0,
            ))
        return articles

    @staticmethod
    def _extract_query(question: str) -> str:
        """Extract meaningful search terms from a market question."""
        # Remove common question prefixes
        q = question.strip()
        for prefix in ["Will ", "Is ", "Are ", "Has ", "Have ", "Does ", "Do ", "Can ", "Could ", "Would "]:
            if q.startswith(prefix):
                q = q[len(prefix):]
                break
        # Remove trailing "?" and limit length
        q = q.rstrip("?").strip()
        # Take first 80 chars to keep search focused
        return q[:80]

    def _score_sentiment(self, text: str) -> tuple[str, float]:
        """Score text sentiment using FinBERT. Returns (label, score)."""
        if not self._finbert_available or self._finbert_pipeline is None:
            return ("neutral", 0.0)

        try:
            results = self._finbert_pipeline(text[:512])
            if results and isinstance(results[0], list):
                # top_k returns list of lists
                best = max(results[0], key=lambda x: x["score"])
                return (best["label"], round(best["score"], 4))
            elif results and isinstance(results[0], dict):
                return (results[0]["label"], round(results[0]["score"], 4))
        except Exception as e:
            logger.warning("FinBERT scoring failed: %s", e)
        return ("neutral", 0.0)

    def aggregate_sentiment(self, news: list[NewsArticle]) -> dict:
        """
        Aggregate sentiment across articles.
        Returns dict with avg_score, dominant_label, article_count, fresh_count.
        """
        if not news:
            return {
                "avg_score": 0.0,
                "dominant_label": "neutral",
                "article_count": 0,
                "fresh_count": 0,
            }

        labels = [a.sentiment_label for a in news]
        scores = [a.sentiment_score for a in news]
        fresh = [a for a in news if a.age_hours <= 24]

        # Dominant label by count
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        dominant = max(label_counts, key=label_counts.get) if label_counts else "neutral"

        return {
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "dominant_label": dominant,
            "article_count": len(news),
            "fresh_count": len(fresh),
        }

    def should_call_ai(
        self,
        market: dict,
        news: list[NewsArticle],
        sentiment: dict,
        price_history: list[PricePoint],
    ) -> bool:
        """
        Enhanced pre-filter: 6 rules to decide if this market is worth
        an expensive Claude API call. Returns True if AI scoring should proceed.

        Rules (any single True triggers AI call — default is False/skip):
        1. Fresh news exists: >=1 article <24h old → worth scoring
        2. Strong sentiment signal: avg sentiment score > 0.6 (pos or neg) → worth scoring
        3. Price momentum: recent price change >1pp in price history → worth scoring
        4. High volume for price range: volume_24h > 500 at sub-5% price → unusual activity
        5. New market: market_age_hours < 48 → newly listed, needs initial assessment
        6. Multiple news sources: >=3 articles from different sources → event convergence

        If none of the 6 rules trigger → skip AI (save API costs).
        Graceful degradation: if FinBERT is unavailable, sentiment rules always pass.
        """
        # Rule 1: Fresh news (< 24h old)
        fresh_news = [a for a in news if a.age_hours <= 24]
        if fresh_news:
            return True

        # Rule 2: Strong sentiment signal
        avg_score = sentiment.get("avg_score", 0.0)
        if not self._finbert_available:
            # FinBERT unavailable — can't assess sentiment, pass by default
            if news:
                return True
        elif avg_score > 0.6:
            return True

        # Rule 3: Price momentum (>1pp change in recent history)
        if price_history and len(price_history) >= 2:
            recent = price_history[-3:]
            prices = [p.price for p in recent]
            max_change = max(abs(prices[i] - prices[i - 1]) for i in range(1, len(prices)))
            if max_change > 0.01:  # >1 percentage point
                return True

        # Rule 4: High volume for price range
        volume = market.get("volume_24h", 0)
        yes_price = market.get("yes_price", 1.0)
        if yes_price <= 0.05 and volume > 500:
            return True

        # Rule 5: New market (< 48h old)
        age = market.get("market_age_hours", 9999)
        if age < 48:
            return True

        # Rule 6: Multiple news sources (event convergence)
        if news:
            unique_sources = set(a.source for a in news)
            if len(unique_sources) >= 3:
                return True

        # No rule triggered — skip AI scoring
        return False
