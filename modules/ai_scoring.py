"""
AI Scoring Engine: Claude API integration for market opportunity assessment.

Sends market data, news, and order book info to Claude for probability assessment.
Returns RawAIScore with sanity checks enforced.
"""
import json
import logging
from typing import Optional

from modules.rate_limiter import RateLimiter
from modules.types import NewsArticle, OrderBook, PricePoint, RawAIScore

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "claude-sonnet-4-20250514"
TEMPERATURE = 0
MAX_TOKENS = 1500

SYSTEM_PROMPT = (
    "You are a JSON-only prediction market analyst. "
    "NEVER output anything except valid JSON. "
    "No markdown, no backticks, no preamble. "
    "If you cannot assess, return {\"recommendation\": \"SKIP\", \"p50\": 0, \"confidence\": 0, "
    "\"p10\": 0, \"p90\": 0, \"base_rate\": 0, \"evidence_for\": [], \"evidence_against\": [], "
    "\"mispricing\": \"fair\", \"edge_pct\": 0, \"key_factor\": \"insufficient data\", "
    "\"reasoning\": \"Cannot assess\"}."
)

FEW_SHOT_EXAMPLES = """
## EXAMPLE — Correct SKIP
Question: "Will Japan declare war on China by June 2026?"
YES price: $0.02. News: 1 article about diplomatic tensions, 3 days old.
Result: {"base_rate": 0, "p10": 0, "p50": 1, "p90": 3, "recommendation": "SKIP",
         "reasoning": "Base rate near 0%. Diplomatic tension is normal, no specific escalation evidence."}

## EXAMPLE — Correct BUY_YES
Question: "Will Fed cut rates in March 2026?"
YES price: $0.04. News: 2 articles <6h old — unexpected jobs miss, Fed governor signals action.
Result: {"base_rate": 20, "p10": 8, "p50": 15, "p90": 25, "recommendation": "BUY_YES",
         "reasoning": "Base rate ~20%/meeting. Two specific signals: data shock + explicit Fed language."}

## ANTI-PATTERN (never do this)
Question: "Will Yellowstone erupt by 2027?" YES price: $0.01.
BAD: {"p50": 8, "reasoning": "Yellowstone is overdue"} — "overdue" is a myth, no evidence.
"""

VALID_RECOMMENDATIONS = {"BUY_YES", "BUY_NO", "SKIP"}


class AIScoringEngine:
    def __init__(self, claude_limiter: RateLimiter, cfg: dict):
        self.limiter = claude_limiter
        self.cfg = cfg
        self._client = None
        self._api_calls = 0
        self._api_errors = 0

    def _get_client(self):
        """Lazy-load the Anthropic client."""
        if self._client is not None:
            return self._client
        try:
            import anthropic
            api_key = self.cfg.get("ANTHROPIC_API_KEY", "")
            self._client = anthropic.Anthropic(api_key=api_key)
            return self._client
        except Exception as e:
            logger.error("Failed to create Anthropic client: %s", e)
            raise

    @property
    def error_rate(self) -> float:
        """Current API error rate."""
        if self._api_calls == 0:
            return 0.0
        return self._api_errors / self._api_calls

    async def score_opportunity(
        self,
        market: dict,
        news: list[NewsArticle],
        order_book: Optional[OrderBook],
        price_history: list[PricePoint],
    ) -> RawAIScore:
        """
        Score a market opportunity using Claude API.
        Returns RawAIScore with sanity checks applied.
        """
        await self.limiter.acquire()
        self._api_calls += 1

        user_prompt = self._build_user_prompt(market, news, order_book, price_history)

        try:
            import asyncio
            client = self._get_client()
            response = await asyncio.to_thread(
                client.messages.create,
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_text = response.content[0].text.strip()
        except Exception as e:
            self._api_errors += 1
            logger.error("Claude API call failed: %s", e)
            raise

        # Parse response
        score = self._parse_response(raw_text)

        # Apply sanity checks (may force SKIP)
        score = self._sanity_check(score)

        return score

    def _build_user_prompt(
        self,
        market: dict,
        news: list[NewsArticle],
        order_book: Optional[OrderBook],
        price_history: list[PricePoint],
    ) -> str:
        """Build the user prompt with market data and few-shot examples."""
        question = market.get("question", "Unknown")
        yes_price = market.get("yes_price", 0)
        no_price = market.get("no_price", 0)
        volume = market.get("volume_24h", 0)
        category = market.get("category", "Unknown")
        end_date = market.get("end_date", "")
        market_age = market.get("market_age_hours", 0)

        # Format news
        news_text = "None"
        if news:
            articles = []
            for a in news[:5]:
                articles.append(
                    f"- [{a.source}] {a.title} ({a.sentiment_label}, "
                    f"score={a.sentiment_score:.2f}, age={a.age_hours:.0f}h)"
                )
            news_text = "\n".join(articles)

        # Format order book
        ob_text = "Not available"
        if order_book:
            ob_text = f"Spread: {order_book.spread:.4f}, Midpoint: {order_book.midpoint:.4f}"
            if order_book.bids:
                ob_text += f", Best bid: {order_book.bids[0][0]:.4f}"
            if order_book.asks:
                ob_text += f", Best ask: {order_book.asks[0][0]:.4f}"

        # Format price history
        hist_text = "Not available"
        if price_history:
            recent = price_history[-5:]
            prices = [f"{p.price:.4f}" for p in recent]
            hist_text = f"Recent prices: {', '.join(prices)}"

        prompt = f"""Analyze this prediction market opportunity and return a JSON assessment.

{FEW_SHOT_EXAMPLES}

Now analyze this market:

Question: "{question}"
Category: {category}
YES price: ${yes_price:.4f} (implied probability: {yes_price*100:.1f}%)
NO price: ${no_price:.4f}
24h Volume: ${volume:,.0f}
Market age: {market_age:.0f} hours
End date: {end_date or "Not specified"}

News:
{news_text}

Order Book: {ob_text}
Price History: {hist_text}

Return JSON with these exact fields:
- base_rate: int 0-100 (historical base rate for this type of event, percent)
- p10: int 0-100 (10th percentile probability estimate, percent)
- p50: int 0-100 (median probability estimate, percent)
- p90: int 0-100 (90th percentile probability estimate, percent)
- confidence: int 0-100 (your confidence in this assessment, percent)
- mispricing: "underpriced" | "overpriced" | "fair"
- edge_pct: float (estimated edge over market in percentage points)
- key_factor: str (single most important signal)
- reasoning: str (2-3 sentence reasoning)
- evidence_for: list[str] (max 3 items supporting the event)
- evidence_against: list[str] (min 2 items against the event)
- recommendation: "BUY_YES" | "BUY_NO" | "SKIP"
"""
        return prompt

    @staticmethod
    def _parse_response(raw_text: str) -> RawAIScore:
        """Parse Claude's JSON response into RawAIScore. No eval()."""
        try:
            # Strip potential markdown code fences
            text = raw_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                # Remove first and last lines (code fences)
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)

            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse AI response as JSON: %s", e)
            return _skip_score("JSON parse error")

        # Extract with defaults
        try:
            return RawAIScore(
                base_rate=int(data.get("base_rate", 0)),
                p10=int(data.get("p10", 0)),
                p50=int(data.get("p50", 0)),
                p90=int(data.get("p90", 0)),
                confidence=int(data.get("confidence", 0)),
                mispricing=str(data.get("mispricing", "fair")),
                edge_pct=float(data.get("edge_pct", 0)),
                key_factor=str(data.get("key_factor", "unknown")),
                reasoning=str(data.get("reasoning", ""))[:500],
                evidence_for=list(data.get("evidence_for", []))[:3],
                evidence_against=list(data.get("evidence_against", []))[:3],
                recommendation=str(data.get("recommendation", "SKIP")),
            )
        except (ValueError, TypeError) as e:
            logger.warning("Failed to construct RawAIScore: %s", e)
            return _skip_score("Score construction error")

    @staticmethod
    def _sanity_check(score: RawAIScore) -> RawAIScore:
        """
        Enforce sanity checks. Force SKIP on violation.
        1. p50 > 50 → force SKIP (tail events never >50%)
        2. p10 > p50 or p50 > p90 → force SKIP
        3. CI width (p90 - p10) > 20 → force SKIP (AI too uncertain)
        4. recommendation not in valid set → force SKIP
        """
        # Check 1: p50 > 50 → force SKIP
        if score.p50 > 50:
            logger.info("Sanity: p50=%d > 50, forcing SKIP", score.p50)
            return _force_skip(score, "p50 > 50: tail events never >50%")

        # Check 2: monotonicity p10 <= p50 <= p90
        if score.p10 > score.p50 or score.p50 > score.p90:
            logger.info("Sanity: p10=%d, p50=%d, p90=%d non-monotonic, forcing SKIP",
                       score.p10, score.p50, score.p90)
            return _force_skip(score, "Non-monotonic distribution")

        # Check 3: CI width > 20
        ci_width = score.p90 - score.p10
        if ci_width > 20:
            logger.info("Sanity: CI width=%d > 20, forcing SKIP", ci_width)
            return _force_skip(score, f"CI width {ci_width} > 20")

        # Check 4: invalid recommendation
        if score.recommendation not in VALID_RECOMMENDATIONS:
            logger.info("Sanity: invalid recommendation '%s', forcing SKIP",
                       score.recommendation)
            return _force_skip(score, f"Invalid recommendation: {score.recommendation}")

        return score


def _skip_score(reason: str) -> RawAIScore:
    """Return a default SKIP score."""
    return RawAIScore(
        base_rate=0,
        p10=0,
        p50=0,
        p90=0,
        confidence=0,
        mispricing="fair",
        edge_pct=0,
        key_factor="insufficient data",
        reasoning=reason,
        evidence_for=[],
        evidence_against=[],
        recommendation="SKIP",
    )


def _force_skip(score: RawAIScore, reason: str) -> RawAIScore:
    """Return a copy of score with recommendation forced to SKIP."""
    return RawAIScore(
        base_rate=score.base_rate,
        p10=score.p10,
        p50=score.p50,
        p90=score.p90,
        confidence=score.confidence,
        mispricing=score.mispricing,
        edge_pct=score.edge_pct,
        key_factor=score.key_factor,
        reasoning=f"FORCED SKIP: {reason}. Original: {score.reasoning}",
        evidence_for=score.evidence_for,
        evidence_against=score.evidence_against,
        recommendation="SKIP",
    )
