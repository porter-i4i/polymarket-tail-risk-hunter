"""
Shared type definitions for all modules.
Every inter-module data exchange uses these types.
"""
from dataclasses import dataclass, field
from enum import IntEnum, Enum
from datetime import datetime

# === ENUMS ===


class BotPhase(IntEnum):
    CALIBRATION = 1
    SCALED = 2
    FULL = 3
    CIRCUIT_BREAKER = 4
    KILLED = 5


class Recommendation(str, Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    SKIP = "SKIP"


# === MARKET DATA ===


@dataclass(frozen=True)
class MarketCandidate:
    """Immutable candidate from scanner. ALL fields mandatory."""
    condition_id: str           # Non-empty Polymarket condition ID
    question: str               # Non-empty market question
    category: str               # From tags[0].label or groupItemTitle; "Uncategorized" if empty
    yes_token_id: str           # CLOB token for YES
    no_token_id: str            # CLOB token for NO
    yes_price: float            # 0.001–1.0 (UNITS: probability)
    no_price: float             # 0.001–1.0 (UNITS: probability)
    volume_24h: float           # >= 0, USD
    liquidity: float            # >= 0, USD
    end_date: str               # ISO 8601 or "" if no end date
    tags: list[str]             # Category tags, may be empty list
    market_age_hours: float     # Hours since creation, >= 0


@dataclass(frozen=True)
class OrderBook:
    """Parsed order book snapshot."""
    bids: list[tuple[float, float]]   # [(price, size), ...] top 10, price descending
    asks: list[tuple[float, float]]   # [(price, size), ...] top 10, price ascending
    spread: float                      # best_ask - best_bid
    midpoint: float                    # (best_ask + best_bid) / 2


@dataclass(frozen=True)
class PricePoint:
    """Single price history data point."""
    timestamp: str      # ISO 8601
    price: float        # 0.0–1.0
    volume: float       # >= 0


@dataclass(frozen=True)
class NewsArticle:
    """Single news article with sentiment."""
    title: str
    description: str
    source: str
    published_at: str           # ISO 8601
    age_hours: float            # Hours since publication
    sentiment_label: str        # "positive", "negative", "neutral"
    sentiment_score: float      # 0.0–1.0, strength


# === AI SCORING ===


@dataclass
class RawAIScore:
    """Direct output from Claude API (before debiasing/calibration)."""
    base_rate: int              # 0–100 (UNITS: percent)
    p10: int                    # 0–100 (UNITS: percent)
    p50: int                    # 0–100 (UNITS: percent)
    p90: int                    # 0–100 (UNITS: percent)
    confidence: int             # 0–100 (UNITS: percent)
    mispricing: str             # "underpriced" | "overpriced" | "fair"
    edge_pct: float             # Numeric edge estimate
    key_factor: str             # Single most important signal
    reasoning: str              # 2-3 sentence reasoning
    evidence_for: list[str]     # Max 3 items
    evidence_against: list[str]  # Min 2 items
    recommendation: str         # "BUY_YES" | "BUY_NO" | "SKIP"


@dataclass
class AdjustedScore:
    """After full pipeline: debiasing -> Platt -> adjustments."""
    raw: RawAIScore                     # Original AI output
    adjusted_p10: float                 # 0.0–1.0 (UNITS: probability)
    adjusted_p50: float                 # 0.0–1.0 (UNITS: probability)
    adjusted_p90: float                 # 0.0–1.0 (UNITS: probability)
    adjusted_probability: float         # = adjusted_p50
    expected_value: float               # EV with fees, can be negative
    recommendation: str                 # May differ from raw if sanity checks override
    ci_width: float                     # adjusted_p90 - adjusted_p10 (UNITS: probability)
    debiased_p50: float                 # After debiasing, before Platt (for logging)


# === RISK & EXECUTION ===


@dataclass(frozen=True)
class FlowToxicity:
    """Adverse selection assessment."""
    score: float            # 0.0–1.0, higher = more toxic
    reason: str             # Human-readable explanation
    should_skip: bool       # True if score >= toxicity_threshold
    signals: list[str]      # Individual signals triggered


@dataclass(frozen=True)
class BookHealth:
    """Order book quality assessment."""
    score: float            # 0.0–1.0, higher = healthier
    is_healthy: bool        # True if score >= 0.5
    issues: list[str]       # Human-readable issues found


@dataclass
class RiskDecision:
    """Output of risk manager check."""
    approved: bool
    checks: dict[str, bool]       # {check_name: passed}
    failed: list[str]             # Names of failed checks
    bet_size: float               # USD, 0 if not approved
    sizing_method: str            # "flat" | "robust_kelly"
    kelly_raw: float              # Raw Kelly fraction before penalties
    estimation_penalty: float     # sigma^2/p^2 penalty applied
    price_penalty: float          # Price-tier penalty applied


@dataclass
class OrderResult:
    """Result of order placement attempt."""
    success: bool
    order_id: str               # Empty string if failed
    error: str                  # Empty string if success
    price: float
    size: float
    side: str                   # "BUY_YES" | "BUY_NO"
    is_dry_run: bool


# === MONITORING ===


@dataclass
class CalibrationMetrics:
    """Full calibration report."""
    calibration_score: float        # Point estimate
    ci_lower: float                 # 2.5th percentile
    ci_upper: float                 # 97.5th percentile
    ai_brier: float
    market_brier: float
    win_rate_actual: float
    win_rate_expected: float
    win_rate_ratio: float           # actual / expected
    sharpe_30d: float
    prediction_count: int
    phase_recommendation: int       # Recommended BotPhase value


@dataclass
class CycleSummary:
    """Structured cycle report — logged every cycle."""
    cycle_id: int
    phase: int
    duration_seconds: float
    markets_scanned: int
    candidates_found: int
    prefilter_passed: int
    ai_scored: int
    cache_hits: int
    risk_approved: int
    orders_placed: int
    orders_failed: int
    exits_executed: int
    resolutions_processed: int
    daily_spend_so_far: float
    daily_bet_count: int
    balance: float
    calibration_score: float
    errors: list[str]


# === SCAN RESULT ===


@dataclass
class ScanResult:
    """Return type of MarketScanner.scan_all_markets()."""
    candidates: list[MarketCandidate]
    total_scanned: int
    total_filtered: int
    scan_duration_seconds: float
    errors: list[str]


# === COMPOSITE SIGNAL ===


@dataclass
class CompositeSignal:
    """Multi-source signal aggregation result."""
    composite_score: float      # -1.0 to +1.0
    should_score_with_ai: bool  # True if |composite_score| > 0.2
    signals: dict[str, tuple[float, float]]  # {name: (score, confidence)}
