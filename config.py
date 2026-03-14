"""
Configuration loader and validator for Polymarket Tail Risk Hunter.
Loads environment variables, parses to correct types, validates consistency.
Bot MUST NOT start if validate_config() returns any errors.
"""
import os

# === RETRY POLICIES (per API) ===

RETRY_POLICIES = {
    "gamma_api": {
        "max_attempts": 3,
        "initial_wait_sec": 1.0,
        "max_wait_sec": 10.0,
        "exponential_base": 2,
        "jitter": True,
        "retryable_status": [429, 500, 502, 503, 504],
        "retryable_exceptions": ["httpx.ConnectTimeout", "httpx.ReadTimeout", "httpx.ConnectError"],
        "on_exhausted": "return_empty",
    },
    "clob_read": {
        "max_attempts": 3,
        "initial_wait_sec": 1.0,
        "max_wait_sec": 10.0,
        "exponential_base": 2,
        "jitter": True,
        "retryable_status": [429, 500, 502, 503],
        "on_exhausted": "return_none",
    },
    "clob_write": {
        "max_attempts": 1,
        "on_failure": "log_and_skip",
        "cancel_max_attempts": 3,
    },
    "claude_api": {
        "max_attempts": 3,
        "initial_wait_sec": 2.0,
        "max_wait_sec": 30.0,
        "exponential_base": 2,
        "jitter": True,
        "retryable_status": [429, 500, 529],
        "on_exhausted": "return_skip",
    },
    "news_api": {
        "max_attempts": 2,
        "initial_wait_sec": 1.0,
        "max_wait_sec": 5.0,
        "on_exhausted": "return_empty",
    },
    "telegram_api": {
        "max_attempts": 2,
        "initial_wait_sec": 1.0,
        "max_wait_sec": 5.0,
        "on_exhausted": "log_and_continue",
    },
}


# === TYPE PARSING HELPERS ===

def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("true", "1", "yes")


def _parse_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _parse_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# === CONFIG LOADING ===

def load_config() -> dict:
    """
    Load configuration from environment variables.
    Parses values to appropriate Python types (bool/int/float/str).
    Returns a dict ready for validate_config().
    """
    cfg = {}

    # --- Wallet & Auth (strings) ---
    cfg["POLYMARKET_PRIVATE_KEY"] = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
    cfg["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY", "")
    cfg["NEWS_API_KEY"] = os.environ.get("NEWS_API_KEY", "")
    cfg["TELEGRAM_BOT_TOKEN"] = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    cfg["TELEGRAM_CHAT_ID"] = os.environ.get("TELEGRAM_CHAT_ID", "")

    # --- Database ---
    cfg["DB_PATH"] = os.environ.get("DB_PATH", "data/polybot.db")

    # --- Mode (bool / int) ---
    cfg["DRY_RUN"] = _parse_bool(os.environ.get("DRY_RUN", "true"))
    cfg["CYCLE_INTERVAL_SECONDS"] = _parse_int(os.environ.get("CYCLE_INTERVAL_SECONDS", "60"), 60)

    # --- Market Scanning (float / int) ---
    cfg["MAX_PRICE_THRESHOLD"] = _parse_float(os.environ.get("MAX_PRICE_THRESHOLD", "0.05"), 0.05)
    cfg["MIN_VOLUME_24H"] = _parse_float(os.environ.get("MIN_VOLUME_24H", "100"), 100)
    cfg["MIN_LIQUIDITY"] = _parse_float(os.environ.get("MIN_LIQUIDITY", "500"), 500)

    # --- Risk (shared across phases) ---
    cfg["TOTAL_BANKROLL"] = _parse_float(os.environ.get("TOTAL_BANKROLL", "1000"), 1000)
    cfg["MAX_OPEN_POSITIONS"] = _parse_int(os.environ.get("MAX_OPEN_POSITIONS", "200"), 200)
    cfg["MAX_CATEGORY_PCT"] = _parse_float(os.environ.get("MAX_CATEGORY_PCT", "0.15"), 0.15)
    cfg["MAX_DRAWDOWN_PCT"] = _parse_float(os.environ.get("MAX_DRAWDOWN_PCT", "0.30"), 0.30)
    cfg["MIN_AI_CONFIDENCE"] = _parse_int(os.environ.get("MIN_AI_CONFIDENCE", "60"), 60)
    cfg["MIN_EDGE_PCT"] = _parse_float(os.environ.get("MIN_EDGE_PCT", "2.0"), 2.0)
    cfg["MIN_EV"] = _parse_float(os.environ.get("MIN_EV", "0.03"), 0.03)

    # --- Phase 1 (Calibration) ---
    cfg["PHASE1_MAX_DAILY_SPEND"] = _parse_float(os.environ.get("PHASE1_MAX_DAILY_SPEND", "20"), 20)
    cfg["PHASE1_MAX_SINGLE_BET"] = _parse_float(os.environ.get("PHASE1_MAX_SINGLE_BET", "2"), 2)
    cfg["PHASE1_MAX_BETS_PER_DAY"] = _parse_int(os.environ.get("PHASE1_MAX_BETS_PER_DAY", "15"), 15)
    cfg["PHASE1_FLAT_BET"] = _parse_float(os.environ.get("PHASE1_FLAT_BET", "1.5"), 1.5)

    # --- Phase 2 (Scaled) ---
    cfg["PHASE2_MAX_DAILY_SPEND"] = _parse_float(os.environ.get("PHASE2_MAX_DAILY_SPEND", "60"), 60)
    cfg["PHASE2_MAX_SINGLE_BET"] = _parse_float(os.environ.get("PHASE2_MAX_SINGLE_BET", "25"), 25)
    cfg["PHASE2_MAX_BETS_PER_DAY"] = _parse_int(os.environ.get("PHASE2_MAX_BETS_PER_DAY", "25"), 25)
    cfg["PHASE2_KELLY_FRACTION"] = _parse_float(os.environ.get("PHASE2_KELLY_FRACTION", "0.25"), 0.25)

    # --- Phase 3 (Full) ---
    cfg["PHASE3_DAILY_SPEND_PCT"] = _parse_float(os.environ.get("PHASE3_DAILY_SPEND_PCT", "0.10"), 0.10)
    cfg["PHASE3_MAX_SINGLE_BET"] = _parse_float(os.environ.get("PHASE3_MAX_SINGLE_BET", "50"), 50)
    cfg["PHASE3_MAX_BETS_PER_DAY"] = _parse_int(os.environ.get("PHASE3_MAX_BETS_PER_DAY", "30"), 30)
    cfg["PHASE3_KELLY_FRACTION"] = _parse_float(os.environ.get("PHASE3_KELLY_FRACTION", "0.50"), 0.50)

    # --- Calibration & Debiasing ---
    cfg["CALIBRATION_METHOD"] = os.environ.get("CALIBRATION_METHOD", "platt")
    cfg["CALIBRATION_MIN_SAMPLES"] = _parse_int(os.environ.get("CALIBRATION_MIN_SAMPLES", "100"), 100)
    cfg["CALIBRATION_RETRAIN_EVERY"] = _parse_int(os.environ.get("CALIBRATION_RETRAIN_EVERY", "50"), 50)
    # Feature flag checked in main_loop: ENABLE_DEBIASER (see FEATURE FLAGS section below)
    cfg["USE_CONSERVATIVE_ESTIMATE"] = _parse_bool(os.environ.get("USE_CONSERVATIVE_ESTIMATE", "true"))
    cfg["SCORING_CACHE_DEFAULT_TTL"] = _parse_int(os.environ.get("SCORING_CACHE_DEFAULT_TTL", "1800"), 1800)
    cfg["MIN_PREDICTIONS_PHASE2"] = _parse_int(os.environ.get("MIN_PREDICTIONS_PHASE2", "200"), 200)
    cfg["MIN_PREDICTIONS_PHASE3"] = _parse_int(os.environ.get("MIN_PREDICTIONS_PHASE3", "500"), 500)
    cfg["MIN_CAL_SCORE_PHASE2"] = _parse_float(os.environ.get("MIN_CAL_SCORE_PHASE2", "0.60"), 0.60)
    cfg["MIN_CAL_SCORE_PHASE3"] = _parse_float(os.environ.get("MIN_CAL_SCORE_PHASE3", "0.65"), 0.65)
    cfg["CIRCUIT_BREAKER_DAYS"] = _parse_int(os.environ.get("CIRCUIT_BREAKER_DAYS", "7"), 7)
    cfg["CIRCUIT_BREAKER_LOSS_PCT"] = _parse_float(os.environ.get("CIRCUIT_BREAKER_LOSS_PCT", "0.15"), 0.15)

    # --- Adverse Selection ---
    cfg["TOXICITY_THRESHOLD"] = _parse_float(os.environ.get("TOXICITY_THRESHOLD", "0.50"), 0.50)
    cfg["VOLUME_SPIKE_THRESHOLD"] = _parse_float(os.environ.get("VOLUME_SPIKE_THRESHOLD", "3.0"), 3.0)
    cfg["BOOK_IMBALANCE_THRESHOLD"] = _parse_float(os.environ.get("BOOK_IMBALANCE_THRESHOLD", "0.70"), 0.70)
    cfg["MIN_BOOK_DEPTH"] = _parse_float(os.environ.get("MIN_BOOK_DEPTH", "200"), 200)
    cfg["MAX_SPREAD_PCT"] = _parse_float(os.environ.get("MAX_SPREAD_PCT", "0.50"), 0.50)

    # --- Exit Strategy ---
    cfg["EXIT_PROFIT_MULTIPLIER"] = _parse_float(os.environ.get("EXIT_PROFIT_MULTIPLIER", "3.0"), 3.0)
    cfg["EXIT_TIME_DECAY_HOURS"] = _parse_float(os.environ.get("EXIT_TIME_DECAY_HOURS", "24"), 24)
    cfg["EXIT_STOP_LOSS_PCT"] = _parse_float(os.environ.get("EXIT_STOP_LOSS_PCT", "0.70"), 0.70)

    # --- Correlation ---
    cfg["MAX_CLUSTER_EXPOSURE_PCT"] = _parse_float(os.environ.get("MAX_CLUSTER_EXPOSURE_PCT", "0.20"), 0.20)
    cfg["SIMILARITY_THRESHOLD"] = _parse_float(os.environ.get("SIMILARITY_THRESHOLD", "0.35"), 0.35)

    # --- Rate Limits ---
    cfg["GAMMA_RATE_LIMIT"] = _parse_int(os.environ.get("GAMMA_RATE_LIMIT", "80"), 80)
    cfg["CLOB_DATA_RATE_LIMIT"] = _parse_int(os.environ.get("CLOB_DATA_RATE_LIMIT", "80"), 80)
    cfg["CLOB_ORDER_RATE_LIMIT"] = _parse_int(os.environ.get("CLOB_ORDER_RATE_LIMIT", "40"), 40)
    cfg["NEWS_RATE_LIMIT"] = _parse_int(os.environ.get("NEWS_RATE_LIMIT", "12"), 12)
    cfg["CLAUDE_RATE_LIMIT"] = _parse_int(os.environ.get("CLAUDE_RATE_LIMIT", "50"), 50)

    # --- Feature Flags ---
    cfg["ENABLE_DEBIASER"] = _parse_bool(os.environ.get("ENABLE_DEBIASER", "true"))
    cfg["ENABLE_PLATT_SCALING"] = _parse_bool(os.environ.get("ENABLE_PLATT_SCALING", "true"))
    cfg["ENABLE_ADVERSE_SELECTION"] = _parse_bool(os.environ.get("ENABLE_ADVERSE_SELECTION", "true"))
    cfg["ENABLE_ORDER_BOOK_HEALTH"] = _parse_bool(os.environ.get("ENABLE_ORDER_BOOK_HEALTH", "true"))
    cfg["ENABLE_EXIT_MANAGER"] = _parse_bool(os.environ.get("ENABLE_EXIT_MANAGER", "true"))
    cfg["ENABLE_CORRELATION_CLUSTERS"] = _parse_bool(os.environ.get("ENABLE_CORRELATION_CLUSTERS", "true"))
    cfg["ENABLE_SIGNAL_AGGREGATOR"] = _parse_bool(os.environ.get("ENABLE_SIGNAL_AGGREGATOR", "true"))
    cfg["ENABLE_FEE_CHECK"] = _parse_bool(os.environ.get("ENABLE_FEE_CHECK", "true"))

    return cfg


# === CONFIG VALIDATION ===

def validate_config(cfg: dict) -> list[str]:
    """
    Returns list of error strings. Empty list = valid.
    Bot MUST NOT start if any errors.
    """
    errors = []

    # Required secrets
    for key in ["POLYMARKET_PRIVATE_KEY", "ANTHROPIC_API_KEY",
                "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]:
        if not cfg.get(key):
            errors.append(f"MISSING REQUIRED: {key}")

    # Range checks
    if not (0.01 <= cfg.get("MAX_PRICE_THRESHOLD", 0) <= 0.20):
        errors.append(f"MAX_PRICE_THRESHOLD={cfg.get('MAX_PRICE_THRESHOLD')} not in [0.01, 0.20]")
    if not (50 <= cfg.get("TOTAL_BANKROLL", 0) <= 100000):
        errors.append(f"TOTAL_BANKROLL={cfg.get('TOTAL_BANKROLL')} suspicious")
    if not (10 <= cfg.get("CYCLE_INTERVAL_SECONDS", 0) <= 600):
        errors.append(f"CYCLE_INTERVAL too extreme: {cfg.get('CYCLE_INTERVAL_SECONDS')}s")

    # Phase consistency
    p1 = cfg.get("PHASE1_MAX_DAILY_SPEND", 0)
    p2 = cfg.get("PHASE2_MAX_DAILY_SPEND", 0)
    if p1 >= p2:
        errors.append(f"Phase 1 spend ({p1}) >= Phase 2 ({p2})")

    k2 = cfg.get("PHASE2_KELLY_FRACTION", 0)
    k3 = cfg.get("PHASE3_KELLY_FRACTION", 0)
    if k2 >= k3:
        errors.append(f"Kelly fractions not increasing: Phase2={k2}, Phase3={k3}")

    # Calibration thresholds
    c2 = cfg.get("MIN_CAL_SCORE_PHASE2", 0)
    c3 = cfg.get("MIN_CAL_SCORE_PHASE3", 0)
    if c2 >= c3:
        errors.append(f"Cal score thresholds not increasing: Phase2={c2}, Phase3={c3}")

    return errors
