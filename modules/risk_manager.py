"""
Risk Manager: core decision layer with 13 named checks and position sizing.

Phase 1: flat bet (PHASE1_FLAT_BET)
Phase 2/3: robust Kelly with estimation penalty, price penalty, and phase fraction.
"""

import logging
import math
from datetime import date

from modules.types import (
    AdjustedScore,
    BookHealth,
    BotPhase,
    FlowToxicity,
    MarketCandidate,
    OrderBook,
    Recommendation,
    RiskDecision,
)

logger = logging.getLogger(__name__)

# Price-tier penalty table: price_upper_bound -> penalty fraction
PRICE_PENALTY = {
    0.005: 0.10,
    0.01: 0.20,
    0.02: 0.30,
    0.03: 0.50,
    0.05: 0.70,
    0.10: 0.85,
    1.00: 1.00,
}


def _get_price_penalty(price: float) -> float:
    """Lookup price penalty by tier."""
    for upper, penalty in sorted(PRICE_PENALTY.items()):
        if price <= upper:
            return penalty
    return 1.0


class RiskManager:
    def __init__(self, db, fee_calc, correlation_mgr, cfg: dict):
        self.db = db
        self.fee_calc = fee_calc
        self.correlation_mgr = correlation_mgr
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Phase limit lookup
    # ------------------------------------------------------------------

    def get_phase_limits(self, phase: BotPhase) -> dict:
        """Return daily-spend, single-bet, bets-per-day, kelly-fraction for phase."""
        cfg = self.cfg
        if phase == BotPhase.CALIBRATION:
            return {
                "max_daily_spend": cfg.get("PHASE1_MAX_DAILY_SPEND", 20),
                "max_single_bet": cfg.get("PHASE1_MAX_SINGLE_BET", 2),
                "max_bets_per_day": cfg.get("PHASE1_MAX_BETS_PER_DAY", 15),
                "kelly_fraction": 0.0,  # flat sizing
                "flat_bet": cfg.get("PHASE1_FLAT_BET", 1.5),
            }
        elif phase == BotPhase.SCALED:
            return {
                "max_daily_spend": cfg.get("PHASE2_MAX_DAILY_SPEND", 60),
                "max_single_bet": cfg.get("PHASE2_MAX_SINGLE_BET", 25),
                "max_bets_per_day": cfg.get("PHASE2_MAX_BETS_PER_DAY", 25),
                "kelly_fraction": cfg.get("PHASE2_KELLY_FRACTION", 0.25),
            }
        elif phase == BotPhase.FULL:
            return {
                "max_daily_spend": cfg.get("PHASE3_DAILY_SPEND_PCT", 0.10) * cfg.get("TOTAL_BANKROLL", 1000),
                "max_single_bet": cfg.get("PHASE3_MAX_SINGLE_BET", 50),
                "max_bets_per_day": cfg.get("PHASE3_MAX_BETS_PER_DAY", 30),
                "kelly_fraction": cfg.get("PHASE3_KELLY_FRACTION", 0.50),
            }
        else:
            # CIRCUIT_BREAKER / KILLED — no trading
            return {
                "max_daily_spend": 0,
                "max_single_bet": 0,
                "max_bets_per_day": 0,
                "kelly_fraction": 0.0,
            }

    # ------------------------------------------------------------------
    # Main entry: check 13 gates then size
    # ------------------------------------------------------------------

    def check_opportunity(
        self,
        score: AdjustedScore,
        market: MarketCandidate,
        ob: OrderBook,
        phase: BotPhase,
        toxicity: FlowToxicity,
        book_health: BookHealth,
    ) -> RiskDecision:
        """Run all 13 checks. If all pass, compute position size."""
        limits = self.get_phase_limits(phase)
        checks: dict[str, bool] = {}
        failed: list[str] = []

        today = date.today()
        bankroll = self.cfg.get("TOTAL_BANKROLL", 1000)

        # 1. phase_allows_trading
        allows = phase in (BotPhase.CALIBRATION, BotPhase.SCALED, BotPhase.FULL)
        checks["phase_allows_trading"] = allows
        if not allows:
            failed.append("phase_allows_trading")

        # 2. recommendation_is_buy
        is_buy = score.recommendation in (Recommendation.BUY_YES.value, Recommendation.BUY_NO.value)
        checks["recommendation_is_buy"] = is_buy
        if not is_buy:
            failed.append("recommendation_is_buy")

        # 3. min_confidence
        min_conf = self.cfg.get("MIN_AI_CONFIDENCE", 60)
        conf = score.raw.confidence
        checks["min_confidence"] = conf >= min_conf
        if not checks["min_confidence"]:
            failed.append("min_confidence")

        # 4. min_edge
        min_edge = self.cfg.get("MIN_EDGE_PCT", 2.0)
        entry_price = market.yes_price if score.recommendation == Recommendation.BUY_YES.value else market.no_price
        edge_pct = (score.adjusted_probability - entry_price) / entry_price * 100 if entry_price > 0 else 0
        checks["min_edge"] = edge_pct >= min_edge
        if not checks["min_edge"]:
            failed.append("min_edge")

        # 5. min_ev
        min_ev = self.cfg.get("MIN_EV", 0.03)
        ev = self.fee_calc.calc_ev_with_fees(score.adjusted_probability, entry_price)
        checks["min_ev"] = ev >= min_ev
        if not checks["min_ev"]:
            failed.append("min_ev")

        # 6. daily_spend_limit
        daily_spend = self.db.get_daily_spend(today)
        checks["daily_spend_limit"] = daily_spend < limits["max_daily_spend"]
        if not checks["daily_spend_limit"]:
            failed.append("daily_spend_limit")

        # 7. daily_bet_limit
        bet_count = self.db.get_daily_bet_count(today)
        checks["daily_bet_limit"] = bet_count < limits["max_bets_per_day"]
        if not checks["daily_bet_limit"]:
            failed.append("daily_bet_limit")

        # 8. max_positions
        max_pos = self.cfg.get("MAX_OPEN_POSITIONS", 200)
        open_count = self.db.get_open_position_count()
        checks["max_positions"] = open_count < max_pos
        if not checks["max_positions"]:
            failed.append("max_positions")

        # 9. category_exposure
        max_cat_pct = self.cfg.get("MAX_CATEGORY_PCT", 0.15)
        cat_exposure = self.db.get_category_exposure(market.category)
        checks["category_exposure"] = cat_exposure < max_cat_pct * bankroll
        if not checks["category_exposure"]:
            failed.append("category_exposure")

        # 10. cluster_exposure
        cluster_ok = True
        if self.correlation_mgr is not None:
            try:
                cluster_exp = self.correlation_mgr.get_cluster_exposure(market.question, self.db)
                max_cluster = self.cfg.get("MAX_CLUSTER_EXPOSURE_PCT", 0.20) * bankroll
                cluster_ok = cluster_exp < max_cluster
            except Exception:
                cluster_ok = True  # fail-open if correlation not available
        checks["cluster_exposure"] = cluster_ok
        if not cluster_ok:
            failed.append("cluster_exposure")

        # 11. adverse_selection
        checks["adverse_selection"] = not toxicity.should_skip
        if toxicity.should_skip:
            failed.append("adverse_selection")

        # 12. book_health
        checks["book_health"] = book_health.is_healthy
        if not book_health.is_healthy:
            failed.append("book_health")

        # 13. dedup
        has_dup = self.db.has_active_position(market.condition_id)
        checks["dedup"] = not has_dup
        if has_dup:
            failed.append("dedup")

        # If any failed, return early
        if failed:
            return RiskDecision(
                approved=False,
                checks=checks,
                failed=failed,
                bet_size=0.0,
                sizing_method="none",
                kelly_raw=0.0,
                estimation_penalty=0.0,
                price_penalty=0.0,
            )

        # All checks passed — compute position size
        bet_size, sizing_method, kelly_raw, est_penalty, price_pen = (
            self.calculate_position_size(score, entry_price, phase, limits, bankroll)
        )

        # Minimum bet check
        if bet_size < 0.50:
            return RiskDecision(
                approved=False,
                checks=checks,
                failed=["bet_too_small"],
                bet_size=bet_size,
                sizing_method=sizing_method,
                kelly_raw=kelly_raw,
                estimation_penalty=est_penalty,
                price_penalty=price_pen,
            )

        return RiskDecision(
            approved=True,
            checks=checks,
            failed=[],
            bet_size=round(bet_size, 2),
            sizing_method=sizing_method,
            kelly_raw=round(kelly_raw, 6),
            estimation_penalty=round(est_penalty, 6),
            price_penalty=round(price_pen, 4),
        )

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        score: AdjustedScore,
        entry_price: float,
        phase: BotPhase,
        limits: dict,
        bankroll: float,
    ) -> tuple[float, str, float, float, float]:
        """
        Returns (bet_size, sizing_method, kelly_raw, estimation_penalty, price_penalty).

        Phase 1: flat bet.
        Phase 2/3: robust Kelly.
        """
        if phase == BotPhase.CALIBRATION:
            flat = limits.get("flat_bet", 1.5)
            return (flat, "flat", 0.0, 0.0, 0.0)

        # Robust Kelly sizing
        p = score.adjusted_probability
        q = 1.0 - p
        b = (1.0 / entry_price) - 1.0 if entry_price > 0 else 0.0  # odds

        if b <= 0:
            return (0.0, "robust_kelly", 0.0, 0.0, 0.0)

        # Raw Kelly: f* = (p*b - q) / b
        kelly_raw = (p * b - q) / b if b > 0 else 0.0

        if kelly_raw <= 0:
            # No edge — return minimum floor
            return (0.0, "robust_kelly", kelly_raw, 0.0, 0.0)

        # Estimation penalty: sigma^2 / p^2 where sigma = ci_width / 2
        sigma = score.ci_width / 2.0
        est_penalty = (sigma ** 2) / (p ** 2) if p > 0 else 0.0

        # Price penalty from tier table
        price_pen = _get_price_penalty(entry_price)

        # Penalized Kelly
        kelly_adj = kelly_raw - est_penalty
        if kelly_adj <= 0:
            return (0.0, "robust_kelly", kelly_raw, est_penalty, price_pen)

        kelly_penalized = kelly_adj * price_pen

        # Phase fraction
        fraction = limits.get("kelly_fraction", 0.25)
        kelly_final = kelly_penalized * fraction

        # Bet size = kelly_final * bankroll
        bet_size = kelly_final * bankroll

        # Cap at max_single_bet
        max_bet = limits.get("max_single_bet", 25)
        bet_size = min(bet_size, max_bet)

        # Cap at remaining daily spend
        today = date.today()
        daily_spend = self.db.get_daily_spend(today)
        remaining = limits.get("max_daily_spend", 60) - daily_spend
        bet_size = min(bet_size, remaining)

        # Ultra-low price cap: price <= 0.005 -> bet <= 1.00
        if entry_price <= 0.005:
            bet_size = min(bet_size, 1.00)

        bet_size = max(0.0, bet_size)

        return (bet_size, "robust_kelly", kelly_raw, est_penalty, price_pen)
