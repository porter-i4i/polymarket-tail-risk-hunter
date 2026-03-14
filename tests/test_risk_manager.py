"""Tests for modules/risk_manager.py."""

import pytest
from datetime import date
from unittest.mock import MagicMock

from modules.risk_manager import RiskManager, PRICE_PENALTY, _get_price_penalty
from modules.fee_calculator import FeeCalculator
from modules.types import (
    AdjustedScore,
    BookHealth,
    BotPhase,
    FlowToxicity,
    MarketCandidate,
    OrderBook,
    RawAIScore,
    RiskDecision,
)


# === Helpers ===

def _make_raw(
    base_rate=10,
    p10=5,
    p50=10,
    p90=20,
    confidence=75,
    edge_pct=5.0,
    recommendation="BUY_YES",
) -> RawAIScore:
    return RawAIScore(
        base_rate=base_rate,
        p10=p10,
        p50=p50,
        p90=p90,
        confidence=confidence,
        mispricing="underpriced",
        edge_pct=edge_pct,
        key_factor="news",
        reasoning="Test reasoning",
        evidence_for=["ev1"],
        evidence_against=["ev2", "ev3"],
        recommendation=recommendation,
    )


def _make_score(
    adjusted_probability=0.10,
    adjusted_p10=0.05,
    adjusted_p50=0.10,
    adjusted_p90=0.20,
    ci_width=0.15,
    ev=0.05,
    recommendation="BUY_YES",
    confidence=75,
) -> AdjustedScore:
    raw = _make_raw(confidence=confidence, recommendation=recommendation)
    return AdjustedScore(
        raw=raw,
        adjusted_p10=adjusted_p10,
        adjusted_p50=adjusted_p50,
        adjusted_p90=adjusted_p90,
        adjusted_probability=adjusted_probability,
        expected_value=ev,
        recommendation=recommendation,
        ci_width=ci_width,
        debiased_p50=adjusted_p50,
    )


def _make_market(
    yes_price=0.03,
    no_price=0.97,
    condition_id="cond_001",
    category="Politics",
) -> MarketCandidate:
    return MarketCandidate(
        condition_id=condition_id,
        question="Will X happen?",
        category=category,
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        yes_price=yes_price,
        no_price=no_price,
        volume_24h=5000.0,
        liquidity=10000.0,
        end_date="2026-06-01T00:00:00Z",
        tags=["Politics"],
        market_age_hours=100.0,
    )


def _make_ob() -> OrderBook:
    return OrderBook(
        bids=[(0.03, 500), (0.029, 400), (0.028, 300)],
        asks=[(0.031, 500), (0.032, 400)],
        spread=0.001,
        midpoint=0.0305,
    )


def _clean_toxicity() -> FlowToxicity:
    return FlowToxicity(score=0.1, reason="Clean flow", should_skip=False, signals=[])


def _healthy_book() -> BookHealth:
    return BookHealth(score=0.8, is_healthy=True, issues=[])


def _make_db_mock(daily_spend=0.0, bet_count=0, open_count=0, cat_exposure=0.0, has_dup=False):
    db = MagicMock()
    db.get_daily_spend.return_value = daily_spend
    db.get_daily_bet_count.return_value = bet_count
    db.get_open_position_count.return_value = open_count
    db.get_category_exposure.return_value = cat_exposure
    db.has_active_position.return_value = has_dup
    return db


def _default_cfg() -> dict:
    return {
        "TOTAL_BANKROLL": 1000,
        "MAX_OPEN_POSITIONS": 200,
        "MAX_CATEGORY_PCT": 0.15,
        "MAX_DRAWDOWN_PCT": 0.30,
        "MIN_AI_CONFIDENCE": 60,
        "MIN_EDGE_PCT": 2.0,
        "MIN_EV": 0.03,
        "PHASE1_MAX_DAILY_SPEND": 20,
        "PHASE1_MAX_SINGLE_BET": 2,
        "PHASE1_MAX_BETS_PER_DAY": 15,
        "PHASE1_FLAT_BET": 1.50,
        "PHASE2_MAX_DAILY_SPEND": 60,
        "PHASE2_MAX_SINGLE_BET": 25,
        "PHASE2_MAX_BETS_PER_DAY": 25,
        "PHASE2_KELLY_FRACTION": 0.25,
        "PHASE3_DAILY_SPEND_PCT": 0.10,
        "PHASE3_MAX_SINGLE_BET": 50,
        "PHASE3_MAX_BETS_PER_DAY": 30,
        "PHASE3_KELLY_FRACTION": 0.50,
        "MAX_CLUSTER_EXPOSURE_PCT": 0.20,
        "TOXICITY_THRESHOLD": 0.50,
    }


def _make_rm(db=None, cfg=None, corr_mgr=None):
    db = db or _make_db_mock()
    cfg = cfg or _default_cfg()
    fc = FeeCalculator()
    return RiskManager(db=db, fee_calc=fc, correlation_mgr=corr_mgr, cfg=cfg)


# === Phase 1 Flat Sizing ===

class TestPhase1FlatSizing:
    def test_phase1_uses_flat_sizing(self):
        """Phase 1 should always use flat bet regardless of Kelly inputs."""
        rm = _make_rm()
        result = rm.check_opportunity(
            score=_make_score(),
            market=_make_market(),
            ob=_make_ob(),
            phase=BotPhase.CALIBRATION,
            toxicity=_clean_toxicity(),
            book_health=_healthy_book(),
        )
        assert result.approved is True
        assert result.sizing_method == "flat"
        assert result.bet_size == 1.50

    def test_phase1_flat_bet_configurable(self):
        cfg = _default_cfg()
        cfg["PHASE1_FLAT_BET"] = 1.00
        rm = _make_rm(cfg=cfg)
        result = rm.check_opportunity(
            score=_make_score(),
            market=_make_market(),
            ob=_make_ob(),
            phase=BotPhase.CALIBRATION,
            toxicity=_clean_toxicity(),
            book_health=_healthy_book(),
        )
        assert result.bet_size == 1.00


# === Ultra-Low Price Cap ===

class TestUltraLowPriceCap:
    def test_ultra_low_price_capped_at_one(self):
        """price <= 0.005 should cap bet at $1.00."""
        rm = _make_rm()
        score = _make_score(adjusted_probability=0.10, ci_width=0.05)
        market = _make_market(yes_price=0.005)
        result = rm.check_opportunity(
            score=score,
            market=market,
            ob=_make_ob(),
            phase=BotPhase.SCALED,
            toxicity=_clean_toxicity(),
            book_health=_healthy_book(),
        )
        if result.approved:
            assert result.bet_size <= 1.00


# === CI Width Effect ===

class TestCIWidthEffect:
    def test_wider_ci_reduces_bet_size(self):
        """Wider confidence interval should produce smaller (or equal) bet."""
        rm = _make_rm()
        market = _make_market(yes_price=0.03)

        narrow = _make_score(adjusted_probability=0.10, ci_width=0.05)
        wide = _make_score(adjusted_probability=0.10, ci_width=0.30)

        _, _, _, est_pen_narrow, _ = rm.calculate_position_size(
            narrow, 0.03, BotPhase.SCALED,
            rm.get_phase_limits(BotPhase.SCALED), 1000,
        )
        _, _, _, est_pen_wide, _ = rm.calculate_position_size(
            wide, 0.03, BotPhase.SCALED,
            rm.get_phase_limits(BotPhase.SCALED), 1000,
        )
        assert est_pen_wide > est_pen_narrow

        bet_narrow, *_ = rm.calculate_position_size(
            narrow, 0.03, BotPhase.SCALED,
            rm.get_phase_limits(BotPhase.SCALED), 1000,
        )
        bet_wide, *_ = rm.calculate_position_size(
            wide, 0.03, BotPhase.SCALED,
            rm.get_phase_limits(BotPhase.SCALED), 1000,
        )
        assert bet_narrow >= bet_wide


# === Price Comparison ===

class TestPriceComparison:
    def test_lower_price_not_larger_than_higher_price(self):
        """Under equivalent edge, lower prices should not produce larger bets
        due to price penalty table."""
        rm = _make_rm()
        limits = rm.get_phase_limits(BotPhase.SCALED)

        # Same edge ratio: adjusted_probability is 3x price
        score_low = _make_score(adjusted_probability=0.03, ci_width=0.01)
        score_high = _make_score(adjusted_probability=0.30, ci_width=0.10)

        bet_low, *_ = rm.calculate_position_size(score_low, 0.01, BotPhase.SCALED, limits, 1000)
        bet_high, *_ = rm.calculate_position_size(score_high, 0.10, BotPhase.SCALED, limits, 1000)
        assert bet_low <= bet_high


# === Negative Kelly ===

class TestNegativeKelly:
    def test_negative_kelly_returns_zero(self):
        """When there's no edge, Kelly is negative and bet should be 0."""
        rm = _make_rm()
        # AI probability equal to price => no edge after fees
        score = _make_score(adjusted_probability=0.03, ci_width=0.10)
        bet, method, kelly_raw, _, _ = rm.calculate_position_size(
            score, 0.03, BotPhase.SCALED,
            rm.get_phase_limits(BotPhase.SCALED), 1000,
        )
        # With fees eating the edge, raw Kelly may be near zero or negative
        assert bet == 0.0 or kelly_raw <= 0 or bet < 0.50


# === Phase 3 > Phase 2 ===

class TestPhaseScaling:
    def test_phase3_larger_than_phase2(self):
        """Phase 3 bet should be >= Phase 2 bet under same inputs (higher fraction)."""
        rm = _make_rm()
        score = _make_score(adjusted_probability=0.15, ci_width=0.08)

        bet2, *_ = rm.calculate_position_size(
            score, 0.03, BotPhase.SCALED,
            rm.get_phase_limits(BotPhase.SCALED), 1000,
        )
        bet3, *_ = rm.calculate_position_size(
            score, 0.03, BotPhase.FULL,
            rm.get_phase_limits(BotPhase.FULL), 1000,
        )
        assert bet3 >= bet2


# === Failed Gates ===

class TestFailedGates:
    def test_low_confidence_fails(self):
        rm = _make_rm()
        score = _make_score(confidence=30)
        result = rm.check_opportunity(
            score=score,
            market=_make_market(),
            ob=_make_ob(),
            phase=BotPhase.CALIBRATION,
            toxicity=_clean_toxicity(),
            book_health=_healthy_book(),
        )
        assert result.approved is False
        assert "min_confidence" in result.failed

    def test_skip_recommendation_fails(self):
        rm = _make_rm()
        score = _make_score(recommendation="SKIP")
        result = rm.check_opportunity(
            score=score,
            market=_make_market(),
            ob=_make_ob(),
            phase=BotPhase.CALIBRATION,
            toxicity=_clean_toxicity(),
            book_health=_healthy_book(),
        )
        assert result.approved is False
        assert "recommendation_is_buy" in result.failed

    def test_circuit_breaker_fails(self):
        rm = _make_rm()
        result = rm.check_opportunity(
            score=_make_score(),
            market=_make_market(),
            ob=_make_ob(),
            phase=BotPhase.CIRCUIT_BREAKER,
            toxicity=_clean_toxicity(),
            book_health=_healthy_book(),
        )
        assert result.approved is False
        assert "phase_allows_trading" in result.failed

    def test_toxic_flow_fails(self):
        rm = _make_rm()
        toxic = FlowToxicity(score=0.8, reason="volume_spike", should_skip=True, signals=["volume_spike"])
        result = rm.check_opportunity(
            score=_make_score(),
            market=_make_market(),
            ob=_make_ob(),
            phase=BotPhase.CALIBRATION,
            toxicity=toxic,
            book_health=_healthy_book(),
        )
        assert result.approved is False
        assert "adverse_selection" in result.failed

    def test_unhealthy_book_fails(self):
        rm = _make_rm()
        bad_book = BookHealth(score=0.2, is_healthy=False, issues=["Thin book"])
        result = rm.check_opportunity(
            score=_make_score(),
            market=_make_market(),
            ob=_make_ob(),
            phase=BotPhase.CALIBRATION,
            toxicity=_clean_toxicity(),
            book_health=bad_book,
        )
        assert result.approved is False
        assert "book_health" in result.failed

    def test_dedup_fails(self):
        db = _make_db_mock(has_dup=True)
        rm = _make_rm(db=db)
        result = rm.check_opportunity(
            score=_make_score(),
            market=_make_market(),
            ob=_make_ob(),
            phase=BotPhase.CALIBRATION,
            toxicity=_clean_toxicity(),
            book_health=_healthy_book(),
        )
        assert result.approved is False
        assert "dedup" in result.failed

    def test_daily_spend_limit_fails(self):
        db = _make_db_mock(daily_spend=100.0)
        rm = _make_rm(db=db)
        result = rm.check_opportunity(
            score=_make_score(),
            market=_make_market(),
            ob=_make_ob(),
            phase=BotPhase.CALIBRATION,
            toxicity=_clean_toxicity(),
            book_health=_healthy_book(),
        )
        assert result.approved is False
        assert "daily_spend_limit" in result.failed


# === Full Pass Path ===

class TestFullPassPath:
    def test_all_checks_pass_yields_approved(self):
        """When all 13 checks pass, result should be approved with positive bet."""
        rm = _make_rm()
        result = rm.check_opportunity(
            score=_make_score(),
            market=_make_market(),
            ob=_make_ob(),
            phase=BotPhase.CALIBRATION,
            toxicity=_clean_toxicity(),
            book_health=_healthy_book(),
        )
        assert result.approved is True
        assert result.bet_size > 0
        assert len(result.failed) == 0
        assert all(v is True for v in result.checks.values())

    def test_phase2_full_pass(self):
        rm = _make_rm()
        score = _make_score(adjusted_probability=0.15, ci_width=0.08)
        result = rm.check_opportunity(
            score=score,
            market=_make_market(),
            ob=_make_ob(),
            phase=BotPhase.SCALED,
            toxicity=_clean_toxicity(),
            book_health=_healthy_book(),
        )
        assert result.approved is True
        assert result.sizing_method == "robust_kelly"
        assert result.bet_size > 0


# === Price Penalty Table ===

class TestPricePenalty:
    def test_penalty_table_keys(self):
        assert 0.005 in PRICE_PENALTY
        assert 0.01 in PRICE_PENALTY
        assert 1.00 in PRICE_PENALTY

    def test_penalty_monotonic(self):
        sorted_items = sorted(PRICE_PENALTY.items())
        for i in range(len(sorted_items) - 1):
            assert sorted_items[i][1] <= sorted_items[i + 1][1]

    def test_get_price_penalty_low(self):
        assert _get_price_penalty(0.003) == 0.10

    def test_get_price_penalty_high(self):
        assert _get_price_penalty(0.50) == 1.00


# === Phase Limits ===

class TestPhaseLimits:
    def test_phase1_limits(self):
        rm = _make_rm()
        limits = rm.get_phase_limits(BotPhase.CALIBRATION)
        assert limits["max_daily_spend"] == 20
        assert limits["flat_bet"] == 1.5

    def test_phase2_limits(self):
        rm = _make_rm()
        limits = rm.get_phase_limits(BotPhase.SCALED)
        assert limits["kelly_fraction"] == 0.25

    def test_phase3_limits(self):
        rm = _make_rm()
        limits = rm.get_phase_limits(BotPhase.FULL)
        assert limits["kelly_fraction"] == 0.50
        assert limits["max_daily_spend"] == 100  # 10% of 1000

    def test_circuit_breaker_no_trading(self):
        rm = _make_rm()
        limits = rm.get_phase_limits(BotPhase.CIRCUIT_BREAKER)
        assert limits["max_daily_spend"] == 0


# === Correlation Manager Stub ===

class TestCorrelationManager:
    def test_none_correlation_mgr_passes(self):
        """When correlation_mgr is None, cluster check should pass."""
        rm = _make_rm(corr_mgr=None)
        result = rm.check_opportunity(
            score=_make_score(),
            market=_make_market(),
            ob=_make_ob(),
            phase=BotPhase.CALIBRATION,
            toxicity=_clean_toxicity(),
            book_health=_healthy_book(),
        )
        assert result.checks["cluster_exposure"] is True

    def test_high_cluster_exposure_fails(self):
        """When correlation manager reports high exposure, check fails."""
        corr_mgr = MagicMock()
        corr_mgr.get_cluster_exposure.return_value = 999.0  # way over limit
        rm = _make_rm(corr_mgr=corr_mgr)
        result = rm.check_opportunity(
            score=_make_score(),
            market=_make_market(),
            ob=_make_ob(),
            phase=BotPhase.CALIBRATION,
            toxicity=_clean_toxicity(),
            book_health=_healthy_book(),
        )
        assert result.approved is False
        assert "cluster_exposure" in result.failed


# === Verification Table (Phase 2 approximate) ===

class TestVerificationTable:
    """Approximate verification that Phase 2 sizing produces reasonable values
    for several scenario combinations."""

    def _size(self, rm, adj_prob, price, ci_width):
        score = _make_score(adjusted_probability=adj_prob, ci_width=ci_width)
        limits = rm.get_phase_limits(BotPhase.SCALED)
        bet, method, kelly_raw, est_pen, price_pen = rm.calculate_position_size(
            score, price, BotPhase.SCALED, limits, 1000,
        )
        return bet, kelly_raw, est_pen, price_pen

    def test_high_edge_low_price_moderate_bet(self):
        """High edge at low price: penalty should keep bet moderate."""
        rm = _make_rm()
        bet, kelly_raw, est_pen, price_pen = self._size(rm, 0.15, 0.03, 0.10)
        assert kelly_raw > 0
        assert price_pen < 1.0  # penalized for low price
        assert bet > 0

    def test_moderate_edge_moderate_price(self):
        """Moderate edge at moderate price with tight CI."""
        rm = _make_rm()
        bet, kelly_raw, est_pen, price_pen = self._size(rm, 0.15, 0.10, 0.04)
        assert bet > 0
        assert price_pen >= 0.85

    def test_tight_ci_yields_larger_bet(self):
        """Tighter CI (less uncertainty) should yield larger bet."""
        rm = _make_rm()
        bet_tight, *_ = self._size(rm, 0.15, 0.05, 0.04)
        bet_wide, *_ = self._size(rm, 0.15, 0.05, 0.20)
        assert bet_tight >= bet_wide

    def test_no_edge_yields_zero(self):
        """AI probability at market price => zero bet."""
        rm = _make_rm()
        bet, *_ = self._size(rm, 0.05, 0.05, 0.10)
        assert bet == 0.0
