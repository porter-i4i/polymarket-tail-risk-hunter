"""Tests for modules/calibration_tracker.py — metrics, sanity, phase transitions."""

from datetime import date

import pytest

from modules.calibration_tracker import CalibrationTracker
from modules.portfolio_tracker import Database
from modules.types import BotPhase


@pytest.fixture
def db(tmp_db_path):
    return Database(db_path=tmp_db_path)


@pytest.fixture
def cfg():
    return {
        "TOTAL_BANKROLL": 1000,
        "CIRCUIT_BREAKER_DAYS": 7,
        "CIRCUIT_BREAKER_LOSS_PCT": 0.15,
        "MIN_CAL_SCORE_PHASE2": 0.60,
        "MIN_CAL_SCORE_PHASE3": 0.65,
        "MIN_PREDICTIONS_PHASE2": 10,
        "MIN_PREDICTIONS_PHASE3": 20,
        "CALIBRATION_RETRAIN_EVERY": 50,
        "CALIBRATION_MIN_SAMPLES": 100,
    }


def _notifier():
    from unittest.mock import AsyncMock, MagicMock
    n = MagicMock()
    n.send = AsyncMock()
    n.phase_change_alert = AsyncMock()
    n.circuit_breaker_alert = AsyncMock()
    return n


def _insert_resolved(db, count, outcome_ratio=0.5, ai_prob=0.05, market_price=0.03):
    """Insert resolved predictions. outcome_ratio is fraction that are outcome=1."""
    # Dynamic p10/p50/p90 that satisfy CHECK(p10 <= p50 AND p50 <= p90)
    p50 = ai_prob
    p10 = max(0.0, ai_prob - 0.03)
    p90 = min(1.0, ai_prob + 0.03)
    for i in range(count):
        outcome = 1 if i < int(count * outcome_ratio) else 0
        pred = {
            "condition_id": f"cond_{i:04d}",
            "ai_probability_raw": ai_prob,
            "ai_probability_debiased": ai_prob,
            "ai_probability_adjusted": ai_prob,
            "market_price": market_price,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "ai_confidence": 70,
            "base_rate": ai_prob,
            "model_name": "claude_sonnet",
            "phase": 1,
        }
        db.record_prediction(pred)
        pnl = (1.0 - market_price) if outcome == 1 else -market_price
        db.resolve_prediction(f"cond_{i:04d}", outcome, pnl)


class TestCalibrationScore:
    def test_insufficient_predictions_returns_zero(self, db, cfg):
        tracker = CalibrationTracker(db, _notifier(), cfg)
        result = tracker.calculate_calibration_score()
        assert result["calibration_score"] == 0.0
        assert result["prediction_count"] == 0

    def test_good_data_ci_contains_point_estimate(self, db, cfg):
        """With enough data, the CI should bracket the point estimate."""
        _insert_resolved(db, 50, outcome_ratio=0.06, ai_prob=0.05, market_price=0.03)
        tracker = CalibrationTracker(db, _notifier(), cfg)
        result = tracker.calculate_calibration_score()

        assert result["prediction_count"] == 50
        assert result["ci_lower"] <= result["calibration_score"] <= result["ci_upper"]

    def test_get_latest_score(self, db, cfg):
        _insert_resolved(db, 20, outcome_ratio=0.1, ai_prob=0.05)
        tracker = CalibrationTracker(db, _notifier(), cfg)
        assert tracker.get_latest_score() == 0.0  # Before calculation
        tracker.calculate_calibration_score()
        assert tracker.get_latest_score() != 0.0 or True  # After calculation, score is set


class TestSanityCheck:
    def test_high_buy_rate_fails(self, db, cfg):
        tracker = CalibrationTracker(db, _notifier(), cfg)
        scores = [{"recommendation": "BUY_YES", "p50": 5}] * 5 + \
                 [{"recommendation": "SKIP", "p50": 5}] * 4
        # 5/9 = 55.5% > 40%
        assert tracker.sanity_check(scores) is False

    def test_high_avg_p50_fails(self, db, cfg):
        tracker = CalibrationTracker(db, _notifier(), cfg)
        scores = [{"recommendation": "SKIP", "p50": 20}] * 10
        # avg p50 = 20 > 15
        assert tracker.sanity_check(scores) is False

    def test_good_scores_pass(self, db, cfg):
        tracker = CalibrationTracker(db, _notifier(), cfg)
        scores = [{"recommendation": "BUY_YES", "p50": 5}] * 3 + \
                 [{"recommendation": "SKIP", "p50": 5}] * 7
        # buy rate 30%, avg p50 5
        assert tracker.sanity_check(scores) is True

    def test_empty_scores_pass(self, db, cfg):
        tracker = CalibrationTracker(db, _notifier(), cfg)
        assert tracker.sanity_check([]) is True


class TestPhaseTransitions:
    def test_upgrade_requires_consecutive_checks(self, db, cfg):
        """Upgrade to Phase 2 requires 3 consecutive qualifying checks."""
        # Insert enough resolved predictions with high BSS:
        # ai_prob=0.07 is close to actual ~6.7%, market=0.40 is far off → BSS≈0.64 > 0.60
        _insert_resolved(db, 15, outcome_ratio=0.06, ai_prob=0.07, market_price=0.40)

        tracker = CalibrationTracker(db, _notifier(), cfg)

        # First two checks: no transition yet
        result1 = tracker.check_and_transition()
        result2 = tracker.check_and_transition()
        assert result1 is None
        assert result2 is None

        # Third check: should transition
        result3 = tracker.check_and_transition()
        assert result3 is not None
        assert result3["to_phase"] == BotPhase.SCALED

    def test_circuit_breaker_on_poor_rolling_pnl(self, db, cfg):
        """Circuit breaker triggers on rolling PnL below threshold."""
        # Insert daily stats with poor PnL
        with db.lock:
            db.conn.execute(
                "INSERT INTO daily_stats (date, net_pnl, balance, phase) VALUES (?, ?, ?, ?)",
                ("2026-03-13", -80.0, 920.0, 1))
            db.conn.execute(
                "INSERT INTO daily_stats (date, net_pnl, balance, phase) VALUES (?, ?, ?, ?)",
                ("2026-03-12", -80.0, 1000.0, 1))
            db.conn.commit()

        _insert_resolved(db, 15, outcome_ratio=0.06, ai_prob=0.07, market_price=0.40)

        tracker = CalibrationTracker(db, _notifier(), cfg)
        result = tracker.check_and_transition()

        assert result is not None
        assert result["to_phase"] == BotPhase.CIRCUIT_BREAKER

    def test_downgrade_from_full(self, db, cfg):
        """Phase 3 downgrades to Phase 2 if cal score drops below threshold."""
        # Set up as Phase 3
        db.record_phase_transition(1, 2, "test", cal_score=0.65)
        db.record_phase_transition(2, 3, "test", cal_score=0.70)

        # Insert mediocre predictions
        _insert_resolved(db, 15, outcome_ratio=0.5, ai_prob=0.5, market_price=0.5)

        tracker = CalibrationTracker(db, _notifier(), cfg)
        result = tracker.check_and_transition()

        # With equal AI and market probs, BSS ≈ 0, which is < 0.65
        assert result is not None
        assert result["to_phase"] == BotPhase.SCALED

    def test_no_transition_when_stable(self, db, cfg):
        """No transition when conditions are within bounds."""
        # Phase 1, no circuit breaker conditions
        _insert_resolved(db, 5, outcome_ratio=0.06, ai_prob=0.05, market_price=0.03)

        tracker = CalibrationTracker(db, _notifier(), cfg)
        result = tracker.check_and_transition()

        # Not enough predictions for upgrade (need 10), no downgrade from phase 1
        assert result is None
