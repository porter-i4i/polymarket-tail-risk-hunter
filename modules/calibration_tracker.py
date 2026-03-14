"""
Calibration Tracker: Brier Skill Score with bootstrap CI, phase transition logic.

Computes calibration metrics, manages phase upgrades/downgrades, circuit breaker,
and sanity checks.
"""

import logging
import math
from datetime import date, timedelta

import numpy as np

from modules.types import CalibrationMetrics, BotPhase

logger = logging.getLogger(__name__)

# Bootstrap parameters
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_CI_LOW = 2.5
BOOTSTRAP_CI_HIGH = 97.5


class CalibrationTracker:
    def __init__(self, db, notifier, cfg: dict):
        self.db = db
        self.notifier = notifier
        self.cfg = cfg
        self._latest_metrics: CalibrationMetrics | None = None
        self._consecutive_upgrade_checks = 0

    def calculate_calibration_score(self) -> dict:
        """Calculate full calibration metrics with bootstrap CI.

        Returns dict with all CalibrationMetrics fields.
        """
        predictions = self.db.get_resolved_predictions()

        if len(predictions) < 10:
            metrics = CalibrationMetrics(
                calibration_score=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                ai_brier=1.0,
                market_brier=1.0,
                win_rate_actual=0.0,
                win_rate_expected=0.0,
                win_rate_ratio=0.0,
                sharpe_30d=self._calculate_sharpe_30d(),
                prediction_count=len(predictions),
                phase_recommendation=self.db.get_current_phase(),
            )
            self._latest_metrics = metrics
            return self._metrics_to_dict(metrics)

        # Extract arrays
        outcomes = np.array([p["outcome"] for p in predictions], dtype=float)
        ai_probs = np.array([
            p.get("ai_probability_adjusted") or p.get("ai_probability_raw", 0.5)
            for p in predictions
        ], dtype=float)
        market_probs = np.array([
            p.get("market_price", 0.5) for p in predictions
        ], dtype=float)

        # Point estimates
        ai_brier = float(np.mean((ai_probs - outcomes) ** 2))
        market_brier = float(np.mean((market_probs - outcomes) ** 2))

        # Brier Skill Score: 1 - (ai_brier / market_brier)
        if market_brier > 0:
            bss = 1.0 - (ai_brier / market_brier)
        else:
            bss = 0.0

        # Bootstrap CI
        rng = np.random.default_rng(42)
        n = len(predictions)
        bootstrap_scores = []
        for _ in range(BOOTSTRAP_ITERATIONS):
            idx = rng.integers(0, n, size=n)
            b_outcomes = outcomes[idx]
            b_ai = ai_probs[idx]
            b_market = market_probs[idx]
            b_ai_brier = float(np.mean((b_ai - b_outcomes) ** 2))
            b_market_brier = float(np.mean((b_market - b_outcomes) ** 2))
            if b_market_brier > 0:
                bootstrap_scores.append(1.0 - (b_ai_brier / b_market_brier))
            else:
                bootstrap_scores.append(0.0)

        ci_lower = float(np.percentile(bootstrap_scores, BOOTSTRAP_CI_LOW))
        ci_upper = float(np.percentile(bootstrap_scores, BOOTSTRAP_CI_HIGH))

        # Win rates
        win_rate_actual = float(np.mean(outcomes)) if len(outcomes) > 0 else 0.0
        win_rate_expected = float(np.mean(ai_probs)) if len(ai_probs) > 0 else 0.0
        win_rate_ratio = (win_rate_actual / win_rate_expected) if win_rate_expected > 0 else 0.0

        sharpe = self._calculate_sharpe_30d()

        metrics = CalibrationMetrics(
            calibration_score=bss,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ai_brier=ai_brier,
            market_brier=market_brier,
            win_rate_actual=win_rate_actual,
            win_rate_expected=win_rate_expected,
            win_rate_ratio=win_rate_ratio,
            sharpe_30d=sharpe,
            prediction_count=len(predictions),
            phase_recommendation=self.db.get_current_phase(),
        )
        self._latest_metrics = metrics
        return self._metrics_to_dict(metrics)

    def get_latest_score(self) -> float:
        """Return the latest calibration score, or 0.0 if none."""
        if self._latest_metrics is not None:
            return self._latest_metrics.calibration_score
        return 0.0

    def check_and_transition(self) -> dict | None:
        """Check phase transition conditions. Returns transition dict or None."""
        metrics = self.calculate_calibration_score()
        current_phase = self.db.get_current_phase()
        score = metrics["calibration_score"]
        pred_count = metrics["prediction_count"]
        sharpe = metrics["sharpe_30d"]
        balance = self.db.get_current_balance()
        bankroll = self.cfg.get("TOTAL_BANKROLL", 1000)

        # Priority 1: Circuit breaker
        cb_days = self.cfg.get("CIRCUIT_BREAKER_DAYS", 7)
        cb_loss_pct = self.cfg.get("CIRCUIT_BREAKER_LOSS_PCT", 0.15)
        rolling_pnl = self.db.get_rolling_pnl(cb_days)

        if rolling_pnl < -(bankroll * cb_loss_pct) and current_phase != BotPhase.CIRCUIT_BREAKER:
            return self._do_transition(
                current_phase, BotPhase.CIRCUIT_BREAKER,
                f"Circuit breaker: rolling {cb_days}d PnL ${rolling_pnl:.2f}",
                score, pred_count, sharpe
            )

        # Recovery from circuit breaker
        if current_phase == BotPhase.CIRCUIT_BREAKER:
            if rolling_pnl >= 0:
                return self._do_transition(
                    current_phase, BotPhase.CALIBRATION,
                    "Recovery from circuit breaker: rolling PnL positive",
                    score, pred_count, sharpe
                )
            return None

        # Priority 2: Downgrades
        min_score_p2 = self.cfg.get("MIN_CAL_SCORE_PHASE2", 0.60)
        min_score_p3 = self.cfg.get("MIN_CAL_SCORE_PHASE3", 0.65)

        if current_phase == BotPhase.FULL and score < min_score_p3:
            self._consecutive_upgrade_checks = 0
            return self._do_transition(
                current_phase, BotPhase.SCALED,
                f"Downgrade: cal score {score:.3f} < {min_score_p3}",
                score, pred_count, sharpe
            )

        if current_phase == BotPhase.SCALED and score < min_score_p2:
            self._consecutive_upgrade_checks = 0
            return self._do_transition(
                current_phase, BotPhase.CALIBRATION,
                f"Downgrade: cal score {score:.3f} < {min_score_p2}",
                score, pred_count, sharpe
            )

        # Priority 3: Upgrades (require 3 consecutive qualifying checks)
        min_preds_p2 = self.cfg.get("MIN_PREDICTIONS_PHASE2", 200)
        min_preds_p3 = self.cfg.get("MIN_PREDICTIONS_PHASE3", 500)

        upgrade_target = None
        if (current_phase == BotPhase.CALIBRATION
                and score >= min_score_p2
                and pred_count >= min_preds_p2):
            upgrade_target = BotPhase.SCALED
        elif (current_phase == BotPhase.SCALED
              and score >= min_score_p3
              and pred_count >= min_preds_p3
              and sharpe > 0):
            upgrade_target = BotPhase.FULL

        if upgrade_target is not None:
            self._consecutive_upgrade_checks += 1
            if self._consecutive_upgrade_checks >= 3:
                self._consecutive_upgrade_checks = 0
                return self._do_transition(
                    current_phase, upgrade_target,
                    f"Upgrade after 3 consecutive checks: score={score:.3f}, "
                    f"preds={pred_count}, sharpe={sharpe:.3f}",
                    score, pred_count, sharpe
                )
        else:
            self._consecutive_upgrade_checks = 0

        return None

    def sanity_check(self, scores_today: list) -> bool:
        """Sanity gate for trading decisions.

        Accepts both RawAIScore dataclass instances and plain dicts.

        Returns False (fail) if:
        - buy rate > 40% (too many buys)
        - avg p50 > 15 (predictions too confident)
        """
        if not scores_today:
            return True

        def _get(s, key, default=None):
            """Attribute-safe getter: works on both dicts and dataclasses."""
            if isinstance(s, dict):
                return s.get(key, default)
            return getattr(s, key, default)

        buy_count = sum(
            1 for s in scores_today
            if _get(s, "recommendation", "SKIP") in ("BUY_YES", "BUY_NO")
        )
        buy_rate = buy_count / len(scores_today)
        if buy_rate > 0.40:
            return False

        p50_values = [
            _get(s, "p50", 0)
            for s in scores_today
            if _get(s, "p50") is not None
        ]
        if p50_values:
            avg_p50 = sum(p50_values) / len(p50_values)
            if avg_p50 > 15:
                return False

        return True

    def _calculate_sharpe_30d(self) -> float:
        """Calculate 30-day Sharpe ratio from daily stats (thread-safe via DB API)."""
        try:
            daily_returns = self.db.get_daily_pnl_history(30)

            if len(daily_returns) < 5:
                return 0.0

            mean_return = sum(daily_returns) / len(daily_returns)
            variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
            std_return = math.sqrt(variance) if variance > 0 else 0

            if std_return == 0:
                return 0.0

            return mean_return / std_return

        except Exception as e:
            logger.warning("Sharpe calculation failed: %s", e)
            return 0.0

    def _do_transition(self, from_phase: int, to_phase: int, reason: str,
                       cal_score: float, pred_count: int, sharpe: float) -> dict:
        """Record phase transition and return transition dict."""
        self.db.record_phase_transition(
            from_phase, to_phase, reason,
            cal_score=cal_score, pred_count=pred_count, sharpe=sharpe
        )
        logger.info("Phase transition: %d → %d (%s)", from_phase, to_phase, reason)
        return {
            "from_phase": from_phase,
            "to_phase": to_phase,
            "reason": reason,
            "calibration_score": cal_score,
            "prediction_count": pred_count,
            "sharpe": sharpe,
        }

    @staticmethod
    def _metrics_to_dict(m: CalibrationMetrics) -> dict:
        return {
            "calibration_score": m.calibration_score,
            "ci_lower": m.ci_lower,
            "ci_upper": m.ci_upper,
            "ai_brier": m.ai_brier,
            "market_brier": m.market_brier,
            "win_rate_actual": m.win_rate_actual,
            "win_rate_expected": m.win_rate_expected,
            "win_rate_ratio": m.win_rate_ratio,
            "sharpe_30d": m.sharpe_30d,
            "prediction_count": m.prediction_count,
            "phase_recommendation": m.phase_recommendation,
        }

    def retrain(self) -> None:
        """Retrain calibration model. Called by PositionMonitor after resolution threshold."""
        self.calculate_calibration_score()
        logger.info("Calibration model retrained. Score: %.3f", self.get_latest_score())
