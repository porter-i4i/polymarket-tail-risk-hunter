"""Tests for main_loop.py — startup, kill conditions, cycle orchestration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from main_loop import MainLoop, KILL_CONDITIONS
from modules.types import BotPhase


def _make_loop() -> MainLoop:
    """Create a MainLoop with mocked DB and modules for unit testing."""
    ml = MainLoop()
    ml.cfg = {
        "DRY_RUN": True,
        "TOTAL_BANKROLL": 1000,
        "CYCLE_INTERVAL_SECONDS": 60,
        "ENABLE_EXIT_MANAGER": False,
        "ENABLE_CORRELATION_CLUSTERS": False,
        "ENABLE_DEBIASER": False,
        "ENABLE_PLATT_SCALING": False,
        "ENABLE_ADVERSE_SELECTION": False,
        "ENABLE_ORDER_BOOK_HEALTH": False,
        "ENABLE_FEE_CHECK": False,
    }
    ml.dry_run = True
    ml.db = MagicMock()
    ml.db.get_current_balance.return_value = 900.0
    ml.db.get_peak_balance.return_value = 1000.0
    ml.db.get_consecutive_loss_days.return_value = 0
    ml.db.get_resolved_predictions.return_value = []
    ml.db.get_current_phase.return_value = BotPhase.CALIBRATION.value
    ml.db.has_active_position.return_value = False
    ml.db.get_daily_bet_count.return_value = 0
    ml.db.get_daily_spend.return_value = 0.0

    ml.ai = MagicMock()
    ml.ai.error_rate = 0.0

    ml.executor = AsyncMock()
    ml.notifier = AsyncMock()
    ml.notifier.send = AsyncMock()
    ml.notifier.kill_switch_alert = AsyncMock()
    ml.notifier.phase_change_alert = AsyncMock()
    ml.notifier.sanity_fail_alert = AsyncMock()
    ml.notifier.trade_alert = AsyncMock()
    ml.notifier.startup_alert = AsyncMock()
    ml.notifier.shutdown_alert = AsyncMock()

    ml.calibration = MagicMock()
    ml.calibration.check_and_transition.return_value = None
    ml.calibration.get_latest_score.return_value = 0.5
    ml.calibration.sanity_check.return_value = True

    ml.scanner = AsyncMock()
    ml.news = MagicMock()
    ml.scoring_cache = MagicMock()
    ml.scoring_cache.get.return_value = None
    ml.scoring_cache.cleanup.return_value = 0

    ml.signal_agg = MagicMock()
    ml.risk = MagicMock()
    ml.exit_mgr = AsyncMock()
    ml.monitor = AsyncMock()
    ml.monitor.run_checks = AsyncMock(return_value=0)
    ml.debiaser = MagicMock()
    ml.calibrator = MagicMock()
    ml.adverse = MagicMock()
    ml.book_checker = MagicMock()
    ml.fee_calc = MagicMock()
    ml.correlation_mgr = MagicMock()

    return ml


# === KILL CONDITIONS ===


class TestKillConditions:
    def test_no_kill_returns_none(self):
        ml = _make_loop()
        assert ml._check_kill_conditions() is None

    def test_drawdown_kills(self):
        ml = _make_loop()
        ml.db.get_current_balance.return_value = 500.0
        ml.db.get_peak_balance.return_value = 1000.0
        reason = ml._check_kill_conditions()
        assert reason is not None
        assert "Drawdown" in reason

    def test_absolute_loss_kills(self):
        ml = _make_loop()
        ml.db.get_current_balance.return_value = 400.0  # loss = 600 >= 500
        reason = ml._check_kill_conditions()
        assert reason is not None

    def test_consecutive_loss_days_kills(self):
        ml = _make_loop()
        ml.db.get_consecutive_loss_days.return_value = 15
        reason = ml._check_kill_conditions()
        assert reason is not None
        assert "loss days" in reason

    def test_ai_error_rate_kills(self):
        ml = _make_loop()
        ml.ai.error_rate = 0.5
        reason = ml._check_kill_conditions()
        assert reason is not None
        assert "error rate" in reason

    def test_cycle_gap_kills(self):
        ml = _make_loop()
        ml.last_cycle_time = 1.0  # far in the past
        reason = ml._check_kill_conditions()
        assert reason is not None
        assert "Cycle gap" in reason


# === PREFLIGHT ===


class TestPreflight:
    @pytest.mark.asyncio
    async def test_dry_run_always_passes(self):
        ml = _make_loop()
        ml.dry_run = True
        assert await ml._preflight_checks() is True

    @pytest.mark.asyncio
    async def test_live_delegates_to_executor(self):
        ml = _make_loop()
        ml.dry_run = False
        ml.executor.preflight_check = AsyncMock(return_value=True)
        assert await ml._preflight_checks() is True

    @pytest.mark.asyncio
    async def test_live_fails_on_executor_fail(self):
        ml = _make_loop()
        ml.dry_run = False
        ml.executor.preflight_check = AsyncMock(return_value=False)
        assert await ml._preflight_checks() is False


# === RECONCILIATION ===


class TestReconciliation:
    @pytest.mark.asyncio
    async def test_dry_run_always_true(self):
        ml = _make_loop()
        ml.dry_run = True
        assert await ml._reconcile_state() is True

    @pytest.mark.asyncio
    async def test_large_mismatch_kills(self):
        ml = _make_loop()
        ml.dry_run = False
        ml.executor.get_balance = AsyncMock(return_value=800.0)
        ml.db.get_current_balance.return_value = 900.0
        # diff=100 > 50 kill threshold
        result = await ml._reconcile_state()
        assert result is False

    @pytest.mark.asyncio
    async def test_small_mismatch_reconciles(self):
        ml = _make_loop()
        ml.dry_run = False
        ml.executor.get_balance = AsyncMock(return_value=895.0)
        ml.db.get_current_balance.return_value = 900.0
        # diff=5 <= 50 but > 5 → reconcile
        # Actually diff=5 is not > 5 (it's exactly 5, not >5)
        # The code checks "if diff > 5.0" so 5.0 does not trigger warning
        result = await ml._reconcile_state()
        assert result is True


# === ENTER KILLED STATE ===


class TestEnterKilledState:
    @pytest.mark.asyncio
    async def test_sets_running_false(self):
        ml = _make_loop()
        ml.running = True
        await ml._enter_killed_state("test reason")
        assert ml.running is False
        ml.notifier.kill_switch_alert.assert_called_once_with("test reason")


# === CYCLE: CIRCUIT BREAKER ===


class TestCycleCircuitBreaker:
    @pytest.mark.asyncio
    async def test_circuit_breaker_skips_scanning(self):
        ml = _make_loop()
        ml.db.get_current_phase.return_value = BotPhase.CIRCUIT_BREAKER.value
        ml.cfg["ENABLE_EXIT_MANAGER"] = True
        ml.exit_mgr.check_exits = AsyncMock(return_value=0)
        ml.cycle_counter = 1
        await ml._cycle()
        # Should NOT have called scan_all_markets
        ml.scanner.scan_all_markets.assert_not_called()
        # Should have called exit check and monitor
        ml.exit_mgr.check_exits.assert_called_once()
        ml.monitor.run_checks.assert_called_once()


# === CYCLE: EMPTY SCAN ===


class TestCycleEmptyScan:
    @pytest.mark.asyncio
    async def test_no_candidates_no_orders(self):
        ml = _make_loop()
        ml.cycle_counter = 1

        from modules.types import ScanResult
        ml.scanner.scan_all_markets = AsyncMock(return_value=ScanResult(
            candidates=[], total_scanned=100, total_filtered=100,
            scan_duration_seconds=1.0, errors=[],
        ))
        ml.risk.get_phase_limits.return_value = {"max_bets_per_day": 10}
        await ml._cycle()
        ml.db.record_order.assert_not_called()


# === KILL CONDITIONS CONSTANTS ===


class TestKillConditionsConstants:
    def test_drawdown_threshold(self):
        assert KILL_CONDITIONS["drawdown_from_peak_pct"] == 0.40

    def test_absolute_loss(self):
        assert KILL_CONDITIONS["total_loss_absolute_usd"] == 500

    def test_consecutive_days(self):
        assert KILL_CONDITIONS["consecutive_loss_days"] == 14

    def test_heartbeat_gap(self):
        assert KILL_CONDITIONS["heartbeat_gap_seconds"] == 300

    def test_cycle_gap(self):
        assert KILL_CONDITIONS["cycle_gap_seconds"] == 900
