"""Tests for modules/exit_manager.py — exit rules, retry logic, edge cases."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.exit_manager import ExitManager
from modules.portfolio_tracker import Database
from modules.types import OrderResult


@pytest.fixture
def db(tmp_db_path):
    return Database(db_path=tmp_db_path)


@pytest.fixture
def cfg():
    return {
        "EXIT_PROFIT_MULTIPLIER": 3.0,
        "EXIT_STOP_LOSS_PCT": 0.70,
        "EXIT_TIME_DECAY_HOURS": 24,
        "DRY_RUN": True,
    }


def _make_position(condition_id="cond_001", avg_price=0.03, shares=50.0,
                   token_id="tok_yes", end_date="", market_question="Will X?"):
    return {
        "condition_id": condition_id,
        "token_id": token_id,
        "market_question": market_question,
        "side": "BUY_YES",
        "avg_price": avg_price,
        "shares": shares,
        "status": "open",
        "end_date": end_date,
    }


def _insert_position(db, pos):
    with db.lock:
        db.conn.execute("""
            INSERT INTO positions (condition_id, token_id, market_question, side,
                                   avg_price, shares, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (pos["condition_id"], pos["token_id"], pos["market_question"],
              pos["side"], pos["avg_price"], pos["shares"], pos["status"]))
        db.conn.commit()


class TestProfitTarget:
    @pytest.mark.asyncio
    async def test_profit_target_triggers_exit(self, db, cfg):
        pos = _make_position(avg_price=0.03)
        _insert_position(db, pos)

        scanner = MagicMock()
        scanner.get_current_price = AsyncMock(return_value=0.10)  # 3.33x > 3.0x

        executor = MagicMock()
        executor.place_limit_sell = AsyncMock(return_value=OrderResult(
            success=True, order_id="sell_001", error="", price=0.10, size=50.0,
            side="BUY_NO", is_dry_run=True))

        notifier = MagicMock()
        notifier.exit_alert = AsyncMock()

        mgr = ExitManager(executor, db, scanner, notifier, cfg)
        exits = await mgr.check_exits()

        assert exits == 1
        assert db.get_open_position_count() == 0


class TestStopLoss:
    @pytest.mark.asyncio
    async def test_stop_loss_triggers_exit(self, db, cfg):
        pos = _make_position(avg_price=0.10)
        _insert_position(db, pos)

        scanner = MagicMock()
        scanner.get_current_price = AsyncMock(return_value=0.02)  # 80% loss > 70% threshold

        executor = MagicMock()
        executor.place_limit_sell = AsyncMock(return_value=OrderResult(
            success=True, order_id="sell_002", error="", price=0.02, size=50.0,
            side="BUY_NO", is_dry_run=True))

        notifier = MagicMock()
        notifier.exit_alert = AsyncMock()

        mgr = ExitManager(executor, db, scanner, notifier, cfg)
        exits = await mgr.check_exits()

        assert exits == 1


class TestTimeDecay:
    @pytest.mark.asyncio
    async def test_time_decay_triggers_near_expiry(self, db, cfg):
        """Time decay exits when near expiry and price is stagnant."""
        from datetime import datetime, timezone, timedelta
        end_dt = datetime.now(timezone.utc) + timedelta(hours=12)
        # Use _evaluate_position directly with end_date in dict, since the
        # positions table schema does not have an end_date column.
        pos = _make_position(avg_price=0.03, end_date=end_dt.isoformat())
        _insert_position(db, pos)

        scanner = MagicMock()
        # Price hasn't moved much (change < 50%)
        scanner.get_current_price = AsyncMock(return_value=0.035)

        executor = MagicMock()
        executor.place_limit_sell = AsyncMock(return_value=OrderResult(
            success=True, order_id="sell_003", error="", price=0.035, size=50.0,
            side="BUY_NO", is_dry_run=True))

        notifier = MagicMock()
        notifier.exit_alert = AsyncMock()

        mgr = ExitManager(executor, db, scanner, notifier, cfg)
        # Call _evaluate_position directly with the full dict (including end_date)
        exited = await mgr._evaluate_position(pos)

        assert exited is True


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_failed_sell_retries_once_and_succeeds(self, db, cfg):
        pos = _make_position(avg_price=0.03)
        _insert_position(db, pos)

        scanner = MagicMock()
        scanner.get_current_price = AsyncMock(return_value=0.10)

        fail_result = OrderResult(
            success=False, order_id="", error="timeout", price=0.10, size=50.0,
            side="BUY_NO", is_dry_run=True)
        success_result = OrderResult(
            success=True, order_id="sell_retry", error="", price=0.10, size=50.0,
            side="BUY_NO", is_dry_run=True)

        executor = MagicMock()
        executor.place_limit_sell = AsyncMock(side_effect=[fail_result, success_result])

        notifier = MagicMock()
        notifier.exit_alert = AsyncMock()

        with patch("modules.exit_manager.asyncio.sleep", new_callable=AsyncMock):
            mgr = ExitManager(executor, db, scanner, notifier, cfg)
            exits = await mgr.check_exits()

        assert exits == 1
        assert executor.place_limit_sell.call_count == 2


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_invalid_price_does_not_exit(self, db, cfg):
        pos = _make_position(avg_price=0.03)
        _insert_position(db, pos)

        scanner = MagicMock()
        scanner.get_current_price = AsyncMock(return_value=None)

        executor = MagicMock()
        notifier = MagicMock()

        mgr = ExitManager(executor, db, scanner, notifier, cfg)
        exits = await mgr.check_exits()

        assert exits == 0
        assert db.get_open_position_count() == 1
