"""Tests for modules/position_monitor.py — resolution, stale cleanup, retrain."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from modules.position_monitor import PositionMonitor
from modules.portfolio_tracker import Database


@pytest.fixture
def db(tmp_db_path):
    return Database(db_path=tmp_db_path)


@pytest.fixture
def cfg():
    return {
        "CALIBRATION_RETRAIN_EVERY": 2,
        "CALIBRATION_MIN_SAMPLES": 3,
    }


def _insert_prediction(db, condition_id="cond_001", outcome=None):
    pred = {
        "condition_id": condition_id,
        "ai_probability_raw": 0.05,
        "ai_probability_debiased": 0.04,
        "ai_probability_adjusted": 0.035,
        "market_price": 0.03,
        "p10": 0.02,
        "p50": 0.04,
        "p90": 0.08,
        "ai_confidence": 70,
        "base_rate": 0.05,
        "model_name": "claude_sonnet",
        "phase": 1,
    }
    db.record_prediction(pred)
    if outcome is not None:
        db.resolve_prediction(condition_id, outcome, 0.0)


def _insert_position(db, condition_id="cond_001"):
    with db.lock:
        db.conn.execute("""
            INSERT INTO positions (condition_id, token_id, market_question, side,
                                   avg_price, shares, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (condition_id, "tok_yes", "Will X?", "BUY_YES", 0.03, 50.0, "open"))
        db.conn.commit()


def _insert_order(db, order_id="ord_001", condition_id="cond_001", status="filled",
                  created_at=None):
    # Use space-separated datetime format to match get_stale_pending_orders cutoff format
    ts = created_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with db.lock:
        db.conn.execute("""
            INSERT INTO orders (order_id, market_question, condition_id, token_id, side,
                               price, size, total_cost, category, phase, is_dry_run, status,
                               created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (order_id, "Will X?", condition_id, "tok_yes", "BUY_YES",
              0.03, 50.0, 1.50, "Uncategorized", 1, 1, status, ts))
        db.conn.commit()


class TestCheckResolutions:
    @pytest.mark.asyncio
    async def test_resolved_yes_updates_flow(self, db, cfg):
        """Resolved YES market updates prediction, order, and position."""
        _insert_prediction(db, "cond_001")
        _insert_position(db, "cond_001")
        _insert_order(db, "ord_001", "cond_001")

        scanner = MagicMock()
        scanner.get_market_by_condition = AsyncMock(return_value={
            "resolved": True,
            "outcomePrices": json.dumps(["1.0", "0.0"]),
        })

        calibrator = MagicMock()
        calibrator.retrain = MagicMock()
        notifier = MagicMock()

        monitor = PositionMonitor(db, scanner, calibrator, notifier, cfg)
        count = await monitor.run_checks()

        assert count == 1
        assert len(db.get_unresolved_predictions()) == 0
        resolved = db.get_resolved_predictions()
        assert len(resolved) == 1
        assert resolved[0]["outcome"] == 1

    @pytest.mark.asyncio
    async def test_unresolved_market_skipped(self, db, cfg):
        """Unresolved market should not update prediction."""
        _insert_prediction(db, "cond_001")

        scanner = MagicMock()
        scanner.get_market_by_condition = AsyncMock(return_value={
            "resolved": False,
            "outcomePrices": json.dumps(["0.5", "0.5"]),
        })

        calibrator = MagicMock()
        notifier = MagicMock()

        monitor = PositionMonitor(db, scanner, calibrator, notifier, cfg)
        count = await monitor.run_checks()

        assert count == 0
        assert len(db.get_unresolved_predictions()) == 1


class TestStaleOrderCleanup:
    @pytest.mark.asyncio
    async def test_stale_pending_cleanup(self, db, cfg):
        """Stale pending orders should be cancelled."""
        stale_time = (datetime.now(timezone.utc) - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
        _insert_order(db, "ord_stale", "cond_001", status="pending", created_at=stale_time)

        scanner = MagicMock()
        scanner.get_market_by_condition = AsyncMock(return_value=None)
        scanner.executor = MagicMock()
        scanner.executor.cancel_order = AsyncMock(return_value=True)

        calibrator = MagicMock()
        notifier = MagicMock()

        monitor = PositionMonitor(db, scanner, calibrator, notifier, cfg)
        await monitor.run_checks()

        scanner.executor.cancel_order.assert_called_once_with("ord_stale")
        # Order status updated to 'cancelled' — get_orders_by_condition only returns
        # pending/filled, so verify via direct query
        with db.lock:
            row = db.conn.execute(
                "SELECT status FROM orders WHERE order_id='ord_stale'"
            ).fetchone()
        assert row is not None
        assert row[0] == "cancelled"


class TestRetrainTrigger:
    @pytest.mark.asyncio
    async def test_retrain_triggered_after_threshold(self, db, cfg):
        """Retrain is called after enough resolutions with enough samples."""
        # Pre-populate enough resolved predictions so min_samples (3) is met
        for i in range(3):
            _insert_prediction(db, f"cond_pre_{i}", outcome=1)

        # Now add 2 unresolved that will resolve (retrain_every=2)
        _insert_prediction(db, "cond_new_1")
        _insert_prediction(db, "cond_new_2")

        scanner = MagicMock()
        scanner.get_market_by_condition = AsyncMock(return_value={
            "resolved": True,
            "outcomePrices": json.dumps(["1.0", "0.0"]),
        })

        calibrator = MagicMock()
        calibrator.retrain = MagicMock()
        notifier = MagicMock()

        monitor = PositionMonitor(db, scanner, calibrator, notifier, cfg)
        count = await monitor.run_checks()

        assert count == 2
        calibrator.retrain.assert_called_once()
