"""Tests for modules/portfolio_tracker.py — Database initialization, CRUD, and helpers."""

import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from modules.portfolio_tracker import Database


# === TABLE CREATION TESTS ===


class TestDatabaseInit:
    def test_creates_tables(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        conn = sqlite3.connect(tmp_db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = {row[0] for row in cursor.fetchall()}
        expected = {"orders", "positions", "predictions", "daily_stats", "phase_transitions"}
        assert expected.issubset(tables)
        conn.close()

    def test_creates_indexes(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        conn = sqlite3.connect(tmp_db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")
        indexes = {row[0] for row in cursor.fetchall()}
        expected_indexes = {
            "idx_orders_condition",
            "idx_orders_status",
            "idx_orders_date",
            "idx_positions_status",
            "idx_predictions_unresolved",
            "idx_predictions_condition",
            "idx_predictions_phase",
        }
        assert expected_indexes.issubset(indexes)
        conn.close()

    def test_wal_mode(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        row = db.conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"

    def test_creates_data_directory(self, tmp_path):
        db_path = str(tmp_path / "subdir" / "nested" / "test.db")
        db = Database(db_path=db_path)
        assert db.conn is not None

    def test_idempotent_init(self, tmp_db_path):
        """Creating Database twice on same path should not fail."""
        db1 = Database(db_path=tmp_db_path)
        db2 = Database(db_path=tmp_db_path)
        assert db2.conn is not None


# === ORDER CRUD TESTS ===


class TestOrderOperations:
    def _sample_order(self, order_id="ord_001", condition_id="cond_001"):
        return {
            "order_id": order_id,
            "question": "Will X happen?",
            "condition_id": condition_id,
            "token_id": "tok_yes_001",
            "side": "BUY_YES",
            "price": 0.03,
            "size": 50.0,
            "total_cost": 1.50,
            "ai_score_raw": 0.05,
            "ai_score_adjusted": 0.04,
            "ai_confidence": 70,
            "ai_reasoning": "Strong signals",
            "ai_key_factor": "news",
            "p10": 0.02,
            "p50": 0.04,
            "p90": 0.08,
            "toxicity_score": 0.1,
            "book_health_score": 0.8,
            "composite_signal": 0.35,
            "category": "Politics",
            "phase": 1,
            "is_dry_run": 1,
        }

    def test_record_order(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        row_id = db.record_order(self._sample_order())
        assert row_id >= 1

    def test_record_order_returns_incrementing_ids(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        id1 = db.record_order(self._sample_order("ord_001"))
        id2 = db.record_order(self._sample_order("ord_002", "cond_002"))
        assert id2 > id1

    def test_update_order_status(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        db.record_order(self._sample_order())
        db.update_order_status("ord_001", "filled", filled_at="2026-01-01T00:00:00Z")
        orders = db.get_orders_by_condition("cond_001")
        assert len(orders) == 1
        assert orders[0]["status"] == "filled"

    def test_update_order_status_with_pnl(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        db.record_order(self._sample_order())
        db.update_order_status("ord_001", "resolved", pnl=48.50)
        with db.lock:
            row = db.conn.execute("SELECT pnl FROM orders WHERE order_id='ord_001'").fetchone()
        assert row[0] == 48.50

    def test_get_orders_by_condition(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        db.record_order(self._sample_order("ord_001", "cond_001"))
        db.record_order(self._sample_order("ord_002", "cond_001"))
        db.record_order(self._sample_order("ord_003", "cond_002"))
        orders = db.get_orders_by_condition("cond_001")
        assert len(orders) == 2

    def test_get_stale_pending_orders(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        db.record_order(self._sample_order())
        cutoff = datetime.now(timezone.utc) + timedelta(hours=1)
        stale = db.get_stale_pending_orders(cutoff)
        assert len(stale) == 1
        assert stale[0]["order_id"] == "ord_001"

    def test_get_stale_pending_orders_none_stale(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        db.record_order(self._sample_order())
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        stale = db.get_stale_pending_orders(cutoff)
        assert len(stale) == 0


# === POSITION OPERATIONS TESTS ===


class TestPositionOperations:
    def _insert_position(self, db, condition_id="cond_001", status="open"):
        with db.lock:
            db.conn.execute("""
                INSERT INTO positions (condition_id, token_id, market_question, side,
                                       avg_price, shares, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (condition_id, "tok_yes", "Will X happen?", "BUY_YES", 0.03, 50.0, status))
            db.conn.commit()

    def test_has_active_position_true(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        self._insert_position(db)
        assert db.has_active_position("cond_001") is True

    def test_has_active_position_false(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        assert db.has_active_position("cond_999") is False

    def test_has_active_position_via_pending_order(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        order = {
            "order_id": "ord_001", "question": "Q", "condition_id": "cond_001",
            "token_id": "tok_yes", "side": "BUY_YES", "price": 0.03, "size": 50.0,
            "total_cost": 1.50, "ai_score_raw": None, "ai_score_adjusted": None,
            "ai_confidence": None, "ai_reasoning": None, "ai_key_factor": None,
            "p10": None, "p50": None, "p90": None, "toxicity_score": 0,
            "book_health_score": 0, "composite_signal": 0, "category": "Uncategorized",
            "phase": 1, "is_dry_run": 0,
        }
        db.record_order(order)
        assert db.has_active_position("cond_001") is True

    def test_get_open_positions_list(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        self._insert_position(db, "cond_001")
        self._insert_position(db, "cond_002")
        self._insert_position(db, "cond_003", status="closed")
        positions = db.get_open_positions_list()
        assert len(positions) == 2

    def test_get_open_position_count(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        self._insert_position(db, "cond_001")
        self._insert_position(db, "cond_002")
        assert db.get_open_position_count() == 2

    def test_close_position(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        self._insert_position(db)
        db.close_position("cond_001", "profit_target", 10.0)
        positions = db.get_open_positions_list()
        assert len(positions) == 0

    def test_get_category_exposure(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        self._insert_position(db, "cond_001")
        order = {
            "order_id": "ord_001", "question": "Q", "condition_id": "cond_001",
            "token_id": "tok_yes", "side": "BUY_YES", "price": 0.03, "size": 50.0,
            "total_cost": 1.50, "ai_score_raw": None, "ai_score_adjusted": None,
            "ai_confidence": None, "ai_reasoning": None, "ai_key_factor": None,
            "p10": None, "p50": None, "p90": None, "toxicity_score": 0,
            "book_health_score": 0, "composite_signal": 0, "category": "Politics",
            "phase": 1, "is_dry_run": 0,
        }
        db.record_order(order)
        exposure = db.get_category_exposure("Politics")
        assert exposure == 1.50

    def test_get_position_cost(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        order = {
            "order_id": "ord_001", "question": "Q", "condition_id": "cond_001",
            "token_id": "tok_yes", "side": "BUY_YES", "price": 0.03, "size": 50.0,
            "total_cost": 1.50, "ai_score_raw": None, "ai_score_adjusted": None,
            "ai_confidence": None, "ai_reasoning": None, "ai_key_factor": None,
            "p10": None, "p50": None, "p90": None, "toxicity_score": 0,
            "book_health_score": 0, "composite_signal": 0, "category": "Uncategorized",
            "phase": 1, "is_dry_run": 0,
        }
        db.record_order(order)
        cost = db.get_position_cost("cond_001")
        assert cost == 1.50


# === PREDICTION OPERATIONS TESTS ===


class TestPredictionOperations:
    def _sample_prediction(self, condition_id="cond_001"):
        return {
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

    def test_record_prediction(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        row_id = db.record_prediction(self._sample_prediction())
        assert row_id >= 1

    def test_get_unresolved_predictions(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        db.record_prediction(self._sample_prediction("cond_001"))
        db.record_prediction(self._sample_prediction("cond_002"))
        unresolved = db.get_unresolved_predictions()
        assert len(unresolved) == 2

    def test_resolve_prediction(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        db.record_prediction(self._sample_prediction())
        db.resolve_prediction("cond_001", outcome=1, pnl=0.97)
        resolved = db.get_resolved_predictions()
        assert len(resolved) == 1
        assert resolved[0]["outcome"] == 1
        assert resolved[0]["pnl"] == 0.97
        unresolved = db.get_unresolved_predictions()
        assert len(unresolved) == 0

    def test_get_resolved_predictions_empty(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        resolved = db.get_resolved_predictions()
        assert resolved == []


# === PHASE TRANSITION TESTS ===


class TestPhaseTransitions:
    def test_default_phase(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        assert db.get_current_phase() == 1

    def test_record_and_get_phase(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        db.record_phase_transition(1, 2, "cal_score >= 0.6", cal_score=0.65, pred_count=200)
        assert db.get_current_phase() == 2

    def test_multiple_transitions(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        db.record_phase_transition(1, 2, "upgrade", cal_score=0.65)
        db.record_phase_transition(2, 3, "upgrade", cal_score=0.70, sharpe=0.6)
        assert db.get_current_phase() == 3


# === DAILY STATS TESTS ===


class TestDailyStats:
    def test_get_daily_spend_empty(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        assert db.get_daily_spend(date.today()) == 0.0

    def test_get_daily_bet_count_empty(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        assert db.get_daily_bet_count(date.today()) == 0

    def test_update_daily_stats(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        db.update_daily_stats(date.today())
        with db.lock:
            row = db.conn.execute(
                "SELECT * FROM daily_stats WHERE date=?", (date.today().isoformat(),)
            ).fetchone()
        assert row is not None


# === BALANCE & ROLLING METRICS TESTS ===


class TestBalanceAndMetrics:
    def test_default_balance(self, tmp_db_path, monkeypatch):
        monkeypatch.setenv("TOTAL_BANKROLL", "2000")
        db = Database(db_path=tmp_db_path)
        assert db.get_current_balance() == 2000.0

    def test_peak_balance_default(self, tmp_db_path, monkeypatch):
        monkeypatch.setenv("TOTAL_BANKROLL", "1000")
        db = Database(db_path=tmp_db_path)
        assert db.get_peak_balance() == 1000.0

    def test_rolling_pnl_empty(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        assert db.get_rolling_pnl() == 0.0

    def test_consecutive_loss_days_zero(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        assert db.get_consecutive_loss_days() == 0

    def test_consecutive_loss_days(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        with db.lock:
            db.conn.execute(
                "INSERT INTO daily_stats (date, net_pnl, balance, phase) VALUES (?, ?, ?, ?)",
                ("2026-03-13", -5.0, 995.0, 1))
            db.conn.execute(
                "INSERT INTO daily_stats (date, net_pnl, balance, phase) VALUES (?, ?, ?, ?)",
                ("2026-03-12", -3.0, 1000.0, 1))
            db.conn.execute(
                "INSERT INTO daily_stats (date, net_pnl, balance, phase) VALUES (?, ?, ?, ?)",
                ("2026-03-11", 2.0, 1003.0, 1))
            db.conn.commit()
        assert db.get_consecutive_loss_days() == 2

    def test_set_balance_override(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        db.update_daily_stats(date.today())
        db.set_balance_override(1500.0)
        assert db.get_current_balance() == 1500.0
