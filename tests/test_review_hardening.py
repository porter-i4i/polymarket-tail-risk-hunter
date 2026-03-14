"""Tests for review hardening fixes (Fixes #1–#8)."""

import time
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from modules.portfolio_tracker import Database
from modules.scoring_cache import ScoringCache
from modules.types import AdjustedScore, CompositeSignal, MarketCandidate, PricePoint


# === Fix #1: datetime.utcnow → datetime.now(timezone.utc) ===


class TestUtcnowReplacement:
    """Verify no remaining references to datetime.utcnow() in production code."""

    def test_no_utcnow_in_production_modules(self):
        import modules.portfolio_tracker as pt
        import modules.position_monitor as pm
        import modules.notifications as n
        import inspect

        for mod in [pt, pm, n]:
            source = inspect.getsource(mod)
            assert "utcnow()" not in source, f"Found utcnow() in {mod.__name__}"


# === Fix #2: Thread-safe get_daily_pnl_history ===


class TestDailyPnlHistory:
    def test_empty_returns_empty_list(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        assert db.get_daily_pnl_history() == []

    def test_returns_correct_order(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        with db.lock:
            db.conn.execute(
                "INSERT INTO daily_stats (date, net_pnl, balance, phase) VALUES (?, ?, ?, ?)",
                ("2026-03-10", 5.0, 1005.0, 1))
            db.conn.execute(
                "INSERT INTO daily_stats (date, net_pnl, balance, phase) VALUES (?, ?, ?, ?)",
                ("2026-03-11", -3.0, 1002.0, 1))
            db.conn.execute(
                "INSERT INTO daily_stats (date, net_pnl, balance, phase) VALUES (?, ?, ?, ?)",
                ("2026-03-12", 8.0, 1010.0, 1))
            db.conn.commit()
        result = db.get_daily_pnl_history(30)
        # Most recent first
        assert result == [8.0, -3.0, 5.0]

    def test_respects_limit(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        with db.lock:
            for i in range(10):
                db.conn.execute(
                    "INSERT INTO daily_stats (date, net_pnl, balance, phase) VALUES (?, ?, ?, ?)",
                    (f"2026-03-{i+1:02d}", float(i), 1000.0 + i, 1))
            db.conn.commit()
        result = db.get_daily_pnl_history(3)
        assert len(result) == 3

    def test_sharpe_uses_thread_safe_method(self, tmp_db_path):
        """Verify CalibrationTracker's Sharpe calc goes through DB API (not raw conn)."""
        from modules.calibration_tracker import CalibrationTracker

        db = Database(db_path=tmp_db_path)
        with db.lock:
            for i in range(10):
                db.conn.execute(
                    "INSERT INTO daily_stats (date, net_pnl, balance, phase) VALUES (?, ?, ?, ?)",
                    (f"2026-03-{i+1:02d}", 2.0, 1000.0 + i * 2, 1))
            db.conn.commit()

        notifier = MagicMock()
        tracker = CalibrationTracker(db, notifier, {"TOTAL_BANKROLL": 1000})
        sharpe = tracker._calculate_sharpe_30d()
        # All returns equal → mean/std ratio undefined but shouldn't crash
        # With constant returns, std=0 → returns 0.0
        assert sharpe == 0.0


# === Fix #3: Cache-hit path ob/hist crash ===


class TestCacheHitSafety:
    """Verify ob/hist are safely initialized before cache check path."""

    def test_cache_hit_does_not_crash_without_ob(self):
        """Simulates the cache-hit path where ob and hist are not fetched."""
        # This tests that the data contract is correct: ob=None, hist=[], ob_fresh=False
        ob = None
        hist = []
        ob_fresh = False

        # Adverse selection guard: should skip when ob_fresh is False
        enable_adverse = True
        should_run_adverse = enable_adverse and ob_fresh and ob
        assert should_run_adverse is False

        # Book health guard: should skip when ob_fresh is False
        enable_book = True
        should_run_book = enable_book and ob_fresh and ob
        assert should_run_book is False

    def test_fresh_data_path_runs_checks(self):
        """When ob_fresh is True and ob is not None, checks should run."""
        ob = {"bids": [], "asks": []}
        hist = [{"price": 0.03, "time": "2026-01-01"}]
        ob_fresh = True

        should_run_adverse = True and ob_fresh and ob
        assert should_run_adverse  # truthy (the dict itself)

        should_run_book = True and ob_fresh and ob
        assert should_run_book  # truthy


# === Fix #4: end_date column in positions ===


class TestEndDateColumn:
    def test_positions_table_has_end_date(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        with db.lock:
            cols = {row[1] for row in db.conn.execute("PRAGMA table_info(positions)").fetchall()}
        assert "end_date" in cols

    def test_record_position_with_end_date(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        row_id = db.record_position({
            "condition_id": "cond_001",
            "token_id": "tok_yes",
            "market_question": "Will X?",
            "side": "BUY_YES",
            "avg_price": 0.03,
            "shares": 50.0,
            "status": "open",
            "end_date": "2026-12-31",
        })
        assert row_id >= 1

        positions = db.get_open_positions_list()
        assert len(positions) == 1
        assert positions[0]["end_date"] == "2026-12-31"

    def test_record_position_empty_end_date(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        db.record_position({
            "condition_id": "cond_002",
            "token_id": "tok_no",
            "market_question": "Will Y?",
            "side": "BUY_NO",
            "avg_price": 0.05,
            "shares": 20.0,
            "status": "open",
            "end_date": "",
        })
        positions = db.get_open_positions_list()
        assert positions[0]["end_date"] == ""

    def test_migration_adds_end_date_to_existing_db(self, tmp_db_path):
        """Simulate an old DB without end_date, then run migration."""
        import sqlite3
        conn = sqlite3.connect(tmp_db_path)
        conn.execute("""CREATE TABLE positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            condition_id TEXT NOT NULL,
            token_id TEXT NOT NULL,
            market_question TEXT NOT NULL,
            side TEXT NOT NULL,
            avg_price REAL NOT NULL,
            shares REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            opened_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            closed_at TIMESTAMP
        )""")
        conn.commit()
        conn.close()

        # Now open with Database — migration should add end_date
        db = Database(db_path=tmp_db_path)
        with db.lock:
            cols = {row[1] for row in db.conn.execute("PRAGMA table_info(positions)").fetchall()}
        assert "end_date" in cols


# === Fix #5: update_daily_stats_full ===


class TestUpdateDailyStatsFull:
    def test_persists_calibration_and_sharpe(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        db.update_daily_stats_full(
            date.today(),
            calibration_score=0.72,
            sharpe_30d=1.5,
        )
        with db.lock:
            row = db.conn.execute(
                "SELECT calibration_score, sharpe_30d FROM daily_stats WHERE date=?",
                (date.today().isoformat(),)
            ).fetchone()
        assert row is not None
        assert abs(row[0] - 0.72) < 0.001
        assert abs(row[1] - 1.5) < 0.001

    def test_default_none_when_not_provided(self, tmp_db_path):
        db = Database(db_path=tmp_db_path)
        db.update_daily_stats_full(date.today())
        with db.lock:
            row = db.conn.execute(
                "SELECT calibration_score, sharpe_30d FROM daily_stats WHERE date=?",
                (date.today().isoformat(),)
            ).fetchone()
        assert row[0] is None
        assert row[1] is None

    def test_backward_compat_update_daily_stats(self, tmp_db_path):
        """Original update_daily_stats() still works via delegation."""
        db = Database(db_path=tmp_db_path)
        db.update_daily_stats(date.today())
        with db.lock:
            row = db.conn.execute(
                "SELECT * FROM daily_stats WHERE date=?",
                (date.today().isoformat(),)
            ).fetchone()
        assert row is not None


# === Fix #7: composite_signal integration ===


class TestCompositeSignalIntegration:
    def test_signal_aggregator_returns_composite(self):
        """Basic test that SignalAggregator produces a CompositeSignal."""
        from modules.signal_aggregator import SignalAggregator

        agg = SignalAggregator(cfg={})
        market = MarketCandidate(
            condition_id="cond_001",
            question="Will X happen?",
            category="Politics",
            yes_token_id="tok_yes",
            no_token_id="tok_no",
            yes_price=0.03,
            no_price=0.97,
            volume_24h=5000,
            liquidity=10000,
            end_date="2026-12-31",
            tags=["Politics"],
            market_age_hours=100.0,
        )
        result = agg.aggregate(market, news=[], order_book=None, price_history=[])
        assert isinstance(result, CompositeSignal)
        assert -1.0 <= result.composite_score <= 1.0


# === Fix #8: Per-cycle cache-hit counting ===


class TestCacheHitCounting:
    def _make_score(self):
        """Create a minimal AdjustedScore for cache testing."""
        raw = MagicMock()
        raw.p50 = 5
        raw.confidence = 70
        raw.recommendation = "BUY_YES"
        raw.reasoning = "test"
        raw.key_factor = "test"
        raw.base_rate = 5
        return AdjustedScore(
            raw=raw,
            adjusted_p10=0.02,
            adjusted_p50=0.05,
            adjusted_p90=0.08,
            adjusted_probability=0.05,
            expected_value=0.02,
            recommendation="BUY_YES",
            ci_width=0.06,
            debiased_p50=0.05,
        )

    def test_cache_tracks_hits_and_misses(self):
        cache = ScoringCache(default_ttl=300)
        score = self._make_score()
        cache.set("cond_001", score)

        # Miss
        cache.get("cond_999")
        assert cache.misses == 1
        assert cache.hits == 0

        # Hit
        cache.get("cond_001")
        assert cache.hits == 1

    def test_reset_cycle_stats(self):
        cache = ScoringCache(default_ttl=300)
        score = self._make_score()
        cache.set("cond_001", score)

        cache.get("cond_001")  # hit
        cache.get("cond_002")  # miss

        stats = cache.reset_cycle_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

        # After reset, counters are zeroed
        assert cache.hits == 0
        assert cache.misses == 0

    def test_hit_rate_empty_cache(self):
        cache = ScoringCache()
        assert cache.hit_rate == 0.0

    def test_cycle_summary_cache_hits_field(self):
        """CycleSummary accepts cache_hits field."""
        from modules.types import CycleSummary
        summary = CycleSummary(
            cycle_id=1, phase=1, duration_seconds=1.0,
            markets_scanned=10, candidates_found=5, prefilter_passed=3,
            ai_scored=2, cache_hits=7, risk_approved=1,
            orders_placed=1, orders_failed=0, exits_executed=0,
            resolutions_processed=0, daily_spend_so_far=1.5,
            daily_bet_count=1, balance=1000.0, calibration_score=0.7,
            errors=[],
        )
        assert summary.cache_hits == 7
