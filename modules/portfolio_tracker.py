"""
Portfolio tracker: SQLite database for orders, positions, predictions, stats, phase transitions.
Thread-safe with threading.Lock, WAL mode, busy_timeout=5000.
"""
import os
import sqlite3
import threading
from datetime import date, datetime, timezone


class Database:
    def __init__(self, db_path: str = "data/polybot.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        # Ensure the data directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_tables()
        self._migrate()

    def _migrate(self):
        """Apply backward-compatible schema migrations for existing databases."""
        with self.lock:
            # Add end_date column to positions if missing (Fix #4)
            cols = {row[1] for row in self.conn.execute("PRAGMA table_info(positions)").fetchall()}
            if "end_date" not in cols:
                self.conn.execute("ALTER TABLE positions ADD COLUMN end_date TEXT DEFAULT ''")
                self.conn.commit()

    def _init_tables(self):
        """Execute EXACT DDL from DATABASE SCHEMA section."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE NOT NULL,
                    market_question TEXT NOT NULL,
                    condition_id TEXT NOT NULL,
                    token_id TEXT NOT NULL,
                    side TEXT NOT NULL CHECK(side IN ('BUY_YES', 'BUY_NO')),
                    price REAL NOT NULL CHECK(price BETWEEN 0.001 AND 0.999),
                    size REAL NOT NULL CHECK(size > 0),
                    total_cost REAL NOT NULL CHECK(total_cost >= 0),
                    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','filled','cancelled','resolved')),
                    ai_score_raw REAL,
                    ai_score_adjusted REAL,
                    ai_confidence INTEGER CHECK(ai_confidence BETWEEN 0 AND 100),
                    ai_reasoning TEXT,
                    ai_key_factor TEXT,
                    p10 REAL CHECK(p10 BETWEEN 0 AND 1),
                    p50 REAL CHECK(p50 BETWEEN 0 AND 1),
                    p90 REAL CHECK(p90 BETWEEN 0 AND 1),
                    toxicity_score REAL DEFAULT 0 CHECK(toxicity_score BETWEEN 0 AND 1),
                    book_health_score REAL DEFAULT 0 CHECK(book_health_score BETWEEN 0 AND 1),
                    composite_signal REAL DEFAULT 0,
                    category TEXT NOT NULL DEFAULT 'Uncategorized',
                    phase INTEGER NOT NULL CHECK(phase BETWEEN 1 AND 5),
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    filled_at TIMESTAMP,
                    resolved_at TIMESTAMP,
                    pnl REAL DEFAULT 0,
                    is_dry_run INTEGER NOT NULL DEFAULT 0 CHECK(is_dry_run IN (0, 1)),
                    CONSTRAINT valid_percentiles CHECK(p10 IS NULL OR (p10 <= p50 AND p50 <= p90))
                );

                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    condition_id TEXT NOT NULL,
                    token_id TEXT NOT NULL,
                    market_question TEXT NOT NULL,
                    side TEXT NOT NULL CHECK(side IN ('BUY_YES', 'BUY_NO')),
                    avg_price REAL NOT NULL CHECK(avg_price BETWEEN 0.001 AND 0.999),
                    shares REAL NOT NULL CHECK(shares > 0),
                    current_price REAL,
                    unrealized_pnl REAL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','closed','exited')),
                    exit_reason TEXT,
                    opened_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    closed_at TIMESTAMP,
                    end_date TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    condition_id TEXT NOT NULL,
                    ai_probability_raw REAL NOT NULL CHECK(ai_probability_raw BETWEEN 0 AND 1),
                    ai_probability_debiased REAL CHECK(ai_probability_debiased BETWEEN 0 AND 1),
                    ai_probability_adjusted REAL CHECK(ai_probability_adjusted BETWEEN 0 AND 1),
                    market_price REAL NOT NULL CHECK(market_price BETWEEN 0 AND 1),
                    p10 REAL CHECK(p10 BETWEEN 0 AND 1),
                    p50 REAL CHECK(p50 BETWEEN 0 AND 1),
                    p90 REAL CHECK(p90 BETWEEN 0 AND 1),
                    ai_confidence INTEGER CHECK(ai_confidence BETWEEN 0 AND 100),
                    base_rate REAL CHECK(base_rate BETWEEN 0 AND 1),
                    model_name TEXT NOT NULL DEFAULT 'claude_sonnet',
                    phase INTEGER NOT NULL CHECK(phase BETWEEN 1 AND 5),
                    scored_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    outcome INTEGER CHECK(outcome IS NULL OR outcome IN (0, 1)),
                    pnl REAL,
                    CONSTRAINT valid_pred_percentiles CHECK(p10 IS NULL OR (p10 <= p50 AND p50 <= p90))
                );

                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    total_bets INTEGER NOT NULL DEFAULT 0,
                    total_spent REAL NOT NULL DEFAULT 0,
                    total_won REAL NOT NULL DEFAULT 0,
                    total_lost REAL NOT NULL DEFAULT 0,
                    net_pnl REAL NOT NULL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    balance REAL NOT NULL DEFAULT 0,
                    phase INTEGER NOT NULL DEFAULT 1,
                    calibration_score REAL,
                    sharpe_30d REAL,
                    exits_count INTEGER NOT NULL DEFAULT 0,
                    exits_pnl REAL NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS phase_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    from_phase INTEGER NOT NULL,
                    to_phase INTEGER NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    calibration_score REAL,
                    prediction_count INTEGER,
                    sharpe REAL,
                    balance REAL
                );

                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_orders_condition ON orders(condition_id);
                CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
                CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(created_at);
                CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
                CREATE INDEX IF NOT EXISTS idx_predictions_unresolved ON predictions(outcome) WHERE outcome IS NULL;
                CREATE INDEX IF NOT EXISTS idx_predictions_condition ON predictions(condition_id);
                CREATE INDEX IF NOT EXISTS idx_predictions_phase ON predictions(phase);
            """)
            self.conn.commit()

    # === ORDER OPERATIONS ===

    def record_order(self, order: dict) -> int:
        """Insert new order. Returns row id."""
        with self.lock:
            cursor = self.conn.execute("""
                INSERT INTO orders (
                    order_id, market_question, condition_id, token_id, side,
                    price, size, total_cost, ai_score_raw, ai_score_adjusted,
                    ai_confidence, ai_reasoning, ai_key_factor,
                    p10, p50, p90, toxicity_score, book_health_score,
                    composite_signal, category, phase, is_dry_run
                ) VALUES (
                    :order_id, :question, :condition_id, :token_id, :side,
                    :price, :size, :total_cost, :ai_score_raw, :ai_score_adjusted,
                    :ai_confidence, :ai_reasoning, :ai_key_factor,
                    :p10, :p50, :p90, :toxicity_score, :book_health_score,
                    :composite_signal, :category, :phase, :is_dry_run
                )
            """, order)
            self.conn.commit()
            return cursor.lastrowid

    def update_order_status(self, order_id: str, status: str,
                            filled_at: str = None, pnl: float = None) -> None:
        with self.lock:
            self.conn.execute("""
                UPDATE orders SET status=?, filled_at=?, pnl=?
                WHERE order_id=?
            """, (status, filled_at, pnl, order_id))
            self.conn.commit()

    # === POSITION OPERATIONS ===

    def record_position(self, pos: dict) -> int:
        """Insert new position. Returns row id."""
        with self.lock:
            cursor = self.conn.execute("""
                INSERT INTO positions (
                    condition_id, token_id, market_question, side,
                    avg_price, shares, status, end_date
                ) VALUES (
                    :condition_id, :token_id, :market_question, :side,
                    :avg_price, :shares, :status, :end_date
                )
            """, pos)
            self.conn.commit()
            return cursor.lastrowid

    def has_active_position(self, condition_id: str) -> bool:
        """Check if we already have an open/pending position on this market."""
        with self.lock:
            row = self.conn.execute("""
                SELECT 1 FROM positions WHERE condition_id=? AND status='open'
                UNION
                SELECT 1 FROM orders WHERE condition_id=? AND status IN ('pending','filled')
            """, (condition_id, condition_id)).fetchone()
            return row is not None

    def get_open_positions_list(self) -> list[dict]:
        """Return all open positions as list of dicts."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM positions WHERE status='open'"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_open_position_count(self) -> int:
        with self.lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM positions WHERE status='open'"
            ).fetchone()
            return row[0] if row else 0

    def close_position(self, condition_id: str, exit_reason: str, pnl: float) -> None:
        with self.lock:
            now = datetime.now(timezone.utc).isoformat()
            self.conn.execute("""
                UPDATE positions SET status='closed', exit_reason=?,
                       closed_at=?, unrealized_pnl=?
                WHERE condition_id=? AND status='open'
            """, (exit_reason, now, pnl, condition_id))
            self.conn.commit()

    def get_category_exposure(self, category: str) -> float:
        """Total cost of open positions in given category."""
        with self.lock:
            row = self.conn.execute("""
                SELECT COALESCE(SUM(o.total_cost), 0) FROM orders o
                JOIN positions p ON o.condition_id = p.condition_id
                WHERE o.category=? AND p.status='open'
            """, (category,)).fetchone()
            return float(row[0]) if row else 0.0

    def get_position_cost(self, condition_id: str) -> float:
        """Total cost of a specific position."""
        with self.lock:
            row = self.conn.execute("""
                SELECT COALESCE(SUM(total_cost), 0) FROM orders
                WHERE condition_id=? AND status IN ('pending','filled')
            """, (condition_id,)).fetchone()
            return float(row[0]) if row else 0.0

    # === PREDICTION OPERATIONS ===

    def record_prediction(self, pred: dict) -> int:
        with self.lock:
            cursor = self.conn.execute("""
                INSERT INTO predictions (
                    condition_id, ai_probability_raw, ai_probability_debiased,
                    ai_probability_adjusted, market_price,
                    p10, p50, p90, ai_confidence, base_rate, model_name, phase
                ) VALUES (
                    :condition_id, :ai_probability_raw, :ai_probability_debiased,
                    :ai_probability_adjusted, :market_price,
                    :p10, :p50, :p90, :ai_confidence, :base_rate, :model_name, :phase
                )
            """, pred)
            self.conn.commit()
            return cursor.lastrowid

    def get_resolved_predictions(self, min_count: int = 0) -> list:
        """Return predictions where outcome is known."""
        with self.lock:
            rows = self.conn.execute("""
                SELECT * FROM predictions WHERE outcome IS NOT NULL
                ORDER BY scored_at DESC
            """).fetchall()
            return [dict(r) for r in rows]

    def get_unresolved_predictions(self) -> list:
        with self.lock:
            rows = self.conn.execute("""
                SELECT * FROM predictions WHERE outcome IS NULL
            """).fetchall()
            return [dict(r) for r in rows]

    def resolve_prediction(self, condition_id: str, outcome: int, pnl: float) -> None:
        with self.lock:
            now = datetime.now(timezone.utc).isoformat()
            self.conn.execute("""
                UPDATE predictions SET outcome=?, resolved_at=?, pnl=?
                WHERE condition_id=? AND outcome IS NULL
            """, (outcome, now, pnl, condition_id))
            self.conn.commit()

    # === DAILY STATS ===

    def get_daily_spend(self, today: date) -> float:
        with self.lock:
            row = self.conn.execute("""
                SELECT COALESCE(SUM(total_cost), 0) FROM orders
                WHERE DATE(created_at)=? AND status != 'cancelled'
            """, (today.isoformat(),)).fetchone()
            return float(row[0]) if row else 0.0

    def get_daily_bet_count(self, today: date) -> int:
        with self.lock:
            row = self.conn.execute("""
                SELECT COUNT(*) FROM orders
                WHERE DATE(created_at)=? AND status != 'cancelled'
            """, (today.isoformat(),)).fetchone()
            return row[0] if row else 0

    def update_daily_stats(self, today: date) -> None:
        """Upsert daily_stats row. Called end of each cycle."""
        self.update_daily_stats_full(today)

    def update_daily_stats_full(self, today: date,
                                calibration_score: float = None,
                                sharpe_30d: float = None) -> None:
        """Upsert daily_stats row with optional calibration score and Sharpe."""
        with self.lock:
            stats = self.conn.execute("""
                SELECT
                    COUNT(*) as total_bets,
                    COALESCE(SUM(total_cost), 0) as total_spent,
                    COALESCE(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 0) as total_won,
                    COALESCE(SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END), 0) as total_lost,
                    COALESCE(SUM(pnl), 0) as net_pnl
                FROM orders WHERE DATE(created_at)=?
            """, (today.isoformat(),)).fetchone()

            self.conn.execute("""
                INSERT OR REPLACE INTO daily_stats
                    (date, total_bets, total_spent, total_won, total_lost, net_pnl,
                     balance, phase, calibration_score, sharpe_30d)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (today.isoformat(), stats[0], stats[1], stats[2], stats[3],
                  stats[4], self.get_current_balance_unlocked(),
                  self.get_current_phase_unlocked(),
                  calibration_score, sharpe_30d))
            self.conn.commit()

    # === BALANCE & PHASE ===

    def get_current_balance(self) -> float:
        with self.lock:
            return self.get_current_balance_unlocked()

    def get_current_balance_unlocked(self) -> float:
        """Call only when lock is already held."""
        row = self.conn.execute(
            "SELECT balance FROM daily_stats ORDER BY date DESC LIMIT 1"
        ).fetchone()
        return float(row[0]) if row else float(os.getenv("TOTAL_BANKROLL", "1000"))

    def set_balance_override(self, balance: float) -> None:
        """Force balance to on-chain value (reconciliation)."""
        with self.lock:
            today = date.today().isoformat()
            self.conn.execute(
                "UPDATE daily_stats SET balance=? WHERE date=?", (balance, today))
            self.conn.commit()

    def get_current_phase(self) -> int:
        with self.lock:
            return self.get_current_phase_unlocked()

    def get_current_phase_unlocked(self) -> int:
        row = self.conn.execute(
            "SELECT to_phase FROM phase_transitions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else 1  # Default Phase 1

    def record_phase_transition(self, from_phase: int, to_phase: int,
                                reason: str, cal_score: float = None,
                                pred_count: int = None, sharpe: float = None) -> None:
        with self.lock:
            self.conn.execute("""
                INSERT INTO phase_transitions
                    (from_phase, to_phase, trigger_reason, calibration_score,
                     prediction_count, sharpe, balance)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (from_phase, to_phase, reason, cal_score, pred_count, sharpe,
                  self.get_current_balance_unlocked()))
            self.conn.commit()

    # === ROLLING METRICS ===

    def get_rolling_pnl(self, days: int = 7) -> float:
        """Sum of PnL over last N days."""
        with self.lock:
            row = self.conn.execute("""
                SELECT COALESCE(SUM(net_pnl), 0) FROM daily_stats
                WHERE date >= DATE('now', ?)
            """, (f"-{days} days",)).fetchone()
            return float(row[0]) if row else 0.0

    def get_peak_balance(self) -> float:
        with self.lock:
            row = self.conn.execute(
                "SELECT MAX(balance) FROM daily_stats"
            ).fetchone()
            return float(row[0]) if row and row[0] is not None else float(os.getenv("TOTAL_BANKROLL", "1000"))

    def get_consecutive_loss_days(self) -> int:
        with self.lock:
            rows = self.conn.execute(
                "SELECT net_pnl FROM daily_stats ORDER BY date DESC LIMIT 30"
            ).fetchall()
            count = 0
            for row in rows:
                if row[0] < 0:
                    count += 1
                else:
                    break
            return count

    def get_daily_pnl_history(self, days: int = 30) -> list[float]:
        """Return list of daily net_pnl values (most recent first), thread-safe."""
        with self.lock:
            rows = self.conn.execute("""
                SELECT net_pnl FROM daily_stats
                ORDER BY date DESC LIMIT ?
            """, (days,)).fetchall()
            return [float(r[0]) for r in rows]

    # === ADDITIONAL QUERY METHODS ===

    def get_orders_by_condition(self, condition_id: str) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM orders WHERE condition_id=? AND status IN ('pending','filled')",
                (condition_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_stale_pending_orders(self, cutoff: datetime) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM orders WHERE status='pending' AND created_at < ?",
                (cutoff.strftime("%Y-%m-%d %H:%M:%S"),)
            ).fetchall()
            return [dict(r) for r in rows]

    # === DASHBOARD QUERY HELPERS ===

    def get_recent_orders(self, limit: int = 50) -> list[dict]:
        """Return the most recent orders (all statuses) for dashboard display."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM orders ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_predictions(self, limit: int = 50) -> list[dict]:
        """Return the most recent predictions for dashboard display."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM predictions ORDER BY scored_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_daily_stats_history(self, limit: int = 90) -> list[dict]:
        """Return daily_stats rows ordered by date descending for dashboard charts."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM daily_stats ORDER BY date DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_phase_transitions_history(self, limit: int = 30) -> list[dict]:
        """Return phase transition history for dashboard display."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM phase_transitions ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_closed_positions(self, limit: int = 50) -> list[dict]:
        """Return recently closed positions for PnL review."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM positions WHERE status IN ('closed','exited') "
                "ORDER BY closed_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_total_realized_pnl(self) -> float:
        """Sum of PnL from all resolved orders."""
        with self.lock:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM orders WHERE pnl IS NOT NULL"
            ).fetchone()
            return float(row[0]) if row else 0.0

    def get_total_unrealized_pnl(self) -> float:
        """Sum of unrealized PnL from open positions."""
        with self.lock:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(unrealized_pnl), 0) FROM positions WHERE status='open'"
            ).fetchone()
            return float(row[0]) if row else 0.0
