# Operations Runbook

Day-to-day operating procedures for the Polymarket Tail Risk Hunter bot.

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [First-Time Setup](#first-time-setup)
3. [Exact Run Sequence](#exact-run-sequence)
4. [Starting the Bot](#starting-the-bot)
5. [Starting the Dashboard](#starting-the-dashboard)
6. [Stopping the Bot](#stopping-the-bot)
7. [Log Management](#log-management)
8. [Database Operations](#database-operations)
9. [Common Operational Tasks](#common-operational-tasks)
10. [Systemd Service](#systemd-service)
11. [Troubleshooting](#troubleshooting)

---

## Repository Layout

```
polymarket-tail-risk-hunter/
├── main_loop.py            # Main async bot loop — entry point
├── dashboard.py            # Streamlit dashboard (separate process)
├── config.py               # Config loader and validator
├── requirements.txt        # Python dependencies
├── .env.example            # Config template (copy to .env)
├── .env                    # Local config (gitignored)
│
├── modules/                # Core bot modules
│   ├── types.py            # Shared dataclasses and enums
│   ├── portfolio_tracker.py # SQLite DB layer
│   ├── market_scanner.py   # Gamma API market fetching
│   ├── ai_scoring.py       # Claude scoring pipeline
│   ├── risk_manager.py     # Kelly sizing, phase limits
│   ├── order_executor.py   # CLOB order placement
│   ├── exit_manager.py     # Take-profit / stop-loss exits
│   ├── calibration_tracker.py  # Platt calibration tracking
│   ├── calibration_adjuster.py # Platt/isotonic adjuster
│   ├── tail_event_debiaser.py  # Shrinkage debiaser
│   ├── adverse_selection_detector.py
│   ├── order_book_health.py
│   ├── correlation_manager.py
│   ├── signal_aggregator.py
│   ├── scoring_cache.py
│   ├── fee_calculator.py
│   ├── news_engine.py
│   ├── notifications.py    # Telegram alerts
│   └── rate_limiter.py     # Token-bucket rate limiters
│
├── tests/                  # pytest test suite (360+ tests)
├── scripts/                # Helper scripts
│   ├── init_db.py          # Database initialiser
│   └── health_check.py     # Pre-flight health check
├── docs/                   # Operational documentation
│   ├── RUNBOOK.md          # This file
│   ├── PAPER_TEST_CHECKLIST.md
│   └── LIVE_TRANSITION_CHECKLIST.md
├── deploy/
│   └── polymarket-tailrisk.service  # systemd unit template
├── data/                   # SQLite DB (gitignored)
└── logs/                   # Rotating log files (gitignored)
```

---

## First-Time Setup

```bash
# 1. Clone / navigate to repo
cd /path/to/polymarket-tail-risk-hunter

# 2. Create and activate virtualenv
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create config
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
# Leave POLYMARKET_PRIVATE_KEY empty for dry run
# Keep DRY_RUN=true

# 5. Initialise database
python scripts/init_db.py

# 6. Run health check
python scripts/health_check.py

# 7. Run tests
python -m pytest tests/ -v
```

---

## Exact Run Sequence

This is the canonical sequence from the master specification.

```bash
# ── Step 1: Environment ────────────────────────────────────────────────────
source venv/bin/activate

# ── Step 2: Configure ──────────────────────────────────────────────────────
cp .env.example .env
# Set ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID in .env
# Leave POLYMARKET_PRIVATE_KEY empty   → dry run
# Confirm DRY_RUN=true

# ── Step 3: Initialise database ────────────────────────────────────────────
python scripts/init_db.py
# or equivalently:
# python -c "from modules.portfolio_tracker import Database; Database()"

# ── Step 4: Run tests ──────────────────────────────────────────────────────
python -m pytest tests/ -v

# ── Step 5: Dry run (real API calls, no real orders) ──────────────────────
python main_loop.py

# ── Step 6: Dashboard (separate terminal) ─────────────────────────────────
streamlit run dashboard.py

# ── Step 7: When ready for live (after 7-day paper test) ──────────────────
# See docs/LIVE_TRANSITION_CHECKLIST.md for the full wallet → live workflow
```

---

## Starting the Bot

### Interactive (foreground)

```bash
source venv/bin/activate
python main_loop.py
```

### tmux session (recommended for VPS)

```bash
tmux new-session -d -s polybot 'source venv/bin/activate && python main_loop.py'
tmux attach -t polybot          # attach
# Ctrl-B D to detach
```

### Background with nohup

```bash
nohup python main_loop.py > /dev/null 2>&1 &
echo $! > /tmp/polybot.pid
```

### systemd (production)

```bash
sudo cp deploy/polymarket-tailrisk.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable polymarket-tailrisk
sudo systemctl start polymarket-tailrisk
```

---

## Starting the Dashboard

The dashboard is a **separate read-only process**. Start it in a separate
terminal or tmux pane.

```bash
# Default port 8501
streamlit run dashboard.py

# Custom port
streamlit run dashboard.py --server.port 8502 --server.address 0.0.0.0

# Background with tmux
tmux new-session -d -s polydash 'source venv/bin/activate && streamlit run dashboard.py'
```

Navigate to `http://localhost:8501` (or the VPS IP on the configured port).

The dashboard auto-refreshes every 30 seconds. It reads directly from the
SQLite DB and is safe to run at any time, including while the bot is running.

---

## Stopping the Bot

### Graceful shutdown (preferred)

```bash
# SIGTERM: bot finishes current cycle, cancels open orders, saves state
kill -TERM $(pgrep -f main_loop.py)

# Systemd:
sudo systemctl stop polymarket-tailrisk
```

The bot will:
1. Complete (or abort) the current cycle
2. Cancel any open orders on-chain (live mode only)
3. Write final daily stats to DB
4. Send shutdown Telegram alert
5. Exit with code 0

### Immediate stop

```bash
# SIGINT (same as Ctrl-C) — triggers same graceful shutdown handler
kill -INT $(pgrep -f main_loop.py)
```

### Force kill (last resort)

```bash
kill -9 $(pgrep -f main_loop.py)
# Warning: may leave open orders on-chain in live mode
# After force kill: run python scripts/health_check.py to verify state
```

---

## Log Management

Logs are written to `logs/bot.log` with daily rotation and 30-day retention.

```bash
# Follow live logs (JSON lines)
tail -f logs/bot.log | python -m json.tool

# Search for errors
grep '"level":"ERROR"' logs/bot.log | tail -20

# Count errors per level
grep -o '"level":"[A-Z]*"' logs/bot.log | sort | uniq -c

# View compressed rotated logs
zcat logs/bot.log.2024-01-15.gz | grep '"level":"ERROR"'

# List all log files
ls -lh logs/
```

---

## Database Operations

### Inspect DB

```bash
# Open DB in SQLite shell
sqlite3 data/polybot.db

# Key queries
.tables
SELECT COUNT(*) FROM orders;
SELECT COUNT(*) FROM predictions;
SELECT COUNT(*) FROM predictions WHERE outcome IS NOT NULL;
SELECT * FROM daily_stats ORDER BY date DESC LIMIT 7;
SELECT * FROM phase_transitions ORDER BY id DESC LIMIT 5;
SELECT COUNT(*) FROM positions WHERE status='open';
```

### Backup DB

```bash
# SQLite backup (safe to run while bot is running — WAL mode)
sqlite3 data/polybot.db ".backup data/polybot_backup_$(date +%Y%m%d).db"

# Or copy the file (stop bot first for consistency)
cp data/polybot.db data/polybot_backup_$(date +%Y%m%d).db
```

### Re-initialise DB (development only)

```bash
# WARNING: destroys all data
rm -f data/polybot.db
python scripts/init_db.py
```

---

## Common Operational Tasks

### Check current phase

```bash
python -c "
from modules.portfolio_tracker import Database
db = Database()
print('Phase:', db.get_current_phase())
print('Balance:', db.get_current_balance())
print('Open positions:', db.get_open_position_count())
"
```

### Check calibration score

```bash
python -c "
from modules.portfolio_tracker import Database
from modules.notifications import Notifier
from modules.calibration_tracker import CalibrationTracker
from config import load_config
cfg = load_config()
db = Database()
notifier = Notifier(cfg)
ct = CalibrationTracker(db, notifier, cfg)
stats = ct.calculate_calibration_score()
print(stats)
"
```

### Force balance reconciliation

```bash
python -c "
from modules.portfolio_tracker import Database
db = Database()
db.set_balance_override(float(input('Enter on-chain USDC.e balance: ')))
print('Done')
"
```

### Rotate to a new log file manually

```bash
kill -HUP $(pgrep -f main_loop.py) 2>/dev/null || echo "Bot not running (no HUP needed)"
# loguru handles rotation automatically on date change
```

---

## Systemd Service

A service file template is at `deploy/polymarket-tailrisk.service`.

```bash
# Install
sudo cp deploy/polymarket-tailrisk.service /etc/systemd/system/
sudo nano /etc/systemd/system/polymarket-tailrisk.service
# Edit: User=, WorkingDirectory=, EnvironmentFile= paths

# Enable + start
sudo systemctl daemon-reload
sudo systemctl enable polymarket-tailrisk
sudo systemctl start polymarket-tailrisk

# Status
sudo systemctl status polymarket-tailrisk

# Live log stream
sudo journalctl -u polymarket-tailrisk -f

# Restart after .env change
sudo systemctl restart polymarket-tailrisk
```

---

## Troubleshooting

### Bot exits immediately at startup

1. Check config: `python -c "from config import load_config, validate_config; cfg=load_config(); print(validate_config(cfg))"`
2. Missing secrets? Add `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` to `.env`
3. DB error? Run `python scripts/init_db.py`

### No orders being placed

- Check `MIN_EDGE_PCT` and `MIN_AI_CONFIDENCE` — may be too strict for current market
- Check `PHASE1_MAX_BETS_PER_DAY` limit (resets at midnight UTC)
- Check AI error rate in logs (`"AI error"` grep)
- Check that markets meet `MAX_PRICE_THRESHOLD` (5¢ default — very restrictive by design)

### High AI error rate

- Verify `ANTHROPIC_API_KEY` is valid and has credits
- Check Claude rate limit: `CLAUDE_RATE_LIMIT` (default 50/min)
- Rate limit hits (429s) are retried automatically per the retry policy

### Telegram alerts not arriving

- Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`
- Send a test: `curl https://api.telegram.org/bot<TOKEN>/sendMessage?chat_id=<CHAT_ID>&text=test`
- Alert failures are logged but do not stop the bot

### Dashboard shows empty state

- The dashboard is read-only — it only shows data that exists in the DB
- On first run, the DB is empty; after the first cycle, data appears
- Check that `DB_PATH` in `.env` points to the same DB the bot is using (default: `data/polybot.db`)

### Kill switch triggered

1. Check `phase_transitions` WHERE `to_phase=5` for the trigger reason
2. Review the specific kill condition (drawdown, calibration, heartbeat, etc.)
3. Fix the underlying issue
4. Reset by inserting a new `phase_transitions` row back to phase 1 (manual DB edit)
5. Do NOT restart the bot until the root cause is resolved
