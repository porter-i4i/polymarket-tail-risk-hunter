# Polymarket Tail Risk Hunter

Institutional-grade self-calibrating autonomous trading bot for Polymarket
prediction markets. Hunts tail-risk opportunities (low-probability events)
using Claude AI scoring, Platt calibration, robust Kelly sizing, and a
multi-stage phase system.

---

## Architecture Overview

```
main_loop.py  ←──── orchestrates all modules, async event loop
dashboard.py  ←──── Streamlit read-only dashboard (separate process)

modules/
  market_scanner.py       Gamma API: scan markets, filter by price/volume/liquidity
  ai_scoring.py           Claude: probability distribution (p10/p50/p90) + confidence
  tail_event_debiaser.py  Shrinkage toward market price (correct LLM overconfidence)
  calibration_adjuster.py Platt / Isotonic scaling on resolved predictions
  signal_aggregator.py    Composite signal from news + order book + price history
  adverse_selection_detector.py  Toxic flow / imbalance guard
  order_book_health.py    Book depth + spread health gate
  risk_manager.py         Robust Kelly sizing, phase limits, category exposure
  order_executor.py       py-clob-client: place / cancel CLOB orders
  exit_manager.py         Take-profit / stop-loss / time-decay exits
  position_monitor.py     Resolution tracking, stale position cleanup
  calibration_tracker.py  Brier score, calibration score, phase transitions
  portfolio_tracker.py    SQLite DB (WAL mode, thread-safe)
  notifications.py        Telegram alerts
  scoring_cache.py        Adaptive TTL scoring cache (save Claude API spend)
  fee_calculator.py       Fee-adjusted EV calculation
  correlation_manager.py  Cluster exposure limits
  rate_limiter.py         5 separate token-bucket limiters
```

The dashboard runs as a **separate read-only process** alongside the bot loop.

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/porter-i4i/polymarket-tail-risk-hunter
cd polymarket-tail-risk-hunter
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
# Leave POLYMARKET_PRIVATE_KEY empty for dry run
# Confirm DRY_RUN=true

# 3. Initialise database
python scripts/init_db.py

# 4. Run health check
python scripts/health_check.py

# 5. Run tests
python -m pytest tests/ -v

# 6. Dry run (real Gamma + Claude API calls, no real orders)
python main_loop.py

# 7. Dashboard (open in a separate terminal)
streamlit run dashboard.py
```

Navigate to http://localhost:8501 to view the dashboard.

---

## Exact Run Sequence (from master spec)

```bash
# Step 1 — Environment
source venv/bin/activate

# Step 2 — Configure (.env)
#   Set ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
#   Leave POLYMARKET_PRIVATE_KEY empty   (dry run)
#   Confirm DRY_RUN=true

# Step 3 — Initialise database
python scripts/init_db.py
# or: python -c "from modules.portfolio_tracker import Database; Database()"

# Step 4 — Run tests
python -m pytest tests/ -v

# Step 5 — Dry run (validates full pipeline end-to-end)
python main_loop.py

# Step 6 — Dashboard (separate terminal)
streamlit run dashboard.py

# Step 7 — When ready for live (after 7-day paper test passes)
#   See docs/LIVE_TRANSITION_CHECKLIST.md
```

---

## Phase System

| Phase | Description | Trigger |
|---|---|---|
| 1 | Calibration — flat bets ($1.50), data collection | Default start |
| 2 | Scaled — Kelly sizing (25% fraction) | ≥ 200 resolved predictions, cal score CI ≥ 0.60 |
| 3 | Full — higher Kelly (50% fraction) | ≥ 500 resolved predictions, cal score CI ≥ 0.65 |
| 4 | Circuit breaker — exits only, no new bets | Loss > 15% in 7 days |
| 5 | Killed — bot halted | Kill switch triggered |

Phase transitions are automatic based on calibration score bootstrap CI lower
bound — not point estimate — to prevent upgrades on lucky streaks.

---

## Safety Features

- **Kill switch** — 11 conditions (drawdown, consecutive losses, calibration collapse, AI error rate, heartbeat gap, cycle gap, balance mismatch)
- **Circuit breaker** — trips on 15% loss in 7 days; runs exit-only mode
- **Sanity gate** — halts if buy rate > 50% or average probability > 15%
- **Deduplication** — never bets the same `condition_id` twice
- **Adverse selection** — skips toxic flow / imbalanced books
- **ADR-006** — clob_write max 1 attempt (never duplicate orders)
- **WAL + Lock** — SQLite concurrency safety
- **Feature flags** — every v3/v4 module can be disabled for A/B testing

---

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env`
and edit before running.

Key variables:

| Variable | Default | Description |
|---|---|---|
| `DRY_RUN` | `true` | Paper test mode |
| `POLYMARKET_PRIVATE_KEY` | _(empty)_ | Required for live mode |
| `ANTHROPIC_API_KEY` | _(required)_ | Claude scoring |
| `TOTAL_BANKROLL` | `1000` | USD bankroll for sizing |
| `MAX_PRICE_THRESHOLD` | `0.05` | Max token price (5¢) |
| `PHASE1_MAX_DAILY_SPEND` | `20` | Phase 1 daily cap |
| `PHASE1_FLAT_BET` | `1.50` | Phase 1 fixed bet size |
| `ENABLE_DEBIASER` | `true` | Shrinkage debiaser |
| `ENABLE_PLATT_SCALING` | `true` | Platt calibration |

Full reference: see `.env.example` for all variables with descriptions.

---

## Documentation

| Document | Purpose |
|---|---|
| `docs/RUNBOOK.md` | Day-to-day operating procedures, startup/stop, log management |
| `docs/PAPER_TEST_CHECKLIST.md` | 7-day paper test checklist and acceptance criteria |
| `docs/LIVE_TRANSITION_CHECKLIST.md` | Wallet setup → approvals → go-live steps |
| `.env.example` | Annotated configuration reference |
| `deploy/polymarket-tailrisk.service` | systemd unit file template |

---

## Testing

```bash
# Run full suite
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_risk_manager.py -v
python -m pytest tests/test_calibration_tracker.py -v

# With coverage
python -m pytest tests/ --cov=modules --cov-report=term-missing
```

Expected: 360+ tests, 0 failures.

---

## Production Deployment (VPS / systemd)

```bash
# Install systemd service
sudo cp deploy/polymarket-tailrisk.service /etc/systemd/system/
# Edit paths and user in the service file
sudo systemctl daemon-reload
sudo systemctl enable polymarket-tailrisk
sudo systemctl start polymarket-tailrisk
sudo journalctl -u polymarket-tailrisk -f
```

See `docs/RUNBOOK.md` for the full systemd workflow.

---

## Database Schema

SQLite at `data/polybot.db` (WAL mode, thread-safe).

| Table | Contents |
|---|---|
| `orders` | Every order placed (dry and live), with full scoring metadata |
| `positions` | Open/closed positions with entry price, shares, exit reason |
| `predictions` | Raw/debiased/adjusted probabilities + outcomes for calibration |
| `daily_stats` | Per-day PnL, bet count, balance, calibration score, Sharpe |
| `phase_transitions` | Phase change history with trigger reasons |

---

## License

Private / proprietary.
