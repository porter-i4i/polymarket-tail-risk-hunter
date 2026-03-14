# Live Transition Checklist

Complete this checklist **only after** the
[7-day paper test](./PAPER_TEST_CHECKLIST.md) passes all acceptance criteria.

---

## Phase 0 — Final Paper-Test Sign-Off

- [ ] All 7-day paper test acceptance criteria met (see `docs/PAPER_TEST_CHECKLIST.md`)
- [ ] Full test suite passing: `python -m pytest tests/ -v`
- [ ] No open action items from paper test log review
- [ ] Decision: start with Phase 1 limits ($20/day, $2/bet, 15 bets/day)

---

## Phase 1 — Wallet Setup

### Create / Prepare Wallet

- [ ] Polygon wallet created in MetaMask or Rabby
  - Use a **dedicated wallet** for this bot — do not share with other funds
- [ ] Wallet address recorded for reference: `0x__________________________`
- [ ] Wallet backed up (seed phrase written down and stored securely offline)

### Fund Wallet

- [ ] MATIC (gas) transferred to wallet on Polygon network
  - Minimum: 2 MATIC (covers hundreds of transactions)
  - Recommended: 5–10 MATIC to avoid topping up during Phase 1
- [ ] USDC.e transferred to wallet on Polygon network
  - Phase 1 budget: start with $50–$100 (max daily spend is $20)
  - Verify it is **USDC.e** (Polymarket's accepted stablecoin), not native USDC
- [ ] Balances confirmed on Polygonscan or in wallet UI

### Approve Contracts

These approvals allow the CTF Exchange to settle trades on your behalf.
- [ ] Visit [polymarket.com](https://polymarket.com) with the bot wallet
- [ ] Complete the standard on-boarding flow (sign approval transactions)
- [ ] Alternatively, approve programmatically via `py-clob-client`:
  ```python
  from py_clob_client.client import ClobClient
  client = ClobClient(host="https://clob.polymarket.com", key=PRIVATE_KEY, chain_id=137)
  client.set_allowances()  # approves CTF Exchange + Neg Risk CTF Exchange
  ```
- [ ] Approval transactions confirmed on Polygonscan (status: Success)

---

## Phase 2 — Environment Configuration

- [ ] Stop any running dry-run bot process: `kill $(pgrep -f main_loop.py)`
- [ ] Open `.env` in editor

  ```bash
  # Export private key (hex, WITHOUT 0x prefix)
  POLYMARKET_PRIVATE_KEY=<your_key_here>

  # Switch to live mode
  DRY_RUN=false
  ```

- [ ] Double-check all Phase 1 spend limits are set conservatively:
  ```
  PHASE1_MAX_DAILY_SPEND=20
  PHASE1_MAX_SINGLE_BET=2
  PHASE1_MAX_BETS_PER_DAY=15
  PHASE1_FLAT_BET=1.5
  ```
- [ ] Verify `TOTAL_BANKROLL` matches the actual USDC.e deposited
- [ ] `.env` file is NOT committed to git (verify: `git status` shows it as ignored)

---

## Phase 3 — Pre-Flight Verification

Run the health check script:
```bash
python scripts/health_check.py
```

Expected output: all checks PASS. Investigate any FAIL before proceeding.

Manual checks:
- [ ] `python -m pytest tests/ -v` — all tests pass
- [ ] `python -c "from config import load_config, validate_config; cfg=load_config(); errs=validate_config(cfg); print('OK' if not errs else errs)"` — prints `OK`
- [ ] Database initialised: `python scripts/init_db.py`
- [ ] Executor can reach Polymarket: `python -c "import asyncio; from modules.order_executor import OrderExecutor; from config import load_config; cfg=load_config(); e=OrderExecutor(cfg); asyncio.run(e.preflight_check())"` — no exception

---

## Phase 4 — Launch

Start the bot in a persistent session (tmux / screen / systemd):

```bash
# Option A: tmux (recommended for manual operation)
tmux new-session -s polybot
python main_loop.py
# Detach: Ctrl-B then D

# Option B: systemd (recommended for VPS production)
sudo cp deploy/polymarket-tailrisk.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable polymarket-tailrisk
sudo systemctl start polymarket-tailrisk
sudo journalctl -u polymarket-tailrisk -f   # follow logs

# Option C: screen
screen -S polybot
python main_loop.py
# Detach: Ctrl-A then D
```

Start dashboard in a separate terminal / tmux pane:
```bash
streamlit run dashboard.py --server.port 8501
```

---

## Phase 5 — First-Hour Monitoring

Monitor closely for the first hour after going live.

- [ ] Startup Telegram alert received
- [ ] First cycle completes without fatal error (check logs)
- [ ] First real order placed (check `orders` table: `is_dry_run=0`)
- [ ] Order appears on Polymarket UI
- [ ] Balance on-chain matches `TOTAL_BANKROLL` (within rounding)
- [ ] No kill-switch trigger in `phase_transitions`
- [ ] Dashboard shows live data (non-zero positions, non-zero spend)
- [ ] Heartbeat thread running (check logs for "heartbeat" entries)

---

## Phase 6 — Live Phase 1 Operation

The bot starts in **Phase 1 automatically** (calibration phase).

During Phase 1:
- Flat bets of $1.50 each, max $20/day, max 15 bets/day
- Kelly sizing NOT used — fixed bet size only
- Data is collected to build the Platt calibration model
- Phase 2 requires ≥ 200 resolved predictions AND calibration score CI lower ≥ 0.60

Expected timeline to Phase 2: 4–8 weeks (depends on market resolution speed).

### Daily Live Monitoring

- [ ] Check Telegram alerts daily (trades, errors, phase changes)
- [ ] Check dashboard for open positions and daily PnL
- [ ] Check on-chain USDC.e balance weekly (reconcile with DB)
- [ ] Review `logs/bot.log` for any recurring error patterns
- [ ] Top up MATIC if balance drops below 1 MATIC

---

## Emergency Stop

If anything looks wrong:

```bash
# Send SIGTERM for graceful shutdown (cancels open orders, saves state)
kill -TERM $(pgrep -f main_loop.py)

# Or if using systemd:
sudo systemctl stop polymarket-tailrisk
```

The bot will:
1. Cancel all open orders
2. Save final daily stats to DB
3. Send shutdown Telegram alert
4. Exit cleanly

---

## Rollback to Dry-Run

To revert to paper-test mode without losing data:
```bash
# In .env:
DRY_RUN=true
POLYMARKET_PRIVATE_KEY=   # clear the key

# Restart bot
python main_loop.py
```

All historical data remains in the DB. Orders placed in live mode are
flagged `is_dry_run=0`; future dry-run orders will be flagged `is_dry_run=1`.
