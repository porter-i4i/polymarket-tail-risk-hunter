# 7-Day Paper Test Checklist

Paper testing validates the full bot pipeline — from market scanning through
AI scoring to simulated order placement — without spending real money.
**All orders remain dry-run (`is_dry_run=1` in the DB).**

---

## Pre-Test Setup

- [ ] Repository cloned and on the correct branch
- [ ] Python 3.11+ installed; virtualenv activated
- [ ] `pip install -r requirements.txt` completed without errors
- [ ] `.env` created from `.env.example`
  - [ ] `ANTHROPIC_API_KEY` set to a valid key
  - [ ] `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` set
  - [ ] `POLYMARKET_PRIVATE_KEY` **left empty**
  - [ ] `DRY_RUN=true` confirmed
- [ ] Database initialised: `python scripts/init_db.py`
- [ ] Full test suite passes: `python -m pytest tests/ -v` (expect 360+ passed, 0 failed)
- [ ] Bot starts cleanly: `python main_loop.py` (ctrl-C after one cycle to confirm)
- [ ] Dashboard launches: `streamlit run dashboard.py` (opens in browser, no errors)

---

## Daily Checklist (Days 1–7)

Perform each morning after the overnight run.

### Data Integrity
- [ ] DB file present at `data/polybot.db` (non-zero size)
- [ ] Predictions table has new rows: `SELECT COUNT(*) FROM predictions`
- [ ] No duplicate `condition_id` entries in open positions
- [ ] Daily stats row written for yesterday: `SELECT * FROM daily_stats ORDER BY date DESC LIMIT 3`

### Operational Health
- [ ] Bot process alive (no crash, no zombie)
- [ ] `logs/bot.log` shows recent timestamps (< 2 × `CYCLE_INTERVAL_SECONDS`)
- [ ] Telegram alerts arriving (startup, trades, any errors)
- [ ] Dashboard KPI row shows non-zero values after first successful cycle

### Scoring Pipeline Sanity
- [ ] `ai_probability_raw` values in predictions are in reasonable range (< 0.15 for tail events)
- [ ] `ai_probability_debiased` <= `ai_probability_raw` (debiaser is pulling toward market)
- [ ] `ai_probability_adjusted` is a further adjustment (Platt may not fit yet with < 100 samples)
- [ ] `p10 <= p50 <= p90` constraint holds for all rows

### Risk Guard Verification
- [ ] Daily spend stays within `PHASE1_MAX_DAILY_SPEND` ($20 default)
- [ ] Daily bet count stays within `PHASE1_MAX_BETS_PER_DAY` (15 default)
- [ ] No single bet exceeds `PHASE1_MAX_SINGLE_BET` ($2 default)
- [ ] No position opened on the same `condition_id` twice

### Cycle Performance
- [ ] Average cycle duration < `CYCLE_INTERVAL_SECONDS`
- [ ] Cache hit rate > 0% after day 1 (scoring cache is working)
- [ ] No persistent error classes appearing in cycle logs (transient 429s are OK)

---

## End-of-Test Acceptance Criteria

All of the following must be true before considering live transition:

| Criterion | Target | Source |
|---|---|---|
| Predictions logged | ≥ 50 (ideally 100+) | `SELECT COUNT(*) FROM predictions` |
| Zero DB constraint violations | 0 | No errors in logs |
| Risk limits respected every cycle | 0 breaches | Log review |
| Kill switch never triggered | 0 triggers | `phase_transitions` WHERE `to_phase=5` |
| Bot uptime over 7 days | ≥ 90% | Log timestamps |
| Telegram alerts delivered | Startup + all trades | Manual check |
| Dashboard shows data | Non-empty KPIs | Visual check |
| Full test suite still passing | 360+ passed | `python -m pytest tests/ -v` |
| Log file rotation working | `logs/bot.log.*` gz files present | `ls -lh logs/` |

---

## Paper Test Metrics to Record

Fill in at end of day 7 before making live-transition decision:

```
Total predictions logged   : ____
Total simulated bets       : ____
Unique markets evaluated   : ____
Average cycle duration (s) : ____
Cache hit rate (%)         : ____
Resolved predictions (if any): ____
Current calibration score  : ____ (N/A until 100 resolved)
Phase at end of test       : 1
Any kill-switch triggers   : ____
Any DB errors              : ____
```

---

## Escalation Criteria (Stop and Investigate)

Stop the paper test and investigate if any of the following occur:

- Kill switch triggers (`to_phase=5` in `phase_transitions`)
- DB write failure (`UNIQUE constraint`, `CHECK constraint` errors)
- AI error rate > 30% in a single day (check cycle logs)
- `buy_rate` > 50% (sanity gate should have caught this)
- Average adjusted probability > 15% across predictions
- Cycle gap > 15 minutes without a known cause

---

## Next Step

If all day-7 acceptance criteria pass → proceed to
[`docs/LIVE_TRANSITION_CHECKLIST.md`](./LIVE_TRANSITION_CHECKLIST.md).
