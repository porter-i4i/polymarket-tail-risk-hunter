"""
Main orchestrator for Polymarket Tail Risk Hunter.

Async event loop with startup sequence, cycle logic, reconciliation,
graceful shutdown, and kill switch enforcement.
"""
import asyncio
import signal
import sys
import time
from datetime import date

from loguru import logger

from config import load_config, validate_config
from modules.types import (
    AdjustedScore,
    BotPhase,
    BookHealth,
    CycleSummary,
    FlowToxicity,
    OrderResult,
)
from modules.rate_limiter import RateLimiter
from modules.portfolio_tracker import Database
from modules.market_scanner import MarketScanner
from modules.news_engine import NewsEngine
from modules.ai_scoring import AIScoringEngine
from modules.scoring_cache import ScoringCache
from modules.correlation_manager import CorrelationManager
from modules.signal_aggregator import SignalAggregator
from modules.order_executor import OrderExecutor
from modules.risk_manager import RiskManager
from modules.exit_manager import ExitManager
from modules.position_monitor import PositionMonitor
from modules.tail_event_debiaser import TailEventDebiaser
from modules.calibration_adjuster import CalibrationAdjuster
from modules.calibration_tracker import CalibrationTracker
from modules.adverse_selection_detector import AdverseSelectionDetector
from modules.order_book_health import OrderBookHealthChecker
from modules.fee_calculator import FeeCalculator
from modules.notifications import Notifier

# Kill conditions — any single condition triggers KILLED state
KILL_CONDITIONS = {
    "drawdown_from_peak_pct": 0.40,
    "total_loss_absolute_usd": 500,
    "consecutive_loss_days": 14,
    "calibration_score_negative": True,
    "calibration_below": 0.10,
    "daily_buy_rate_above": 0.50,
    "daily_avg_probability_above": 20,
    "ai_api_error_rate_above": 0.30,
    "heartbeat_gap_seconds": 300,
    "cycle_gap_seconds": 900,
    "balance_mismatch_usd": 50,
    "db_write_failure": True,
}


def _setup_logging():
    """Configure loguru: console + JSON-lines file."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {level:<8} | {extra[module]:<20} | {message}",
        filter=lambda record: "module" in record["extra"],
    )
    logger.add(
        "logs/bot.log",
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        compression="gz",
        serialize=True,
    )


class MainLoop:
    def __init__(self):
        self.cfg: dict = {}
        self.db: Database | None = None
        self.dry_run: bool = True
        self.running: bool = False
        self.cycle_counter: int = 0
        self.last_cycle_time: float = 0.0

        # Rate limiters
        self.gamma_limiter: RateLimiter | None = None
        self.clob_data_limiter: RateLimiter | None = None
        self.clob_order_limiter: RateLimiter | None = None
        self.news_limiter: RateLimiter | None = None
        self.claude_limiter: RateLimiter | None = None

        # Modules
        self.scanner: MarketScanner | None = None
        self.news: NewsEngine | None = None
        self.ai: AIScoringEngine | None = None
        self.scoring_cache: ScoringCache | None = None
        self.correlation_mgr: CorrelationManager | None = None
        self.signal_agg: SignalAggregator | None = None
        self.executor: OrderExecutor | None = None
        self.risk: RiskManager | None = None
        self.exit_mgr: ExitManager | None = None
        self.monitor: PositionMonitor | None = None
        self.debiaser: TailEventDebiaser | None = None
        self.calibrator: CalibrationAdjuster | None = None
        self.calibration: CalibrationTracker | None = None
        self.adverse: AdverseSelectionDetector | None = None
        self.book_checker: OrderBookHealthChecker | None = None
        self.fee_calc: FeeCalculator | None = None
        self.notifier: Notifier | None = None

        self._log = logger.bind(module="main")

    # === INITIALIZATION ===

    def _init_rate_limiters(self, cfg: dict) -> None:
        """Create 5 separate rate limiters with exact quotas from spec."""
        self.gamma_limiter = RateLimiter(max_per_minute=cfg.get("GAMMA_RATE_LIMIT", 80))
        self.clob_data_limiter = RateLimiter(max_per_minute=cfg.get("CLOB_DATA_RATE_LIMIT", 80))
        self.clob_order_limiter = RateLimiter(max_per_minute=cfg.get("CLOB_ORDER_RATE_LIMIT", 40))
        self.news_limiter = RateLimiter(max_per_minute=cfg.get("NEWS_RATE_LIMIT", 12))
        self.claude_limiter = RateLimiter(max_per_minute=cfg.get("CLAUDE_RATE_LIMIT", 50))

    def _init_modules(self, cfg: dict) -> None:
        """Initialize all modules in dependency order."""
        self.fee_calc = FeeCalculator()
        self.debiaser = TailEventDebiaser()
        self.calibrator = CalibrationAdjuster(cfg=cfg)
        self.correlation_mgr = CorrelationManager()
        self.scoring_cache = ScoringCache(
            default_ttl=cfg.get("SCORING_CACHE_DEFAULT_TTL", 1800),
        )
        self.notifier = Notifier(cfg)
        self.scanner = MarketScanner(
            cfg, self.gamma_limiter, self.clob_data_limiter, db=self.db,
        )
        self.news = NewsEngine(self.news_limiter, cfg)
        self.ai = AIScoringEngine(self.claude_limiter, cfg)
        self.signal_agg = SignalAggregator(cfg)
        self.executor = OrderExecutor(cfg, clob_order_limiter=self.clob_order_limiter)
        self.risk = RiskManager(self.db, self.fee_calc, self.correlation_mgr, cfg)
        self.calibration = CalibrationTracker(self.db, self.notifier, cfg)
        self.exit_mgr = ExitManager(self.executor, self.db, self.scanner, self.notifier, cfg)
        self.monitor = PositionMonitor(self.db, self.scanner, self.calibration, self.notifier, cfg)
        self.adverse = AdverseSelectionDetector(cfg=cfg)
        self.book_checker = OrderBookHealthChecker(cfg=cfg)

    # === PREFLIGHT ===

    async def _preflight_checks(self) -> bool:
        """Run preflight checks. Returns True if all pass."""
        if self.dry_run:
            self._log.info("Preflight: dry run mode — skipping executor check")
            return True
        try:
            passed = await self.executor.preflight_check()
            if not passed:
                self._log.critical("Executor preflight failed")
                return False
            return True
        except Exception as e:
            self._log.critical("Preflight exception: %s", e)
            return False

    # === RECONCILIATION ===

    async def _reconcile_state(self) -> bool:
        """Compare on-chain state with DB. Returns False if mismatch > threshold."""
        if self.dry_run:
            return True

        try:
            actual_balance = await self.executor.get_balance()
        except Exception as e:
            self._log.error("Failed to get on-chain balance: %s", e)
            return False

        expected_balance = self.db.get_current_balance()
        diff = abs(actual_balance - expected_balance)

        if diff > 5.0:
            self._log.warning(
                "Balance mismatch: on-chain=$%.2f, DB=$%.2f",
                actual_balance, expected_balance,
            )
            if diff > float(KILL_CONDITIONS["balance_mismatch_usd"]):
                self._log.critical("Balance mismatch $%.2f exceeds kill threshold", diff)
                await self.notifier.send(f"BALANCE MISMATCH: ${diff:.2f}")
                return False
            # Small mismatch: update DB to match on-chain (source of truth)
            self.db.set_balance_override(actual_balance)
            self._log.info("Balance reconciled to on-chain: $%.2f", actual_balance)

        return True

    # === KILL SWITCH ===

    def _check_kill_conditions(self) -> str | None:
        """Check all kill conditions. Returns reason string if triggered, None otherwise."""
        balance = self.db.get_current_balance()
        peak = self.db.get_peak_balance()

        # Financial: drawdown from peak
        if peak > 0:
            drawdown = (peak - balance) / peak
            if drawdown >= KILL_CONDITIONS["drawdown_from_peak_pct"]:
                return f"Drawdown {drawdown:.1%} >= {KILL_CONDITIONS['drawdown_from_peak_pct']:.0%}"

        # Financial: absolute loss
        bankroll = self.cfg.get("TOTAL_BANKROLL", 1000)
        total_loss = bankroll - balance
        if total_loss >= KILL_CONDITIONS["total_loss_absolute_usd"]:
            return f"Total loss ${total_loss:.0f} >= ${KILL_CONDITIONS['total_loss_absolute_usd']}"

        # Financial: consecutive loss days
        consec = self.db.get_consecutive_loss_days()
        if consec >= KILL_CONDITIONS["consecutive_loss_days"]:
            return f"Consecutive loss days {consec} >= {KILL_CONDITIONS['consecutive_loss_days']}"

        # Calibration checks (only after enough resolutions)
        resolved = self.db.get_resolved_predictions()
        if len(resolved) >= 50:
            cal_score = self.calibration.get_latest_score()
            if KILL_CONDITIONS["calibration_score_negative"] and cal_score < 0:
                return f"Calibration score {cal_score:.3f} is negative"
            if cal_score < KILL_CONDITIONS["calibration_below"]:
                return f"Calibration score {cal_score:.3f} < {KILL_CONDITIONS['calibration_below']}"

        # AI health: error rate
        if hasattr(self.ai, 'error_rate') and self.ai.error_rate > KILL_CONDITIONS["ai_api_error_rate_above"]:
            return f"AI error rate {self.ai.error_rate:.1%} > {KILL_CONDITIONS['ai_api_error_rate_above']:.0%}"

        # Infrastructure: heartbeat gap
        if not self.dry_run and hasattr(self.executor, 'heartbeat_failures'):
            hb_failures = self.executor.heartbeat_failures
            if hb_failures >= 10:
                return f"Heartbeat failures {hb_failures} >= 10"

        # Infrastructure: cycle gap
        if self.last_cycle_time > 0:
            gap = time.time() - self.last_cycle_time
            if gap > KILL_CONDITIONS["cycle_gap_seconds"]:
                return f"Cycle gap {gap:.0f}s > {KILL_CONDITIONS['cycle_gap_seconds']}s"

        return None

    async def _enter_killed_state(self, reason: str) -> None:
        """Transition to KILLED state and notify."""
        self._log.critical("KILL SWITCH: %s", reason)
        self.db.record_phase_transition(
            from_phase=self.db.get_current_phase(),
            to_phase=BotPhase.KILLED.value,
            reason=reason,
        )
        await self.notifier.kill_switch_alert(reason)
        self.running = False

    # === MAIN CYCLE ===

    async def _cycle(self) -> None:
        """Execute one full bot cycle."""
        cycle_start = time.time()
        phase = BotPhase(self.db.get_current_phase())
        cycle_errors: list[str] = []
        scores_today: list = []

        # STEP 1: Kill switch check (highest priority)
        kill_reason = self._check_kill_conditions()
        if kill_reason:
            await self._enter_killed_state(kill_reason)
            return

        # STEP 2: Phase transition check
        transition = self.calibration.check_and_transition()
        if transition:
            phase = BotPhase(self.db.get_current_phase())
            old_phase = transition.get("from_phase", 0)
            new_phase = transition.get("to_phase", 0)
            reason = transition.get("reason", "")
            cal_score = transition.get("calibration_score", None)
            await self.notifier.phase_change_alert(old_phase, new_phase, reason, cal_score)

        # STEP 3: Circuit breaker
        if phase == BotPhase.CIRCUIT_BREAKER:
            if self.cfg.get("ENABLE_EXIT_MANAGER"):
                await self.exit_mgr.check_exits()
            await self.monitor.run_checks()
            return

        # STEP 4: Update correlation clusters
        if self.cfg.get("ENABLE_CORRELATION_CLUSTERS"):
            try:
                positions = self.db.get_open_positions_list()
                self.correlation_mgr.update_clusters(positions)
            except Exception as e:
                cycle_errors.append(f"Correlation update failed: {e}")

        # STEP 5: Check exits on existing positions
        exits_count = 0
        if self.cfg.get("ENABLE_EXIT_MANAGER"):
            try:
                exits_count = await self.exit_mgr.check_exits()
            except Exception as e:
                cycle_errors.append(f"Exit check failed: {e}")

        # STEP 6: Scan markets
        scan_result = await self.scanner.scan_all_markets()
        cycle_errors.extend(scan_result.errors)

        # STEP 7: Process candidates
        placed = 0
        failed = 0
        prefilter_count = 0
        ai_scored_count = 0
        risk_approved_count = 0
        cache_hits = 0
        daily_count = self.db.get_daily_bet_count(date.today())
        max_bets = self.risk.get_phase_limits(phase)["max_bets_per_day"]

        for m in scan_result.candidates[:50]:
            if daily_count + placed >= max_bets:
                break

            # 7a. Deduplication on condition_id
            if self.db.has_active_position(m.condition_id):
                continue

            # 7b. Check scoring cache
            cached = self.scoring_cache.get(m.condition_id)
            ob = None
            hist = []
            news = []
            ob_fresh = False  # True only when enrichment data fetched this cycle
            if cached:
                score = cached
                cache_hits += 1
            else:
                # 7c. Fetch enrichment data
                try:
                    news = await self.news.find_relevant_news(m.question)
                except Exception as e:
                    cycle_errors.append(f"News fetch error for {m.condition_id[:8]}: {e}")
                    news = []

                try:
                    ob = await self.scanner.get_order_book(m.yes_token_id)
                except Exception:
                    ob = None

                try:
                    hist = await self.scanner.get_price_history(m.yes_token_id)
                except Exception:
                    hist = []

                ob_fresh = True  # enrichment data fetched this cycle

                # 7d. Pre-filter
                sentiment = self.news.aggregate_sentiment(news)
                if not self.news.should_call_ai(m.__dict__, news, sentiment, hist):
                    continue
                prefilter_count += 1

                # 7e. AI scoring → RawAIScore
                try:
                    raw_score = await self.ai.score_opportunity(m.__dict__, news, ob, hist)
                except Exception as e:
                    cycle_errors.append(f"AI error for {m.condition_id[:8]}: {e}")
                    continue
                ai_scored_count += 1
                scores_today.append(raw_score)

                # 7f. SCORING PIPELINE: Raw → Debias → Platt → Adjusted
                mkt_price = m.yes_price if raw_score.recommendation in ("BUY_YES",) else m.no_price

                # Debias
                if self.cfg.get("ENABLE_DEBIASER"):
                    debiased = self.debiaser.debias_distribution(
                        raw_score.p10 / 100,  # UNITS: percent → probability
                        raw_score.p50 / 100,  # UNITS: percent → probability
                        raw_score.p90 / 100,  # UNITS: percent → probability
                        mkt_price,
                    )
                else:
                    debiased = {
                        "p10": raw_score.p10 / 100,  # UNITS: percent → probability
                        "p50": raw_score.p50 / 100,  # UNITS: percent → probability
                        "p90": raw_score.p90 / 100,  # UNITS: percent → probability
                    }

                # Platt calibration
                if self.cfg.get("ENABLE_PLATT_SCALING"):
                    adj_p10 = self.calibrator.adjust(debiased["p10"])
                    adj_p50 = self.calibrator.adjust(debiased["p50"])
                    adj_p90 = self.calibrator.adjust(debiased["p90"])
                else:
                    adj_p10, adj_p50, adj_p90 = debiased["p10"], debiased["p50"], debiased["p90"]

                # Build AdjustedScore
                ev = self.fee_calc.calc_ev_with_fees(adj_p50, mkt_price)
                score = AdjustedScore(
                    raw=raw_score,
                    adjusted_p10=adj_p10,          # UNITS: probability 0-1
                    adjusted_p50=adj_p50,          # UNITS: probability 0-1
                    adjusted_p90=adj_p90,          # UNITS: probability 0-1
                    adjusted_probability=adj_p50,  # UNITS: probability 0-1
                    expected_value=ev,
                    recommendation=raw_score.recommendation,
                    ci_width=adj_p90 - adj_p10,
                    debiased_p50=debiased["p50"],   # UNITS: probability 0-1
                )

                # Cache with adaptive TTL
                ttl = self.scoring_cache.get_ttl(m.__dict__, hist)
                self.scoring_cache.set(m.condition_id, score, ttl)

            # 7g. Skip check
            if score.recommendation == "SKIP":
                continue

            # 7h. Sanity gate
            if len(scores_today) >= 10 and not self.calibration.sanity_check(scores_today):
                await self.notifier.sanity_fail_alert(
                    [f"Buy rate or avg prob too high after {len(scores_today)} scores"]
                )
                break

            # 7i. Adverse selection check (skip on cache hit — no fresh ob/hist)
            toxicity = FlowToxicity(0.0, "Disabled", False, [])
            if self.cfg.get("ENABLE_ADVERSE_SELECTION") and ob_fresh and ob:
                try:
                    toxicity = self.adverse.assess_toxicity(m.__dict__, ob, hist)
                    if toxicity.should_skip:
                        continue
                except Exception as e:
                    cycle_errors.append(f"Adverse selection error: {e}")

            # 7j. Book health check (skip on cache hit — no fresh ob/hist)
            book_health = BookHealth(1.0, True, [])
            mkt_price = m.yes_price if score.recommendation == "BUY_YES" else m.no_price
            if self.cfg.get("ENABLE_ORDER_BOOK_HEALTH") and ob_fresh and ob:
                try:
                    book_health = self.book_checker.assess(ob, mkt_price)
                    if not book_health.is_healthy:
                        continue
                except Exception as e:
                    cycle_errors.append(f"Book health error: {e}")

            # 7k. Risk check (all checks must pass)
            risk = self.risk.check_opportunity(score, m, ob, phase, toxicity, book_health)
            if not risk.approved:
                continue
            risk_approved_count += 1

            # 7l. Fee check
            entry_price = m.yes_price if score.recommendation == "BUY_YES" else m.no_price
            if self.cfg.get("ENABLE_FEE_CHECK"):
                if risk.bet_size < self.fee_calc.min_bet_for_positive_ev(entry_price):
                    continue

            # 7m. Compute composite signal (only when enrichment data is fresh)
            composite = self.signal_agg.aggregate(m, news, ob, hist) if ob_fresh else None

            # 7n. Execute order
            side = score.recommendation
            token = m.yes_token_id if side == "BUY_YES" else m.no_token_id
            price = m.yes_price if side == "BUY_YES" else m.no_price
            shares = round(risk.bet_size / price, 2) if price > 0 else 0

            if self.dry_run:
                res = OrderResult(
                    success=True,
                    order_id=f"dry_{self.cycle_counter}_{placed}",
                    error="",
                    price=price,
                    size=shares,
                    side=side,
                    is_dry_run=True,
                )
            else:
                try:
                    res = await self.executor.place_limit_buy(token, price, shares)
                except Exception as e:
                    cycle_errors.append(f"Order execution error: {e}")
                    failed += 1
                    continue

            if res.success:
                placed += 1

                # Record order (dict with named bindings for SQLite)
                self.db.record_order({
                    "order_id": res.order_id,
                    "question": m.question,
                    "condition_id": m.condition_id,
                    "token_id": token,
                    "side": side,
                    "price": price,
                    "size": shares,
                    "total_cost": round(price * shares, 4),
                    "ai_score_raw": score.raw.p50,                     # UNITS: int 0-100 (raw AI %)
                    "ai_score_adjusted": score.adjusted_p50 * 100,     # UNITS: prob→% for display
                    "ai_confidence": score.raw.confidence,
                    "ai_reasoning": score.raw.reasoning[:500],
                    "ai_key_factor": score.raw.key_factor,
                    "p10": score.adjusted_p10,                         # UNITS: probability 0-1
                    "p50": score.adjusted_p50,                         # UNITS: probability 0-1
                    "p90": score.adjusted_p90,                         # UNITS: probability 0-1
                    "toxicity_score": toxicity.score,
                    "book_health_score": book_health.score,
                    "composite_signal": composite.composite_score if composite else 0,
                    "category": m.category,
                    "phase": phase.value,
                    "is_dry_run": 1 if self.dry_run else 0,
                })

                # Record position with end_date for time-decay exits
                self.db.record_position({
                    "condition_id": m.condition_id,
                    "token_id": token,
                    "market_question": m.question,
                    "side": side,
                    "avg_price": price,
                    "shares": shares,
                    "status": "open",
                    "end_date": getattr(m, "end_date", ""),
                })

                # Record prediction (dict with named bindings for SQLite)
                self.db.record_prediction({
                    "condition_id": m.condition_id,
                    "ai_probability_raw": score.raw.p50 / 100,         # UNITS: %→prob
                    "ai_probability_debiased": score.debiased_p50,      # UNITS: probability 0-1
                    "ai_probability_adjusted": score.adjusted_p50,     # UNITS: probability 0-1
                    "market_price": price,                             # UNITS: probability 0-1
                    "p10": score.adjusted_p10,                         # UNITS: probability 0-1
                    "p50": score.adjusted_p50,                         # UNITS: probability 0-1
                    "p90": score.adjusted_p90,                         # UNITS: probability 0-1
                    "ai_confidence": score.raw.confidence,             # UNITS: int 0-100
                    "base_rate": score.raw.base_rate / 100,             # UNITS: %→prob
                    "model_name": "claude_sonnet",
                    "phase": phase.value,
                })

                # Send trade alert (adapt to existing notifier API)
                await self.notifier.trade_alert(
                    side=side,
                    question=m.question[:80],
                    price=price,
                    size=shares,
                    order_id=res.order_id,
                )
            else:
                failed += 1

            await asyncio.sleep(1.5)

        # STEP 8: Position monitor (resolutions, stale cleanup)
        resolutions = 0
        try:
            resolutions = await self.monitor.run_checks()
        except Exception as e:
            cycle_errors.append(f"Monitor error: {e}")

        # STEP 9: Scoring cache cleanup (every 10 cycles) and per-cycle stats
        cache_stats = self.scoring_cache.reset_cycle_stats()
        self._log.info(
            "Cache stats: hits=%d misses=%d rate=%.1f%% size=%d",
            cache_stats["hits"], cache_stats["misses"],
            cache_stats["hit_rate"] * 100, self.scoring_cache.size,
        )
        if self.cycle_counter % 10 == 0:
            removed = self.scoring_cache.cleanup()
            self._log.debug("Cache cleanup: removed %d entries", removed)

        # STEP 10: Log cycle summary
        summary = CycleSummary(
            cycle_id=self.cycle_counter,
            phase=phase.value,
            duration_seconds=round(time.time() - cycle_start, 2),
            markets_scanned=scan_result.total_scanned,
            candidates_found=len(scan_result.candidates),
            prefilter_passed=prefilter_count,
            ai_scored=ai_scored_count,
            cache_hits=cache_hits,
            risk_approved=risk_approved_count,
            orders_placed=placed,
            orders_failed=failed,
            exits_executed=exits_count,
            resolutions_processed=resolutions,
            daily_spend_so_far=self.db.get_daily_spend(date.today()),
            daily_bet_count=daily_count + placed,
            balance=self.db.get_current_balance(),
            calibration_score=self.calibration.get_latest_score(),
            errors=cycle_errors,
        )
        self._log.info("Cycle complete", **summary.__dict__)

    # === GRACEFUL SHUTDOWN ===

    async def stop(self, reason: str = "manual") -> None:
        """Graceful shutdown: cancel orders, save state, notify."""
        self._log.info("Shutdown initiated: %s", reason)
        self.running = False
        await asyncio.sleep(3)  # Wait for current cycle

        if not self.dry_run:
            try:
                await self.executor.cancel_all()
                remaining = await self.executor.get_open_orders()
                if remaining:
                    self._log.warning("%d orders still open after cancel_all", len(remaining))
            except Exception as e:
                self._log.error("Shutdown cancel failed: %s", e)

        # Final state save with calibration and Sharpe
        stats = self.calibration.calculate_calibration_score()
        self.db.update_daily_stats_full(
            date.today(),
            calibration_score=stats.get("calibration_score"),
            sharpe_30d=stats.get("sharpe_30d"),
        )
        phase = self.db.get_current_phase()
        balance = self.db.get_current_balance()
        cal_est = stats.get("point_estimate", 0)

        await self.notifier.shutdown_alert(
            f"{reason} | Phase {phase} | Cal: {cal_est:.3f} | Balance: ${balance:.2f}"
        )
        self._log.info("Bot stopped cleanly")

        # Cleanup
        if self.scanner:
            await self.scanner.close()
        if self.executor and not self.dry_run:
            await self.executor.stop_heartbeat()

    # === MAIN RUN ===

    async def run(self) -> None:
        """Full startup sequence + main loop."""
        _setup_logging()

        # 1. Load and validate config
        cfg = load_config()
        errors = validate_config(cfg)
        if errors:
            for e in errors:
                logger.critical("Config error: %s", e)
            sys.exit(1)

        self.cfg = cfg
        self.dry_run = cfg.get("DRY_RUN", True)

        # 2. Initialize database
        db_path = cfg.get("DB_PATH", "data/polybot.db")
        self.db = Database(db_path)

        # 3. Initialize all modules (order matters)
        self._init_rate_limiters(cfg)
        self._init_modules(cfg)

        # 4. Preflight checks
        passed = await self._preflight_checks()
        if not passed:
            logger.critical("Preflight failed")
            sys.exit(1)

        # 5. Reconcile state (on-chain vs DB)
        reconciled = await self._reconcile_state()
        if not reconciled:
            logger.critical("State reconciliation failed — manual review needed")
            sys.exit(2)

        # 6. Start heartbeat (if not dry run)
        if not self.dry_run:
            await self.executor.start_heartbeat()

        # 7. Notify startup
        phase = self.db.get_current_phase()
        balance = 0.0
        if not self.dry_run:
            try:
                balance = await self.executor.get_balance()
            except Exception:
                balance = self.db.get_current_balance()
        await self.notifier.startup_alert(phase, balance, self.dry_run)

        # 8. Main loop
        self.running = True
        self.cycle_counter = 0

        # Handle SIGTERM/SIGINT for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self.stop(f"signal_{s.name}")),
            )

        while self.running:
            self.cycle_counter += 1
            cycle_start = time.time()
            try:
                await self._cycle()
            except Exception as e:
                self._log.error("Unhandled cycle error: %s", e, exc_info=True)
                await self.notifier.send(
                    f"Cycle {self.cycle_counter} error: {str(e)[:200]}"
                )
            self.last_cycle_time = time.time()
            elapsed = time.time() - cycle_start
            wait = max(1, self.cfg["CYCLE_INTERVAL_SECONDS"] - elapsed)
            await asyncio.sleep(wait)

            # Periodic reconciliation (every 100 cycles)
            if self.cycle_counter % 100 == 0:
                reconciled = await self._reconcile_state()
                if not reconciled:
                    await self._enter_killed_state("Periodic reconciliation failed")


def main():
    bot = MainLoop()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
