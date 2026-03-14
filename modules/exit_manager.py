"""
Exit Manager: evaluates open positions against profit, stop-loss, and time-decay rules.

Executes sells via executor.place_limit_sell. Retries once on first failure.
On success, closes position in DB and notifies.
"""

import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ExitManager:
    def __init__(self, executor, db, scanner, notifier, cfg: dict):
        self.executor = executor
        self.db = db
        self.scanner = scanner
        self.notifier = notifier
        self.cfg = cfg

    async def check_exits(self) -> int:
        """Evaluate all open positions for exit conditions. Returns exit count."""
        positions = self.db.get_open_positions_list()
        exit_count = 0
        for pos in positions:
            try:
                exited = await self._evaluate_position(pos)
                if exited:
                    exit_count += 1
            except Exception as e:
                logger.error("Error evaluating position %s: %s",
                             pos.get("condition_id", "?"), e)
        return exit_count

    async def _evaluate_position(self, pos: dict) -> bool:
        """Check exit rules in priority order. Returns True if exited."""
        token_id = pos.get("token_id", "")
        avg_price = pos.get("avg_price", 0)
        shares = pos.get("shares", 0)

        if not token_id or avg_price <= 0 or shares <= 0:
            return False

        current_price = await self.scanner.get_current_price(token_id)
        if current_price is None or current_price <= 0:
            return False

        # Rule 1: Profit target
        profit_multiplier = self.cfg.get("EXIT_PROFIT_MULTIPLIER", 3.0)
        if current_price >= avg_price * profit_multiplier:
            return await self._execute_exit(
                pos, current_price, shares, "profit_target"
            )

        # Rule 2: Stop loss
        stop_loss_pct = self.cfg.get("EXIT_STOP_LOSS_PCT", 0.70)
        if current_price <= avg_price * (1 - stop_loss_pct):
            return await self._execute_exit(
                pos, current_price, shares, "stop_loss"
            )

        # Rule 3: Time decay
        end_date_str = pos.get("end_date", "")
        if end_date_str:
            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                hours_remaining = (end_dt - now).total_seconds() / 3600.0
                decay_hours = self.cfg.get("EXIT_TIME_DECAY_HOURS", 24)
                if hours_remaining <= decay_hours:
                    # Only exit on time decay if price hasn't moved much
                    price_change = abs(current_price - avg_price) / avg_price
                    if price_change < 0.5:
                        return await self._execute_exit(
                            pos, current_price, shares, "time_decay"
                        )
            except (ValueError, TypeError):
                pass

        return False

    async def _execute_exit(self, pos: dict, sell_price: float,
                            shares: float, reason: str) -> bool:
        """Execute a sell order. Retry once on failure."""
        condition_id = pos.get("condition_id", "")
        token_id = pos.get("token_id", "")
        avg_price = pos.get("avg_price", 0)
        question = pos.get("market_question", "Unknown")

        result = await self.executor.place_limit_sell(token_id, sell_price, shares)

        if not result.success:
            logger.warning("First sell attempt failed for %s, retrying in 30s", condition_id)
            await asyncio.sleep(30)
            result = await self.executor.place_limit_sell(token_id, sell_price, shares)

        if result.success:
            pnl = (sell_price - avg_price) * shares
            self.db.close_position(condition_id, reason, pnl)
            try:
                await self.notifier.exit_alert(question, reason, pnl)
            except Exception as e:
                logger.warning("Exit notification failed: %s", e)
            return True

        logger.error("Exit failed after retry for %s", condition_id)
        return False
