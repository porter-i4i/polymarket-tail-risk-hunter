"""
Position Monitor: polls unresolved predictions for resolution, cleans up stale orders.

Triggers calibrator retrain when resolution threshold is reached.
"""

import json
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class PositionMonitor:
    def __init__(self, db, scanner, calibrator, notifier, cfg: dict):
        self.db = db
        self.scanner = scanner
        self.calibrator = calibrator
        self.notifier = notifier
        self.cfg = cfg
        self._resolutions_since_retrain = 0

    async def run_checks(self) -> int:
        """Run resolution checks and stale order cleanup. Returns resolutions processed."""
        resolutions = await self._check_resolutions()
        await self._cleanup_stale_orders()
        return resolutions

    async def _check_resolutions(self) -> int:
        """Poll unresolved predictions for market resolution."""
        unresolved = self.db.get_unresolved_predictions()
        resolved_count = 0

        for pred in unresolved:
            condition_id = pred.get("condition_id", "")
            if not condition_id:
                continue

            try:
                market_data = await self.scanner.get_market_by_condition(condition_id)
                if market_data is None:
                    continue

                if not market_data.get("resolved", False):
                    continue

                # Determine outcome from outcomePrices
                outcome_prices_raw = market_data.get("outcomePrices", "[]")
                if isinstance(outcome_prices_raw, str):
                    outcome_prices = json.loads(outcome_prices_raw)
                else:
                    outcome_prices = outcome_prices_raw

                if not outcome_prices or len(outcome_prices) < 2:
                    continue

                yes_price = float(outcome_prices[0])
                # YES resolved if yes_price ~= 1.0
                outcome = 1 if yes_price > 0.5 else 0

                # Calculate PnL
                ai_prob = pred.get("ai_probability_adjusted") or pred.get("ai_probability_raw", 0)
                market_price = pred.get("market_price", 0)
                pnl = (1.0 - market_price) if outcome == 1 else -market_price

                # Update prediction
                self.db.resolve_prediction(condition_id, outcome, pnl)

                # Update related orders
                orders = self.db.get_orders_by_condition(condition_id)
                for order in orders:
                    self.db.update_order_status(
                        order["order_id"], "resolved",
                        filled_at=datetime.now(timezone.utc).isoformat(), pnl=pnl
                    )

                # Close position
                self.db.close_position(condition_id, "resolved", pnl)

                resolved_count += 1
                self._resolutions_since_retrain += 1

                # Trigger retrain if threshold reached
                retrain_every = self.cfg.get("CALIBRATION_RETRAIN_EVERY", 50)
                min_samples = self.cfg.get("CALIBRATION_MIN_SAMPLES", 100)
                total_resolved = len(self.db.get_resolved_predictions())

                if (self._resolutions_since_retrain >= retrain_every
                        and total_resolved >= min_samples):
                    try:
                        self.calibrator.retrain()
                        self._resolutions_since_retrain = 0
                    except Exception as e:
                        logger.warning("Calibrator retrain failed: %s", e)

            except Exception as e:
                logger.error("Error checking resolution for %s: %s", condition_id, e)

        return resolved_count

    async def _cleanup_stale_orders(self) -> None:
        """Cancel pending orders older than 2 hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
        stale_orders = self.db.get_stale_pending_orders(cutoff)

        for order in stale_orders:
            order_id = order.get("order_id", "")
            if not order_id:
                continue
            try:
                cancelled = await self.scanner.executor.cancel_order(order_id)
                if cancelled:
                    self.db.update_order_status(order_id, "cancelled")
                    logger.info("Cancelled stale order: %s", order_id)
            except Exception as e:
                logger.warning("Failed to cancel stale order %s: %s", order_id, e)
