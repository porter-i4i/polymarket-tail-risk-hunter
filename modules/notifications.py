"""
Telegram notification system for Polymarket Tail Risk Hunter.

Lazy-loaded, optional, never raises from send(). If telegram package is absent
or initialization fails, the notifier disables itself and logs a warning.
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4000


class Notifier:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._bot = None
        self._enabled = bool(cfg.get("TELEGRAM_BOT_TOKEN") and cfg.get("TELEGRAM_CHAT_ID"))
        self._chat_id = cfg.get("TELEGRAM_CHAT_ID", "")
        self._initialized = False

    async def _get_bot(self):
        """Lazy-load telegram bot. Disables on failure."""
        if self._bot is not None:
            return self._bot
        if not self._enabled:
            return None
        try:
            import telegram
            self._bot = telegram.Bot(token=self.cfg["TELEGRAM_BOT_TOKEN"])
            self._initialized = True
            return self._bot
        except Exception as e:
            logger.warning("Telegram init failed, disabling notifications: %s", e)
            self._enabled = False
            return None

    async def send(self, message: str) -> None:
        """Send a message. Never raises."""
        try:
            if not self._enabled:
                return
            bot = await self._get_bot()
            if bot is None:
                return
            # Truncate long messages
            if len(message) > MAX_MESSAGE_LENGTH:
                message = message[:MAX_MESSAGE_LENGTH - 20] + "\n... [truncated]"
            await bot.send_message(chat_id=self._chat_id, text=message)
        except Exception as e:
            logger.warning("Failed to send notification: %s", e)

    # === STRUCTURED METHODS ===

    async def startup_alert(self, phase: int, balance: float, dry_run: bool) -> None:
        mode = "DRY RUN" if dry_run else "LIVE"
        msg = (
            f"🚀 Bot Started [{mode}]\n"
            f"Phase: {phase}\n"
            f"Balance: ${balance:.2f}\n"
            f"Time: {datetime.now(timezone.utc).isoformat()}"
        )
        await self.send(msg)

    async def shutdown_alert(self, reason: str) -> None:
        msg = (
            f"🛑 Bot Stopped\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now(timezone.utc).isoformat()}"
        )
        await self.send(msg)

    async def trade_alert(self, side: str, question: str, price: float,
                          size: float, order_id: str) -> None:
        msg = (
            f"📊 Trade Executed\n"
            f"Side: {side}\n"
            f"Market: {question[:100]}\n"
            f"Price: {price:.4f}\n"
            f"Size: {size:.2f}\n"
            f"Order: {order_id}"
        )
        await self.send(msg)

    async def phase_change_alert(self, from_phase: int, to_phase: int,
                                 reason: str, cal_score: float = None) -> None:
        msg = (
            f"🔄 Phase Change: {from_phase} → {to_phase}\n"
            f"Reason: {reason}\n"
            f"Cal Score: {cal_score:.3f}" if cal_score is not None else
            f"🔄 Phase Change: {from_phase} → {to_phase}\n"
            f"Reason: {reason}"
        )
        await self.send(msg)

    async def circuit_breaker_alert(self, rolling_pnl: float, threshold: float) -> None:
        msg = (
            f"⚠️ Circuit Breaker Triggered\n"
            f"Rolling PnL: ${rolling_pnl:.2f}\n"
            f"Threshold: ${threshold:.2f}"
        )
        await self.send(msg)

    async def kill_switch_alert(self, reason: str) -> None:
        msg = (
            f"🚨 KILL SWITCH ACTIVATED\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now(timezone.utc).isoformat()}"
        )
        await self.send(msg)

    async def sanity_fail_alert(self, failures: list[str]) -> None:
        msg = (
            f"⚠️ Sanity Check Failed\n"
            f"Failures: {', '.join(failures)}"
        )
        await self.send(msg)

    async def exit_alert(self, question: str, reason: str, pnl: float) -> None:
        msg = (
            f"🚪 Position Exited\n"
            f"Market: {question[:100]}\n"
            f"Reason: {reason}\n"
            f"PnL: ${pnl:.2f}"
        )
        await self.send(msg)

    async def daily_summary(self, stats: dict) -> None:
        msg = (
            f"📈 Daily Summary\n"
            f"Bets: {stats.get('total_bets', 0)}\n"
            f"Spent: ${stats.get('total_spent', 0):.2f}\n"
            f"PnL: ${stats.get('net_pnl', 0):.2f}\n"
            f"Balance: ${stats.get('balance', 0):.2f}\n"
            f"Phase: {stats.get('phase', 1)}"
        )
        await self.send(msg)
