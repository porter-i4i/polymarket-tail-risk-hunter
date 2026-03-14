"""
Order Executor: async wrapper around py-clob-client.

Supports dry-run mode, buy/sell/cancel methods, preflight checks, and heartbeat.
All sync py-clob-client calls are wrapped in asyncio.to_thread().
Order placement is never retried. Cancel may be retried up to 3 times.
"""

import asyncio
import logging
import time
import uuid

from modules.types import OrderResult

logger = logging.getLogger(__name__)

# Defensive import: py-clob-client may not be available in test environments
try:
    from py_clob_client.client import ClobClient
    HAS_CLOB_CLIENT = True
except ImportError:
    ClobClient = None
    HAS_CLOB_CLIENT = False


class OrderExecutor:
    def __init__(self, cfg: dict, clob_order_limiter=None, client=None):
        """
        Args:
            cfg: Configuration dict. Must include DRY_RUN.
            clob_order_limiter: Optional RateLimiter for order operations.
            client: Optional ClobClient instance (or mock). If None and not dry-run,
                    will attempt to create one from config.
        """
        self.cfg = cfg
        self.dry_run = cfg.get("DRY_RUN", True)
        self._limiter = clob_order_limiter
        self._client = client
        self._heartbeat_task: asyncio.Task | None = None
        self._heartbeat_failures = 0
        self._running = False

    @property
    def heartbeat_failures(self) -> int:
        return self._heartbeat_failures

    def _get_client(self):
        """Get or create the CLOB client."""
        if self._client is not None:
            return self._client
        if not HAS_CLOB_CLIENT:
            raise RuntimeError("py-clob-client is not installed")
        raise RuntimeError("No CLOB client configured")

    async def _rate_limit(self) -> None:
        if self._limiter is not None:
            await self._limiter.acquire()

    # === ORDER PLACEMENT (never retry) ===

    async def place_limit_buy(self, token_id: str, price: float, size: float) -> OrderResult:
        """Place a limit buy order. Never retried."""
        if self.dry_run:
            return OrderResult(
                success=True,
                order_id=f"dry-{uuid.uuid4().hex[:12]}",
                error="",
                price=price,
                size=size,
                side="BUY_YES",
                is_dry_run=True,
            )

        await self._rate_limit()
        try:
            client = self._get_client()
            result = await asyncio.to_thread(
                client.create_order,
                {
                    "tokenID": token_id,
                    "price": price,
                    "size": size,
                    "side": "BUY",
                },
            )
            order_id = ""
            if isinstance(result, dict):
                order_id = result.get("orderID", result.get("id", ""))
            elif hasattr(result, "orderID"):
                order_id = result.orderID
            return OrderResult(
                success=True,
                order_id=str(order_id),
                error="",
                price=price,
                size=size,
                side="BUY_YES",
                is_dry_run=False,
            )
        except Exception as e:
            logger.error("Buy order failed: %s", e)
            return OrderResult(
                success=False,
                order_id="",
                error=str(e),
                price=price,
                size=size,
                side="BUY_YES",
                is_dry_run=False,
            )

    async def place_limit_sell(self, token_id: str, price: float, size: float) -> OrderResult:
        """Place a limit sell order. Never retried."""
        if self.dry_run:
            return OrderResult(
                success=True,
                order_id=f"dry-{uuid.uuid4().hex[:12]}",
                error="",
                price=price,
                size=size,
                side="BUY_NO",
                is_dry_run=True,
            )

        await self._rate_limit()
        try:
            client = self._get_client()
            result = await asyncio.to_thread(
                client.create_order,
                {
                    "tokenID": token_id,
                    "price": price,
                    "size": size,
                    "side": "SELL",
                },
            )
            order_id = ""
            if isinstance(result, dict):
                order_id = result.get("orderID", result.get("id", ""))
            elif hasattr(result, "orderID"):
                order_id = result.orderID
            return OrderResult(
                success=True,
                order_id=str(order_id),
                error="",
                price=price,
                size=size,
                side="BUY_NO",
                is_dry_run=False,
            )
        except Exception as e:
            logger.error("Sell order failed: %s", e)
            return OrderResult(
                success=False,
                order_id="",
                error=str(e),
                price=price,
                size=size,
                side="BUY_NO",
                is_dry_run=False,
            )

    # === CANCEL (up to 3 retries with exponential backoff) ===

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order. Retried up to 3 times with exponential backoff."""
        if self.dry_run:
            return True

        backoff = [1, 2, 4]
        for attempt in range(3):
            try:
                await self._rate_limit()
                client = self._get_client()
                await asyncio.to_thread(client.cancel, order_id)
                return True
            except Exception as e:
                logger.warning("Cancel attempt %d failed for %s: %s", attempt + 1, order_id, e)
                if attempt < 2:
                    await asyncio.sleep(backoff[attempt])
        return False

    async def cancel_all(self) -> None:
        """Cancel all open orders."""
        if self.dry_run:
            return

        try:
            client = self._get_client()
            await asyncio.to_thread(client.cancel_all)
        except Exception as e:
            logger.error("Cancel all failed: %s", e)

    # === QUERIES ===

    async def get_open_orders(self) -> list:
        """Get all open orders."""
        if self.dry_run:
            return []

        try:
            client = self._get_client()
            return await asyncio.to_thread(client.get_orders)
        except Exception as e:
            logger.error("Failed to get open orders: %s", e)
            return []

    async def get_balance(self) -> float:
        """Get current USDC balance."""
        if self.dry_run:
            return self.cfg.get("TOTAL_BANKROLL", 1000.0)

        try:
            client = self._get_client()
            result = await asyncio.to_thread(client.get_balance)
            return float(result)
        except Exception as e:
            logger.error("Failed to get balance: %s", e)
            return 0.0

    # === PREFLIGHT ===

    async def preflight_check(self) -> bool:
        """Verify the executor is ready to operate."""
        if self.dry_run:
            return True

        try:
            balance = await self.get_balance()
            if balance <= 0:
                logger.error("Preflight failed: zero balance")
                return False
            return True
        except Exception as e:
            logger.error("Preflight check failed: %s", e)
            return False

    # === HEARTBEAT ===

    async def start_heartbeat(self) -> None:
        """Start the heartbeat loop (5s interval)."""
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop_heartbeat(self) -> None:
        """Stop the heartbeat loop."""
        self._running = False
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat to verify connectivity."""
        while self._running:
            try:
                await asyncio.sleep(5)
                if not self._running:
                    break
                if self.dry_run:
                    continue
                await self.get_balance()
                self._heartbeat_failures = 0
            except asyncio.CancelledError:
                break
            except Exception:
                self._heartbeat_failures += 1
                logger.warning("Heartbeat failure #%d", self._heartbeat_failures)
