"""Tests for modules/order_executor.py."""

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from modules.order_executor import OrderExecutor
from modules.types import OrderResult


def _dry_run_cfg():
    return {"DRY_RUN": True, "TOTAL_BANKROLL": 1000.0}


def _live_cfg():
    return {"DRY_RUN": False, "TOTAL_BANKROLL": 1000.0}


class TestDryRunMode:
    @pytest.mark.asyncio
    async def test_dry_run_buy_no_real_calls(self):
        """Dry-run buy should return success without any real API calls."""
        mock_client = MagicMock()
        executor = OrderExecutor(_dry_run_cfg(), client=mock_client)
        result = await executor.place_limit_buy("tok_001", 0.03, 50.0)
        assert result.success is True
        assert result.is_dry_run is True
        assert result.order_id.startswith("dry-")
        # Client should never be called
        mock_client.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_sell_no_real_calls(self):
        """Dry-run sell should return success without any real API calls."""
        mock_client = MagicMock()
        executor = OrderExecutor(_dry_run_cfg(), client=mock_client)
        result = await executor.place_limit_sell("tok_001", 0.97, 50.0)
        assert result.success is True
        assert result.is_dry_run is True
        assert result.order_id.startswith("dry-")
        mock_client.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_buy_returns_order_result(self):
        """Dry-run buy should return a valid OrderResult."""
        executor = OrderExecutor(_dry_run_cfg())
        result = await executor.place_limit_buy("tok_001", 0.03, 50.0)
        assert isinstance(result, OrderResult)
        assert result.price == 0.03
        assert result.size == 50.0
        assert result.side == "BUY_YES"
        assert result.error == ""

    @pytest.mark.asyncio
    async def test_dry_run_sell_returns_order_result(self):
        """Dry-run sell should return a valid OrderResult."""
        executor = OrderExecutor(_dry_run_cfg())
        result = await executor.place_limit_sell("tok_001", 0.97, 50.0)
        assert isinstance(result, OrderResult)
        assert result.side == "BUY_NO"

    @pytest.mark.asyncio
    async def test_dry_run_cancel_succeeds(self):
        """Cancel in dry-run should return True immediately."""
        executor = OrderExecutor(_dry_run_cfg())
        result = await executor.cancel_order("ord_123")
        assert result is True

    @pytest.mark.asyncio
    async def test_dry_run_get_balance(self):
        """Balance in dry-run should return TOTAL_BANKROLL."""
        executor = OrderExecutor(_dry_run_cfg())
        balance = await executor.get_balance()
        assert balance == 1000.0

    @pytest.mark.asyncio
    async def test_dry_run_get_open_orders_empty(self):
        """Open orders in dry-run should return empty list."""
        executor = OrderExecutor(_dry_run_cfg())
        orders = await executor.get_open_orders()
        assert orders == []


class TestCancelRetries:
    @pytest.mark.asyncio
    async def test_cancel_succeeds_on_first_try(self):
        """Cancel that succeeds immediately should return True."""
        mock_client = MagicMock()
        mock_client.cancel = MagicMock(return_value=None)
        executor = OrderExecutor(_live_cfg(), client=mock_client)
        result = await executor.cancel_order("ord_123")
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_retries_on_failure_then_succeeds(self):
        """Cancel should retry and succeed on second attempt."""
        mock_client = MagicMock()
        call_count = 0

        def cancel_side_effect(order_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            return None

        mock_client.cancel = cancel_side_effect
        executor = OrderExecutor(_live_cfg(), client=mock_client)
        result = await executor.cancel_order("ord_123")
        assert result is True
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cancel_exhausts_retries(self):
        """Cancel should return False after 3 failed attempts."""
        mock_client = MagicMock()
        mock_client.cancel = MagicMock(side_effect=Exception("Permanent failure"))
        executor = OrderExecutor(_live_cfg(), client=mock_client)
        result = await executor.cancel_order("ord_123")
        assert result is False
        assert mock_client.cancel.call_count == 3


class TestPreflightCheck:
    @pytest.mark.asyncio
    async def test_preflight_passes_in_dry_run(self):
        """Preflight should always pass in dry-run."""
        executor = OrderExecutor(_dry_run_cfg())
        result = await executor.preflight_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_preflight_fails_with_zero_balance(self):
        """Preflight with zero balance should fail."""
        mock_client = MagicMock()
        mock_client.get_balance = MagicMock(return_value=0.0)
        executor = OrderExecutor(_live_cfg(), client=mock_client)
        result = await executor.preflight_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_preflight_passes_with_positive_balance(self):
        """Preflight with positive balance should pass."""
        mock_client = MagicMock()
        mock_client.get_balance = MagicMock(return_value=500.0)
        executor = OrderExecutor(_live_cfg(), client=mock_client)
        result = await executor.preflight_check()
        assert result is True


class TestHeartbeatProperty:
    def test_initial_heartbeat_failures_zero(self):
        executor = OrderExecutor(_dry_run_cfg())
        assert executor.heartbeat_failures == 0
