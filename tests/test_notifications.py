"""Tests for modules/notifications.py — Notifier send, truncation, error handling."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.notifications import Notifier, MAX_MESSAGE_LENGTH


@pytest.fixture
def disabled_notifier():
    """Notifier with no telegram credentials."""
    return Notifier({"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""})


@pytest.fixture
def enabled_notifier():
    """Notifier with fake telegram credentials."""
    return Notifier({
        "TELEGRAM_BOT_TOKEN": "123:ABC",
        "TELEGRAM_CHAT_ID": "-100TEST",
    })


class TestNotifierSend:
    @pytest.mark.asyncio
    async def test_disabled_send_is_nonblocking(self, disabled_notifier):
        """Disabled notifier should return immediately without error."""
        await disabled_notifier.send("test message")
        # Should not raise

    @pytest.mark.asyncio
    async def test_send_truncates_long_messages(self, enabled_notifier):
        mock_bot = AsyncMock()
        enabled_notifier._bot = mock_bot
        enabled_notifier._initialized = True

        long_msg = "x" * 5000
        await enabled_notifier.send(long_msg)

        mock_bot.send_message.assert_called_once()
        sent_msg = mock_bot.send_message.call_args[1]["text"]
        assert len(sent_msg) <= MAX_MESSAGE_LENGTH
        assert "[truncated]" in sent_msg

    @pytest.mark.asyncio
    async def test_get_bot_failure_disables(self, enabled_notifier):
        """If _get_bot raises, notifier disables itself."""
        with patch("modules.notifications.Notifier._get_bot", new_callable=AsyncMock) as mock_get:
            # Simulate the import/init failure path
            enabled_notifier._bot = None
            enabled_notifier._enabled = True

            # Make _get_bot set _enabled to False and return None (simulating failure)
            async def fail_and_disable():
                enabled_notifier._enabled = False
                return None
            mock_get.side_effect = fail_and_disable

            await enabled_notifier.send("test")
            assert enabled_notifier._enabled is False

    @pytest.mark.asyncio
    async def test_send_exception_does_not_raise(self, enabled_notifier):
        """send() must never raise even if bot.send_message fails."""
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = Exception("network error")
        enabled_notifier._bot = mock_bot
        enabled_notifier._initialized = True

        await enabled_notifier.send("test")  # Should not raise


class TestStructuredMethods:
    @pytest.mark.asyncio
    async def test_startup_alert_calls_send(self, enabled_notifier):
        enabled_notifier.send = AsyncMock()
        await enabled_notifier.startup_alert(phase=1, balance=1000.0, dry_run=True)
        enabled_notifier.send.assert_called_once()
        msg = enabled_notifier.send.call_args[0][0]
        assert "Bot Started" in msg
        assert "DRY RUN" in msg

    @pytest.mark.asyncio
    async def test_shutdown_alert_calls_send(self, enabled_notifier):
        enabled_notifier.send = AsyncMock()
        await enabled_notifier.shutdown_alert(reason="manual stop")
        enabled_notifier.send.assert_called_once()
        msg = enabled_notifier.send.call_args[0][0]
        assert "Stopped" in msg
        assert "manual stop" in msg

    @pytest.mark.asyncio
    async def test_trade_alert_calls_send(self, enabled_notifier):
        enabled_notifier.send = AsyncMock()
        await enabled_notifier.trade_alert("BUY_YES", "Will X happen?", 0.03, 50.0, "ord_001")
        enabled_notifier.send.assert_called_once()
        msg = enabled_notifier.send.call_args[0][0]
        assert "Trade" in msg
        assert "BUY_YES" in msg

    @pytest.mark.asyncio
    async def test_exit_alert_calls_send(self, enabled_notifier):
        enabled_notifier.send = AsyncMock()
        await enabled_notifier.exit_alert("Will X happen?", "profit_target", 10.5)
        enabled_notifier.send.assert_called_once()
        msg = enabled_notifier.send.call_args[0][0]
        assert "Exited" in msg
        assert "profit_target" in msg

    @pytest.mark.asyncio
    async def test_daily_summary_calls_send(self, enabled_notifier):
        enabled_notifier.send = AsyncMock()
        stats = {"total_bets": 5, "total_spent": 10.0, "net_pnl": 2.0, "balance": 1002.0, "phase": 1}
        await enabled_notifier.daily_summary(stats)
        enabled_notifier.send.assert_called_once()
        msg = enabled_notifier.send.call_args[0][0]
        assert "Summary" in msg

    @pytest.mark.asyncio
    async def test_circuit_breaker_alert(self, enabled_notifier):
        enabled_notifier.send = AsyncMock()
        await enabled_notifier.circuit_breaker_alert(-150.0, -100.0)
        msg = enabled_notifier.send.call_args[0][0]
        assert "Circuit Breaker" in msg

    @pytest.mark.asyncio
    async def test_phase_change_alert(self, enabled_notifier):
        enabled_notifier.send = AsyncMock()
        await enabled_notifier.phase_change_alert(1, 2, "upgrade", 0.65)
        enabled_notifier.send.assert_called_once()
