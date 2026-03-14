"""Shared fixtures for the Polymarket Tail Risk Hunter test suite."""

import os
import tempfile

import pytest


@pytest.fixture
def tmp_db_path(tmp_path):
    """Return a temporary database file path."""
    return str(tmp_path / "test_polybot.db")


@pytest.fixture
def valid_config():
    """Return a minimal valid configuration dict."""
    return {
        "POLYMARKET_PRIVATE_KEY": "0x_test_key",
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "TELEGRAM_BOT_TOKEN": "123456:ABC-DEF",
        "TELEGRAM_CHAT_ID": "-100TEST",
        "MAX_PRICE_THRESHOLD": 0.05,
        "TOTAL_BANKROLL": 1000,
        "CYCLE_INTERVAL_SECONDS": 60,
        "PHASE1_MAX_DAILY_SPEND": 20,
        "PHASE2_MAX_DAILY_SPEND": 60,
        "PHASE2_KELLY_FRACTION": 0.25,
        "PHASE3_KELLY_FRACTION": 0.50,
        "MIN_CAL_SCORE_PHASE2": 0.60,
        "MIN_CAL_SCORE_PHASE3": 0.65,
        "DRY_RUN": True,
        "MIN_VOLUME_24H": 100,
        "MIN_LIQUIDITY": 500,
        "MAX_OPEN_POSITIONS": 200,
        "MAX_CATEGORY_PCT": 0.15,
        "MAX_DRAWDOWN_PCT": 0.30,
        "MIN_AI_CONFIDENCE": 60,
        "MIN_EDGE_PCT": 2.0,
        "MIN_EV": 0.03,
    }
