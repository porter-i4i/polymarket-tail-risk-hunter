"""Tests for config.py — validation happy path and failure cases."""

import os

import pytest

from config import load_config, validate_config, RETRY_POLICIES


# === RETRY_POLICIES TESTS ===


class TestRetryPolicies:
    def test_all_policies_present(self):
        expected = {"gamma_api", "clob_read", "clob_write", "claude_api", "news_api", "telegram_api"}
        assert set(RETRY_POLICIES.keys()) == expected

    def test_gamma_api_policy(self):
        p = RETRY_POLICIES["gamma_api"]
        assert p["max_attempts"] == 3
        assert p["jitter"] is True
        assert 429 in p["retryable_status"]

    def test_clob_write_no_retry(self):
        p = RETRY_POLICIES["clob_write"]
        assert p["max_attempts"] == 1

    def test_claude_api_overloaded(self):
        p = RETRY_POLICIES["claude_api"]
        assert 529 in p["retryable_status"]


# === VALIDATE_CONFIG HAPPY PATH ===


class TestValidateConfigHappy:
    def test_valid_config_returns_empty(self, valid_config):
        errors = validate_config(valid_config)
        assert errors == []

    def test_valid_config_boundary_low(self, valid_config):
        valid_config["MAX_PRICE_THRESHOLD"] = 0.01
        valid_config["TOTAL_BANKROLL"] = 50
        valid_config["CYCLE_INTERVAL_SECONDS"] = 10
        errors = validate_config(valid_config)
        assert errors == []

    def test_valid_config_boundary_high(self, valid_config):
        valid_config["MAX_PRICE_THRESHOLD"] = 0.20
        valid_config["TOTAL_BANKROLL"] = 100000
        valid_config["CYCLE_INTERVAL_SECONDS"] = 600
        errors = validate_config(valid_config)
        assert errors == []


# === VALIDATE_CONFIG FAILURE CASES ===


class TestValidateConfigFailures:
    def test_missing_private_key(self, valid_config):
        valid_config["POLYMARKET_PRIVATE_KEY"] = ""
        errors = validate_config(valid_config)
        assert any("POLYMARKET_PRIVATE_KEY" in e for e in errors)

    def test_missing_anthropic_key(self, valid_config):
        valid_config["ANTHROPIC_API_KEY"] = ""
        errors = validate_config(valid_config)
        assert any("ANTHROPIC_API_KEY" in e for e in errors)

    def test_missing_telegram_token(self, valid_config):
        valid_config["TELEGRAM_BOT_TOKEN"] = ""
        errors = validate_config(valid_config)
        assert any("TELEGRAM_BOT_TOKEN" in e for e in errors)

    def test_missing_telegram_chat_id(self, valid_config):
        valid_config["TELEGRAM_CHAT_ID"] = ""
        errors = validate_config(valid_config)
        assert any("TELEGRAM_CHAT_ID" in e for e in errors)

    def test_all_secrets_missing(self, valid_config):
        for key in ["POLYMARKET_PRIVATE_KEY", "ANTHROPIC_API_KEY",
                     "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]:
            valid_config[key] = ""
        errors = validate_config(valid_config)
        assert len(errors) == 4

    def test_price_threshold_too_low(self, valid_config):
        valid_config["MAX_PRICE_THRESHOLD"] = 0.005
        errors = validate_config(valid_config)
        assert any("MAX_PRICE_THRESHOLD" in e for e in errors)

    def test_price_threshold_too_high(self, valid_config):
        valid_config["MAX_PRICE_THRESHOLD"] = 0.25
        errors = validate_config(valid_config)
        assert any("MAX_PRICE_THRESHOLD" in e for e in errors)

    def test_bankroll_too_low(self, valid_config):
        valid_config["TOTAL_BANKROLL"] = 10
        errors = validate_config(valid_config)
        assert any("TOTAL_BANKROLL" in e for e in errors)

    def test_bankroll_too_high(self, valid_config):
        valid_config["TOTAL_BANKROLL"] = 200000
        errors = validate_config(valid_config)
        assert any("TOTAL_BANKROLL" in e for e in errors)

    def test_cycle_interval_too_low(self, valid_config):
        valid_config["CYCLE_INTERVAL_SECONDS"] = 5
        errors = validate_config(valid_config)
        assert any("CYCLE_INTERVAL" in e for e in errors)

    def test_cycle_interval_too_high(self, valid_config):
        valid_config["CYCLE_INTERVAL_SECONDS"] = 1000
        errors = validate_config(valid_config)
        assert any("CYCLE_INTERVAL" in e for e in errors)

    def test_phase_spend_not_increasing(self, valid_config):
        valid_config["PHASE1_MAX_DAILY_SPEND"] = 60
        valid_config["PHASE2_MAX_DAILY_SPEND"] = 60
        errors = validate_config(valid_config)
        assert any("Phase 1 spend" in e for e in errors)

    def test_phase_spend_inverted(self, valid_config):
        valid_config["PHASE1_MAX_DAILY_SPEND"] = 100
        valid_config["PHASE2_MAX_DAILY_SPEND"] = 20
        errors = validate_config(valid_config)
        assert any("Phase 1 spend" in e for e in errors)

    def test_kelly_fractions_not_increasing(self, valid_config):
        valid_config["PHASE2_KELLY_FRACTION"] = 0.50
        valid_config["PHASE3_KELLY_FRACTION"] = 0.25
        errors = validate_config(valid_config)
        assert any("Kelly fractions" in e for e in errors)

    def test_kelly_fractions_equal(self, valid_config):
        valid_config["PHASE2_KELLY_FRACTION"] = 0.50
        valid_config["PHASE3_KELLY_FRACTION"] = 0.50
        errors = validate_config(valid_config)
        assert any("Kelly fractions" in e for e in errors)

    def test_cal_score_thresholds_not_increasing(self, valid_config):
        valid_config["MIN_CAL_SCORE_PHASE2"] = 0.70
        valid_config["MIN_CAL_SCORE_PHASE3"] = 0.65
        errors = validate_config(valid_config)
        assert any("Cal score thresholds" in e for e in errors)

    def test_cal_score_thresholds_equal(self, valid_config):
        valid_config["MIN_CAL_SCORE_PHASE2"] = 0.65
        valid_config["MIN_CAL_SCORE_PHASE3"] = 0.65
        errors = validate_config(valid_config)
        assert any("Cal score thresholds" in e for e in errors)

    def test_multiple_errors(self, valid_config):
        valid_config["POLYMARKET_PRIVATE_KEY"] = ""
        valid_config["MAX_PRICE_THRESHOLD"] = 0.50
        valid_config["TOTAL_BANKROLL"] = 5
        errors = validate_config(valid_config)
        assert len(errors) >= 3


# === LOAD_CONFIG TESTS ===


class TestLoadConfig:
    def test_defaults_loaded(self, monkeypatch):
        # Clear relevant env vars to use defaults
        for key in ["POLYMARKET_PRIVATE_KEY", "ANTHROPIC_API_KEY",
                     "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]:
            monkeypatch.delenv(key, raising=False)
        cfg = load_config()
        assert cfg["TOTAL_BANKROLL"] == 1000
        assert cfg["CYCLE_INTERVAL_SECONDS"] == 60
        assert cfg["DRY_RUN"] is True
        assert cfg["MAX_PRICE_THRESHOLD"] == 0.05

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("TOTAL_BANKROLL", "5000")
        monkeypatch.setenv("DRY_RUN", "false")
        monkeypatch.setenv("CYCLE_INTERVAL_SECONDS", "120")
        cfg = load_config()
        assert cfg["TOTAL_BANKROLL"] == 5000
        assert cfg["DRY_RUN"] is False
        assert cfg["CYCLE_INTERVAL_SECONDS"] == 120

    def test_bool_parsing(self, monkeypatch):
        monkeypatch.setenv("DRY_RUN", "true")
        cfg = load_config()
        assert cfg["DRY_RUN"] is True

        monkeypatch.setenv("DRY_RUN", "1")
        cfg = load_config()
        assert cfg["DRY_RUN"] is True

        monkeypatch.setenv("DRY_RUN", "yes")
        cfg = load_config()
        assert cfg["DRY_RUN"] is True

        monkeypatch.setenv("DRY_RUN", "no")
        cfg = load_config()
        assert cfg["DRY_RUN"] is False

    def test_feature_flags(self, monkeypatch):
        monkeypatch.setenv("ENABLE_DEBIASER", "false")
        cfg = load_config()
        assert cfg["ENABLE_DEBIASER"] is False
