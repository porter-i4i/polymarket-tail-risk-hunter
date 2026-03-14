"""Tests for modules/market_scanner.py."""

import json

import pytest

from modules.market_scanner import MarketScanner
from modules.rate_limiter import RateLimiter


def _make_scanner(cfg=None, db=None):
    """Create a MarketScanner with default test config."""
    if cfg is None:
        cfg = {
            "MAX_PRICE_THRESHOLD": 0.05,
            "MIN_VOLUME_24H": 100,
            "MIN_LIQUIDITY": 500,
        }
    gamma_limiter = RateLimiter(max_per_minute=600)
    clob_data_limiter = RateLimiter(max_per_minute=600)
    return MarketScanner(
        cfg=cfg,
        gamma_limiter=gamma_limiter,
        clob_data_limiter=clob_data_limiter,
        db=db,
    )


def _sample_market(
    condition_id="cond_001",
    question="Will X happen?",
    yes_price=0.03,
    no_price=0.97,
    volume_24h=5000,
    liquidity=10000,
    tags=None,
    group_item_title="",
    outcome_prices=None,
    clob_token_ids=None,
):
    """Generate a sample market dict mimicking Gamma API response."""
    if outcome_prices is None:
        outcome_prices = json.dumps([yes_price, no_price])
    if clob_token_ids is None:
        clob_token_ids = json.dumps(["tok_yes_001", "tok_no_001"])
    market = {
        "conditionId": condition_id,
        "question": question,
        "outcomePrices": outcome_prices,
        "clobTokenIds": clob_token_ids,
        "volume24hr": volume_24h,
        "liquidity": liquidity,
        "endDate": "2026-12-31T00:00:00Z",
        "createdAt": "2026-01-01T00:00:00Z",
    }
    if tags is not None:
        market["tags"] = tags
    if group_item_title:
        market["groupItemTitle"] = group_item_title
    return market


class TestJsonLoadsSafety:
    def test_string_outcome_prices_parsed(self):
        """outcomePrices as JSON string should be parsed with json.loads, not eval."""
        scanner = _make_scanner()
        market = _sample_market(outcome_prices=json.dumps([0.03, 0.97]))
        candidate = scanner._parse_market(market)
        assert candidate is not None
        assert candidate.yes_price == 0.03

    def test_list_outcome_prices_handled(self):
        """outcomePrices as native list should also work."""
        scanner = _make_scanner()
        market = _sample_market(outcome_prices=[0.04, 0.96])
        candidate = scanner._parse_market(market)
        assert candidate is not None
        assert candidate.yes_price == 0.04

    def test_malformed_json_does_not_crash(self):
        """Invalid JSON in outcomePrices should skip market, not crash."""
        scanner = _make_scanner()
        market = _sample_market(outcome_prices="not_valid_json")
        candidate = scanner._parse_market(market)
        assert candidate is None

    def test_string_clob_token_ids_parsed(self):
        """clobTokenIds as JSON string should be parsed with json.loads."""
        scanner = _make_scanner()
        market = _sample_market(clob_token_ids=json.dumps(["tok_a", "tok_b"]))
        candidate = scanner._parse_market(market)
        assert candidate is not None
        assert candidate.yes_token_id == "tok_a"


class TestDedupExclusion:
    def test_dedup_excludes_existing_positions(self, tmp_db_path):
        """Markets with existing active positions should be excluded."""
        from modules.portfolio_tracker import Database
        from modules.types import MarketCandidate

        db = Database(db_path=tmp_db_path)
        # Insert an active position for cond_001
        with db.lock:
            db.conn.execute("""
                INSERT INTO positions (condition_id, token_id, market_question, side,
                                       avg_price, shares, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ("cond_001", "tok_yes", "Q", "BUY_YES", 0.03, 50.0, "open"))
            db.conn.commit()

        scanner = _make_scanner(db=db)
        candidates = [
            MarketCandidate(
                condition_id="cond_001", question="Q", category="X",
                yes_token_id="t1", no_token_id="t2", yes_price=0.03,
                no_price=0.97, volume_24h=1000, liquidity=5000,
                end_date="", tags=[], market_age_hours=10,
            ),
            MarketCandidate(
                condition_id="cond_002", question="Q2", category="X",
                yes_token_id="t3", no_token_id="t4", yes_price=0.03,
                no_price=0.97, volume_24h=1000, liquidity=5000,
                end_date="", tags=[], market_age_hours=10,
            ),
        ]
        deduped = scanner._apply_dedup(candidates)
        assert len(deduped) == 1
        assert deduped[0].condition_id == "cond_002"


class TestCategoryResolution:
    def test_tags_label_first_priority(self):
        """tags[0]['label'] takes priority."""
        scanner = _make_scanner()
        market = _sample_market(tags=[{"label": "Politics"}], group_item_title="Sports")
        candidate = scanner._parse_market(market)
        assert candidate.category == "Politics"

    def test_group_item_title_second_priority(self):
        """groupItemTitle used when tags empty."""
        scanner = _make_scanner()
        market = _sample_market(tags=[], group_item_title="Crypto")
        candidate = scanner._parse_market(market)
        assert candidate.category == "Crypto"

    def test_uncategorized_fallback(self):
        """When both tags and groupItemTitle missing, use 'Uncategorized'."""
        scanner = _make_scanner()
        market = _sample_market(tags=[], group_item_title="")
        candidate = scanner._parse_market(market)
        assert candidate.category == "Uncategorized"

    def test_tags_as_string_list(self):
        """Tags as plain string list should also work."""
        scanner = _make_scanner()
        market = _sample_market(tags=["Science"], group_item_title="")
        candidate = scanner._parse_market(market)
        assert candidate.category == "Science"

    def test_tags_with_empty_label(self):
        """Tags with empty label should fall through to groupItemTitle."""
        scanner = _make_scanner()
        market = _sample_market(tags=[{"label": ""}], group_item_title="Finance")
        candidate = scanner._parse_market(market)
        assert candidate.category == "Finance"


class TestFilterCandidates:
    def test_filters_by_price(self):
        """Markets above MAX_PRICE_THRESHOLD should be filtered out."""
        scanner = _make_scanner(cfg={"MAX_PRICE_THRESHOLD": 0.05, "MIN_VOLUME_24H": 0, "MIN_LIQUIDITY": 0})
        markets = [
            _sample_market(yes_price=0.03, no_price=0.97),
            _sample_market(condition_id="cond_002", yes_price=0.10, no_price=0.90,
                           outcome_prices=json.dumps([0.10, 0.90])),
        ]
        candidates = scanner._filter_candidates(markets)
        assert len(candidates) == 1

    def test_filters_by_volume(self):
        """Markets below MIN_VOLUME_24H should be filtered out."""
        scanner = _make_scanner(cfg={"MAX_PRICE_THRESHOLD": 1.0, "MIN_VOLUME_24H": 1000, "MIN_LIQUIDITY": 0})
        markets = [
            _sample_market(volume_24h=5000),
            _sample_market(condition_id="cond_002", volume_24h=50),
        ]
        candidates = scanner._filter_candidates(markets)
        assert len(candidates) == 1
