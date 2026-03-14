"""Tests for modules/types.py — enum sanity and dataclass construction."""

import pytest
from modules.types import (
    BotPhase,
    Recommendation,
    MarketCandidate,
    OrderBook,
    PricePoint,
    NewsArticle,
    RawAIScore,
    AdjustedScore,
    FlowToxicity,
    BookHealth,
    RiskDecision,
    OrderResult,
    CalibrationMetrics,
    CycleSummary,
    ScanResult,
    CompositeSignal,
)


# === ENUM TESTS ===


class TestBotPhase:
    def test_values(self):
        assert BotPhase.CALIBRATION == 1
        assert BotPhase.SCALED == 2
        assert BotPhase.FULL == 3
        assert BotPhase.CIRCUIT_BREAKER == 4
        assert BotPhase.KILLED == 5

    def test_is_int(self):
        assert isinstance(BotPhase.CALIBRATION, int)
        assert BotPhase.CALIBRATION + 1 == 2

    def test_all_members(self):
        assert len(BotPhase) == 5


class TestRecommendation:
    def test_values(self):
        assert Recommendation.BUY_YES == "BUY_YES"
        assert Recommendation.BUY_NO == "BUY_NO"
        assert Recommendation.SKIP == "SKIP"

    def test_is_string(self):
        assert isinstance(Recommendation.BUY_YES, str)

    def test_all_members(self):
        assert len(Recommendation) == 3


# === FROZEN DATACLASS TESTS ===


class TestMarketCandidate:
    def test_creation(self):
        mc = MarketCandidate(
            condition_id="abc123",
            question="Will X happen?",
            category="Politics",
            yes_token_id="tok_yes",
            no_token_id="tok_no",
            yes_price=0.03,
            no_price=0.97,
            volume_24h=5000.0,
            liquidity=10000.0,
            end_date="2026-12-31T00:00:00Z",
            tags=["Politics"],
            market_age_hours=48.0,
        )
        assert mc.condition_id == "abc123"
        assert mc.yes_price == 0.03

    def test_frozen(self):
        mc = MarketCandidate(
            condition_id="abc123",
            question="Q",
            category="Cat",
            yes_token_id="y",
            no_token_id="n",
            yes_price=0.03,
            no_price=0.97,
            volume_24h=100.0,
            liquidity=500.0,
            end_date="",
            tags=[],
            market_age_hours=0.0,
        )
        with pytest.raises(AttributeError):
            mc.condition_id = "changed"


class TestOrderBook:
    def test_creation(self):
        ob = OrderBook(
            bids=[(0.03, 100.0), (0.02, 200.0)],
            asks=[(0.04, 100.0), (0.05, 200.0)],
            spread=0.01,
            midpoint=0.035,
        )
        assert ob.spread == 0.01
        assert len(ob.bids) == 2

    def test_frozen(self):
        ob = OrderBook(bids=[], asks=[], spread=0.0, midpoint=0.0)
        with pytest.raises(AttributeError):
            ob.spread = 1.0


class TestPricePoint:
    def test_creation(self):
        pp = PricePoint(timestamp="2026-01-01T00:00:00Z", price=0.03, volume=500.0)
        assert pp.price == 0.03


class TestNewsArticle:
    def test_creation(self):
        na = NewsArticle(
            title="Breaking",
            description="Something happened",
            source="Reuters",
            published_at="2026-01-01T00:00:00Z",
            age_hours=2.5,
            sentiment_label="positive",
            sentiment_score=0.85,
        )
        assert na.sentiment_label == "positive"


# === MUTABLE DATACLASS TESTS ===


class TestRawAIScore:
    def test_creation(self):
        score = RawAIScore(
            base_rate=5,
            p10=2,
            p50=5,
            p90=10,
            confidence=70,
            mispricing="underpriced",
            edge_pct=3.5,
            key_factor="news",
            reasoning="Strong signals",
            evidence_for=["a", "b"],
            evidence_against=["c", "d"],
            recommendation="BUY_YES",
        )
        assert score.p50 == 5
        assert score.recommendation == "BUY_YES"


class TestAdjustedScore:
    def test_creation(self):
        raw = RawAIScore(
            base_rate=5, p10=2, p50=5, p90=10, confidence=70,
            mispricing="underpriced", edge_pct=3.5, key_factor="news",
            reasoning="x", evidence_for=["a"], evidence_against=["b", "c"],
            recommendation="BUY_YES",
        )
        adj = AdjustedScore(
            raw=raw,
            adjusted_p10=0.02,
            adjusted_p50=0.04,
            adjusted_p90=0.08,
            adjusted_probability=0.04,
            expected_value=0.05,
            recommendation="BUY_YES",
            ci_width=0.06,
            debiased_p50=0.035,
        )
        assert adj.adjusted_probability == 0.04
        assert adj.ci_width == 0.06


class TestFlowToxicity:
    def test_creation_and_frozen(self):
        ft = FlowToxicity(score=0.3, reason="Clean flow", should_skip=False, signals=[])
        assert ft.should_skip is False
        with pytest.raises(AttributeError):
            ft.score = 0.9


class TestBookHealth:
    def test_creation(self):
        bh = BookHealth(score=0.7, is_healthy=True, issues=[])
        assert bh.is_healthy is True


class TestRiskDecision:
    def test_creation(self):
        rd = RiskDecision(
            approved=True,
            checks={"position_limit": True, "spend_limit": True},
            failed=[],
            bet_size=1.5,
            sizing_method="flat",
            kelly_raw=0.0,
            estimation_penalty=0.0,
            price_penalty=0.0,
        )
        assert rd.approved is True
        assert rd.bet_size == 1.5


class TestOrderResult:
    def test_creation(self):
        result = OrderResult(
            success=True,
            order_id="ord_123",
            error="",
            price=0.03,
            size=50.0,
            side="BUY_YES",
            is_dry_run=True,
        )
        assert result.success is True
        assert result.side == "BUY_YES"


class TestCalibrationMetrics:
    def test_creation(self):
        cm = CalibrationMetrics(
            calibration_score=0.65,
            ci_lower=0.55,
            ci_upper=0.75,
            ai_brier=0.01,
            market_brier=0.03,
            win_rate_actual=0.05,
            win_rate_expected=0.04,
            win_rate_ratio=1.25,
            sharpe_30d=0.8,
            prediction_count=300,
            phase_recommendation=2,
        )
        assert cm.phase_recommendation == 2


class TestCycleSummary:
    def test_creation(self):
        cs = CycleSummary(
            cycle_id=1,
            phase=1,
            duration_seconds=45.2,
            markets_scanned=500,
            candidates_found=50,
            prefilter_passed=20,
            ai_scored=10,
            cache_hits=5,
            risk_approved=3,
            orders_placed=3,
            orders_failed=0,
            exits_executed=1,
            resolutions_processed=2,
            daily_spend_so_far=12.5,
            daily_bet_count=8,
            balance=995.0,
            calibration_score=0.0,
            errors=[],
        )
        assert cs.markets_scanned == 500


class TestScanResult:
    def test_creation(self):
        sr = ScanResult(
            candidates=[],
            total_scanned=1000,
            total_filtered=950,
            scan_duration_seconds=3.2,
            errors=[],
        )
        assert sr.total_scanned == 1000


class TestCompositeSignal:
    def test_creation(self):
        cs = CompositeSignal(
            composite_score=0.35,
            should_score_with_ai=True,
            signals={"news": (0.5, 0.8), "momentum": (0.2, 0.6)},
        )
        assert cs.should_score_with_ai is True
        assert len(cs.signals) == 2
