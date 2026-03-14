"""Tests for modules/ai_scoring.py — Claude API scoring, parsing, sanity checks."""

import json

import pytest

from modules.ai_scoring import AIScoringEngine, _skip_score, _force_skip, VALID_RECOMMENDATIONS
from modules.rate_limiter import RateLimiter
from modules.types import RawAIScore


def _raw_score(**overrides) -> RawAIScore:
    defaults = dict(
        base_rate=5,
        p10=2,
        p50=5,
        p90=10,
        confidence=60,
        mispricing="underpriced",
        edge_pct=3.0,
        key_factor="breaking news",
        reasoning="Test reasoning",
        evidence_for=["a", "b"],
        evidence_against=["c", "d"],
        recommendation="BUY_YES",
    )
    defaults.update(overrides)
    return RawAIScore(**defaults)


def _valid_json(**overrides) -> str:
    data = {
        "base_rate": 5,
        "p10": 2,
        "p50": 5,
        "p90": 10,
        "confidence": 60,
        "mispricing": "underpriced",
        "edge_pct": 3.0,
        "key_factor": "breaking news",
        "reasoning": "Test reasoning",
        "evidence_for": ["a", "b"],
        "evidence_against": ["c", "d"],
        "recommendation": "BUY_YES",
    }
    data.update(overrides)
    return json.dumps(data)


# === PARSE RESPONSE ===


class TestParseResponse:
    def test_valid_json(self):
        score = AIScoringEngine._parse_response(_valid_json())
        assert score.p50 == 5
        assert score.recommendation == "BUY_YES"

    def test_json_with_code_fences(self):
        text = "```json\n" + _valid_json() + "\n```"
        score = AIScoringEngine._parse_response(text)
        assert score.p50 == 5

    def test_invalid_json_returns_skip(self):
        score = AIScoringEngine._parse_response("not valid json at all")
        assert score.recommendation == "SKIP"
        assert "JSON parse error" in score.reasoning

    def test_truncates_long_reasoning(self):
        text = _valid_json(reasoning="A" * 1000)
        score = AIScoringEngine._parse_response(text)
        assert len(score.reasoning) <= 500

    def test_evidence_capped_at_three(self):
        text = _valid_json(evidence_for=["a", "b", "c", "d", "e"])
        score = AIScoringEngine._parse_response(text)
        assert len(score.evidence_for) <= 3


# === SANITY CHECKS ===


class TestSanityCheck:
    def test_valid_score_passes(self):
        score = _raw_score(p10=2, p50=5, p90=10, recommendation="BUY_YES")
        result = AIScoringEngine._sanity_check(score)
        assert result.recommendation == "BUY_YES"

    def test_p50_above_50_forced_skip(self):
        score = _raw_score(p50=55, p90=60)
        result = AIScoringEngine._sanity_check(score)
        assert result.recommendation == "SKIP"
        assert "FORCED SKIP" in result.reasoning

    def test_non_monotonic_forced_skip(self):
        score = _raw_score(p10=10, p50=5, p90=15)  # p10 > p50
        result = AIScoringEngine._sanity_check(score)
        assert result.recommendation == "SKIP"

    def test_wide_ci_forced_skip(self):
        score = _raw_score(p10=1, p50=12, p90=30)  # width 29 > 20
        result = AIScoringEngine._sanity_check(score)
        assert result.recommendation == "SKIP"

    def test_invalid_recommendation_forced_skip(self):
        score = _raw_score(recommendation="SELL")
        result = AIScoringEngine._sanity_check(score)
        assert result.recommendation == "SKIP"

    def test_skip_recommendation_passes_through(self):
        score = _raw_score(p10=0, p50=1, p90=3, recommendation="SKIP")
        result = AIScoringEngine._sanity_check(score)
        assert result.recommendation == "SKIP"
        assert "FORCED" not in result.reasoning

    def test_buy_no_passes(self):
        score = _raw_score(p10=2, p50=5, p90=10, recommendation="BUY_NO")
        result = AIScoringEngine._sanity_check(score)
        assert result.recommendation == "BUY_NO"

    def test_exact_boundary_ci_width_20_passes(self):
        score = _raw_score(p10=0, p50=10, p90=20)
        result = AIScoringEngine._sanity_check(score)
        assert result.recommendation == "BUY_YES"  # width exactly 20 is OK

    def test_ci_width_21_fails(self):
        score = _raw_score(p10=0, p50=10, p90=21)
        result = AIScoringEngine._sanity_check(score)
        assert result.recommendation == "SKIP"


# === HELPER FUNCTIONS ===


class TestHelpers:
    def test_skip_score(self):
        score = _skip_score("test reason")
        assert score.recommendation == "SKIP"
        assert score.p50 == 0
        assert "test reason" in score.reasoning

    def test_force_skip(self):
        original = _raw_score(recommendation="BUY_YES", reasoning="orig")
        forced = _force_skip(original, "forced reason")
        assert forced.recommendation == "SKIP"
        assert "FORCED SKIP" in forced.reasoning
        assert "orig" in forced.reasoning
        # Other fields preserved
        assert forced.p50 == original.p50

    def test_valid_recommendations_set(self):
        assert VALID_RECOMMENDATIONS == {"BUY_YES", "BUY_NO", "SKIP"}


# === ERROR RATE ===


class TestErrorRate:
    def test_zero_calls(self):
        engine = AIScoringEngine(RateLimiter(max_per_minute=60), {})
        assert engine.error_rate == 0.0

    def test_with_errors(self):
        engine = AIScoringEngine(RateLimiter(max_per_minute=60), {})
        engine._api_calls = 10
        engine._api_errors = 3
        assert engine.error_rate == pytest.approx(0.3)
