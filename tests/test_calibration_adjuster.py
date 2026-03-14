"""Tests for modules/calibration_adjuster.py."""

import os
import pickle

import numpy as np
import pytest

from modules.calibration_adjuster import CalibrationAdjuster


class TestIdentityBehavior:
    def test_identity_when_no_model(self, tmp_path):
        """With no model file, adjust should return the input (identity)."""
        model_path = str(tmp_path / "nonexistent.pkl")
        adj = CalibrationAdjuster(cfg=None, model_path=model_path)
        assert adj.adjust(0.5) == 0.5
        assert adj.adjust(0.03) == 0.03
        assert adj.adjust(0.99) == 0.99

    def test_identity_clamped(self, tmp_path):
        """Identity mode should still clamp outputs."""
        model_path = str(tmp_path / "nonexistent.pkl")
        adj = CalibrationAdjuster(cfg=None, model_path=model_path)
        assert adj.adjust(1.5) == 1.0
        assert adj.adjust(-0.1) == 0.0


class TestRetrainAndAdjust:
    def _generate_predictions(self, n=200):
        """Generate synthetic resolved predictions with both classes."""
        preds = []
        for i in range(n):
            p = i / n
            outcome = 1 if p > 0.5 else 0
            # Add some noise
            if i % 7 == 0:
                outcome = 1 - outcome
            preds.append({
                "ai_probability_adjusted": p,
                "outcome": outcome,
            })
        return preds

    def test_retrain_insufficient_samples(self, tmp_path):
        """Retrain should return False with too few samples."""
        model_path = str(tmp_path / "cal.pkl")
        adj = CalibrationAdjuster(
            cfg={"CALIBRATION_MIN_SAMPLES": 100},
            model_path=model_path,
        )
        preds = [{"ai_probability_adjusted": 0.5, "outcome": 1}] * 10
        assert adj.retrain(preds) is False

    def test_retrain_load_adjust_flow(self, tmp_path):
        """Full flow: retrain, save, load, adjust."""
        model_path = str(tmp_path / "cal.pkl")
        adj = CalibrationAdjuster(
            cfg={"CALIBRATION_MIN_SAMPLES": 50, "CALIBRATION_METHOD": "platt"},
            model_path=model_path,
        )
        preds = self._generate_predictions(200)
        result = adj.retrain(preds)
        assert result is True
        assert os.path.exists(model_path)

        # Load in a fresh instance
        adj2 = CalibrationAdjuster(
            cfg={"CALIBRATION_MIN_SAMPLES": 50, "CALIBRATION_METHOD": "platt"},
            model_path=model_path,
        )
        adjusted = adj2.adjust(0.5)
        assert 0.0 <= adjusted <= 1.0

    def test_retrain_isotonic(self, tmp_path):
        """Isotonic regression retrain flow."""
        model_path = str(tmp_path / "cal_iso.pkl")
        adj = CalibrationAdjuster(
            cfg={"CALIBRATION_MIN_SAMPLES": 50, "CALIBRATION_METHOD": "isotonic"},
            model_path=model_path,
        )
        preds = self._generate_predictions(200)
        result = adj.retrain(preds)
        assert result is True

        adj2 = CalibrationAdjuster(
            cfg={"CALIBRATION_MIN_SAMPLES": 50, "CALIBRATION_METHOD": "isotonic"},
            model_path=model_path,
        )
        adjusted = adj2.adjust(0.5)
        assert 0.0 <= adjusted <= 1.0


class TestCorruptedModel:
    def test_corrupted_model_falls_back_to_identity(self, tmp_path):
        """A corrupted .pkl should be deleted and adjuster should fall back to identity."""
        model_path = str(tmp_path / "corrupt.pkl")
        with open(model_path, "wb") as f:
            f.write(b"NOT_A_VALID_PICKLE_FILE_\x00\xff")

        adj = CalibrationAdjuster(cfg=None, model_path=model_path)
        # Model should have been deleted
        assert not os.path.exists(model_path)
        # Should behave as identity
        assert adj.adjust(0.42) == 0.42

    def test_delete_corrupted_model_method(self, tmp_path):
        """Explicit delete_corrupted_model call removes file."""
        model_path = str(tmp_path / "to_delete.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({"fake": True}, f)

        adj = CalibrationAdjuster.__new__(CalibrationAdjuster)
        adj.cfg = {}
        adj.model_path = model_path
        adj.method = "platt"
        adj.min_samples = 100
        adj._model = "something"

        adj.delete_corrupted_model()
        assert adj._model is None
        assert not os.path.exists(model_path)
