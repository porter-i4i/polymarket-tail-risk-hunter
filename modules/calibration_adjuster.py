"""
Calibration Adjuster: Platt-style calibration layer with persistence.

Uses scikit-learn LogisticRegression (Platt) or IsotonicRegression.
Model saved to data/calibrator.pkl.
Falls back to identity when model unavailable/disabled.
"""

import logging
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = "data/calibrator.pkl"
_DEFAULT_MIN_SAMPLES = 100


class CalibrationAdjuster:
    def __init__(self, cfg: dict | None = None, model_path: str = _DEFAULT_MODEL_PATH):
        self.cfg = cfg or {}
        self.model_path = model_path
        self.method = self.cfg.get("CALIBRATION_METHOD", "platt")
        self.min_samples = self.cfg.get("CALIBRATION_MIN_SAMPLES", _DEFAULT_MIN_SAMPLES)
        self._model = None
        self.load()

    def adjust(self, debiased_probability: float) -> float:
        """
        Apply calibration to a debiased probability.
        Returns identity when model unavailable or feature disabled.
        """
        if self._model is None:
            return max(0.0, min(1.0, debiased_probability))

        try:
            X = np.array([[debiased_probability]])
            if self.method == "isotonic":
                result = float(self._model.predict(X)[0])
            else:
                result = float(self._model.predict_proba(X)[0, 1])
            return max(0.0, min(1.0, result))
        except Exception:
            logger.warning("Calibration model prediction failed, using identity")
            return max(0.0, min(1.0, debiased_probability))

    def retrain(self, resolved_predictions: list[dict]) -> bool:
        """
        Retrain calibration model on resolved predictions.
        Each dict must have 'ai_probability_adjusted' (or 'ai_probability_debiased') and 'outcome'.
        Returns True if model was successfully retrained, False otherwise.
        """
        if len(resolved_predictions) < self.min_samples:
            logger.info(
                "Insufficient samples for calibration: %d < %d",
                len(resolved_predictions), self.min_samples,
            )
            return False

        try:
            predicted = []
            outcomes = []
            for pred in resolved_predictions:
                p = pred.get("ai_probability_adjusted") or pred.get("ai_probability_debiased")
                outcome = pred.get("outcome")
                if p is not None and outcome is not None:
                    predicted.append(float(p))
                    outcomes.append(int(outcome))

            if len(predicted) < self.min_samples:
                return False

            X = np.array(predicted).reshape(-1, 1)
            y = np.array(outcomes)

            # Need both classes present
            if len(set(y)) < 2:
                logger.warning("Cannot calibrate: only one class present in outcomes")
                return False

            if self.method == "isotonic":
                model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                model.fit(X.ravel(), y)
            else:
                model = LogisticRegression(solver="lbfgs", max_iter=1000)
                model.fit(X, y)

            self._model = model
            self._save()
            return True

        except Exception:
            logger.exception("Calibration retrain failed")
            return False

    def load(self) -> bool:
        """Load model from disk. Returns True if successful."""
        if not os.path.exists(self.model_path):
            return False

        try:
            with open(self.model_path, "rb") as f:
                self._model = pickle.load(f)
            return True
        except Exception:
            logger.warning("Corrupted calibrator model at %s, deleting", self.model_path)
            self.delete_corrupted_model()
            return False

    def delete_corrupted_model(self) -> None:
        """Delete corrupted model file and revert to identity."""
        self._model = None
        try:
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
        except OSError:
            logger.warning("Failed to delete corrupted model file: %s", self.model_path)

    def _save(self) -> None:
        """Persist model to disk."""
        model_dir = os.path.dirname(self.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self._model, f)
