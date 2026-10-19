"""
ExoSift — Confidence Calibration (V2, Priority B5)

Raw classifier outputs (CNN sigmoid, RF probability) are not calibrated
probabilities. This module fits Platt scaling (logistic regression) and
isotonic regression calibrators on the V1 validation set, then maps raw
scores to calibrated probabilities.

Uses only validation data — never the held-out test set — to avoid
optimistically biased calibration.

Pipeline
--------
1. Load validation predictions (cnn_validation_predictions.csv) and
   RF validation predictions.
2. Fit Platt scaling (LogisticRegression) and/or isotonic regression.
3. Save calibrators to models/.
4. Provide calibrate() to map raw score → calibrated probability.

No retraining of the base models; calibration is a post-processing step.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

CNN_VAL_PREDICTIONS = MODEL_DIR / "cnn_validation_predictions.csv"

CNN_PLATT_PATH = MODEL_DIR / "cnn_platt_calibrator.joblib"
CNN_ISOTONIC_PATH = MODEL_DIR / "cnn_isotonic_calibrator.joblib"


# ============================================================
# CALIBRATOR FITTING
# ============================================================

def fit_platt_scaling(raw_scores, y_true):
    """
    Fit Platt scaling: logistic regression on raw scores.

    Maps raw_score -> calibrated probability via P(y=1 | score) = sigmoid(a * score + b).
    """
    raw_scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1, 1)
    y_true = np.asarray(y_true, dtype=np.int64)

    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(raw_scores, y_true)
    return calibrator


def fit_isotonic(raw_scores, y_true):
    """
    Fit isotonic regression: non-parametric monotonic calibration.
    """
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_scores, y_true)
    return calibrator


def evaluate_calibration(calibrator, raw_scores, y_true, method="platt"):
    """
    Compute Brier score for a calibrator on given data.
    Lower Brier = better calibrated.
    """
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    if method == "platt":
        calibrated = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
    else:
        calibrated = calibrator.predict(raw_scores)

    brier = brier_score_loss(y_true, calibrated)
    return float(brier)


def fit_and_save_calibrators():
    """
    Fit Platt and isotonic calibrators on CNN validation predictions,
    save them, and report Brier scores.

    Returns dict with results.
    """
    if not CNN_VAL_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"Validation predictions not found: {CNN_VAL_PREDICTIONS}. "
            "Run src/train_cnn.py first to generate them."
        )

    df = pd.read_csv(CNN_VAL_PREDICTIONS)
    required = {"actual_label", "predicted_probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {CNN_VAL_PREDICTIONS}: {missing}"
        )

    raw_scores = df["predicted_probability"].values
    y_true = df["actual_label"].values

    # Fit calibrators.
    platt = fit_platt_scaling(raw_scores, y_true)
    isotonic = fit_isotonic(raw_scores, y_true)

    # Evaluate on validation data (note: this is an optimistic estimate;
    # true calibration should be checked on held-out data).
    brier_platt = evaluate_calibration(platt, raw_scores, y_true, method="platt")
    brier_isotonic = evaluate_calibration(isotonic, raw_scores, y_true, method="isotonic")
    brier_raw = brier_score_loss(y_true, raw_scores)

    # Save calibrators.
    joblib.dump(platt, CNN_PLATT_PATH)
    joblib.dump(isotonic, CNN_ISOTONIC_PATH)

    results = {
        "n_validation_samples": len(df),
        "n_positives": int(np.sum(y_true == 1)),
        "brier_raw": brier_raw,
        "brier_platt": brier_platt,
        "brier_isotonic": brier_isotonic,
        "platt_saved": str(CNN_PLATT_PATH),
        "isotonic_saved": str(CNN_ISOTONIC_PATH),
    }
    return results


# ============================================================
# CALIBRATED PREDICTION
# ============================================================

class CNNCalibrator:
    """Loads a saved calibrator and maps raw CNN scores to calibrated probabilities."""

    def __init__(self, method="platt"):
        if method == "platt":
            path = CNN_PLATT_PATH
        elif method == "isotonic":
            path = CNN_ISOTONIC_PATH
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        if not path.exists():
            raise FileNotFoundError(
                f"Calibrator not found: {path}. "
                "Run fit_and_save_calibrators() first."
            )

        self.method = method
        self.calibrator = joblib.load(path)

    def calibrate(self, raw_score):
        """
        Map a raw CNN score to a calibrated probability.

        Parameters
        ----------
        raw_score : float or array-like
            Raw CNN output (sigmoid probability, 0-1).

        Returns
        -------
        float or array-like : calibrated probability (0-1).
        """
        raw_score = np.asarray(raw_score, dtype=np.float64)

        if self.method == "platt":
            calibrated = self.calibrator.predict_proba(
                raw_score.reshape(-1, 1)
            )[:, 1]
        else:
            calibrated = self.calibrator.predict(raw_score)

        calibrated = np.clip(calibrated, 0.0, 1.0)

        # Return a Python float for scalar input, array for array input.
        if raw_score.ndim == 0:
            return float(calibrated[0])
        return calibrated


# ============================================================
# MAIN — SELF-TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 72)
    print("EXOSIFT — CONFIDENCE CALIBRATION SELF-TEST")
    print("=" * 72)

    try:
        results = fit_and_save_calibrators()
        print(f"\nCalibration results:")
        print(f"  Validation samples : {results['n_validation_samples']}")
        print(f"  Positives          : {results['n_positives']}")
        print(f"  Brier score (raw)  : {results['brier_raw']:.4f}")
        print(f"  Brier score (Platt): {results['brier_platt']:.4f}")
        print(f"  Brier score (Iso):  {results['brier_isotonic']:.4f}")
        print(f"  Platt saved: {results['platt_saved']}")
        print(f"  Isotonic saved: {results['isotonic_saved']}")

        # Test calibration on a few scores.
        cal = CNNCalibrator(method="platt")
        for score in [0.1, 0.5, 0.9]:
            cal_prob = float(cal.calibrate(score))
            print(f"  Raw {score:.1f} -> calibrated {cal_prob:.4f}")

    except FileNotFoundError as exc:
        print(f"\nSkipped: {exc}")
        print("Run src/train_cnn.py to generate validation predictions first.")

    print("\nSelf-test complete.")
