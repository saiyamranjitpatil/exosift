"""
ExoSift — Variable-Length CNN Validation (V1, Priority A)

Validates cnn_1d_variable.keras on labelled validation data at multiple
sequence lengths, preserving the held-out Kaggle test set entirely.

Design
------
- Reuses the exact same stratified 80/20 train/validation split as
  src/train_cnn.py (VALIDATION_SIZE=0.20, RANDOM_STATE=42, stratify=y).
- Only the validation split is scored; test_sequences.npy / test_labels.npy
  are never read.
- Evaluates the saved variable-length model at lengths 1925 (real-star
  Q02 segment) and 3197 (original training length), plus 1024 (lower bound).
- Reports PR-AUC, ROC-AUC, precision, recall and F1 at the
  validation-selected threshold (0.0939) and at the default 0.50.
- Verifies weight-transfer compatibility between cnn_1d.keras and
  cnn_1d_variable.keras.

No retraining occurs. The test set remains untouched.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

TRAIN_SEQUENCE_PATH = DATA_DIR / "train_sequences.npy"
TRAIN_LABEL_PATH = DATA_DIR / "train_labels.npy"

FIXED_MODEL_PATH = MODEL_DIR / "cnn_1d.keras"
VARIABLE_MODEL_PATH = MODEL_DIR / "cnn_1d_variable.keras"

VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

VALIDATION_THRESHOLD = 0.0939
DEFAULT_THRESHOLD = 0.50

EVAL_LENGTHS = [1024, 1925, 3197]

OUTPUT_CSV = MODEL_DIR / "variable_length_validation.csv"


# ============================================================
# HELPERS
# ============================================================

def load_validation_split():
    """Return (X_val, y_val) using the same split as train_cnn.py."""
    X = np.load(TRAIN_SEQUENCE_PATH)
    y = np.load(TRAIN_LABEL_PATH)

    if X.ndim != 2:
        raise ValueError(f"Expected X shape (samples, timesteps), got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected y shape (samples,), got {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"X/y mismatch: {len(X)} vs {len(y)}")

    _, X_val, _, y_val = train_test_split(
        X, y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_val, y_val


def slice_to_length(sequences, length):
    """Take a contiguous leading slice of ``length`` timesteps.

    No interpolation or padding is applied.
    """
    if sequences.shape[1] < length:
        raise ValueError(
            f"Sequences have {sequences.shape[1]} timesteps; "
            f"cannot slice to {length}."
        )
    return sequences[:, :length]


def metrics_at_threshold(y_true, y_prob, threshold):
    """Return dict of threshold-based metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


# ============================================================
# WEIGHT-TRANSFER COMPATIBILITY CHECK
# ============================================================

def verify_weight_transfer():
    """Compare weight tensors between fixed and variable models."""
    fixed = tf.keras.models.load_model(FIXED_MODEL_PATH)
    variable = tf.keras.models.load_model(VARIABLE_MODEL_PATH)

    fixed_weights = fixed.get_weights()
    variable_weights = variable.get_weights()

    if len(fixed_weights) != len(variable_weights):
        raise RuntimeError(
            f"Weight tensor count mismatch: fixed={len(fixed_weights)}, "
            f"variable={len(variable_weights)}"
        )

    mismatches = []
    for i, (fw, vw) in enumerate(zip(fixed_weights, variable_weights)):
        if fw.shape != vw.shape:
            mismatches.append((i, fw.shape, vw.shape))
        elif not np.allclose(fw, vw, rtol=1e-5, atol=1e-6):
            mismatches.append((i, "shape match but values differ", fw.shape))

    report = {
        "fixed_input_shape": fixed.input_shape,
        "variable_input_shape": variable.input_shape,
        "num_weight_tensors": len(fixed_weights),
        "shapes_match": all(
            fw.shape == vw.shape for fw, vw in zip(fixed_weights, variable_weights)
        ),
        "values_match": all(
            np.allclose(fw, vw, rtol=1e-5, atol=1e-6)
            for fw, vw in zip(fixed_weights, variable_weights)
        ),
        "mismatches": mismatches,
    }
    return report


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 72)
    print("EXOSIFT — VARIABLE-LENGTH CNN VALIDATION")
    print("=" * 72)

    print("\n[1/3] Verifying weight transfer "
          "(cnn_1d.keras -> cnn_1d_variable.keras)...")
    wt_report = verify_weight_transfer()
    print(f"  Fixed model input shape : {wt_report['fixed_input_shape']}")
    print(f"  Variable input shape    : {wt_report['variable_input_shape']}")
    print(f"  Weight tensors          : {wt_report['num_weight_tensors']}")
    print(f"  Shapes match            : {wt_report['shapes_match']}")
    print(f"  Values match            : {wt_report['values_match']}")
    if wt_report["mismatches"]:
        print(f"  MISMATCHES: {wt_report['mismatches']}")
    if not wt_report["shapes_match"] or not wt_report["values_match"]:
        raise RuntimeError("Weight transfer compatibility check FAILED.")
    print("  ✓ Weight transfer is exact.")

    print("\n[2/3] Loading validation split (test set NOT touched)...")
    X_val, y_val = load_validation_split()
    n_val = len(y_val)
    n_pos = int(np.sum(y_val == 1))
    n_neg = int(np.sum(y_val == 0))
    print(f"  Validation samples : {n_val}")
    print(f"  Positives (planet) : {n_pos}")
    print(f"  Negatives          : {n_neg}")
    print("  Test set           : untouched (not read)")

    print("\n[3/3] Evaluating variable-length CNN at each length...")
    variable_model = tf.keras.models.load_model(VARIABLE_MODEL_PATH)

    rows = []
    for length in EVAL_LENGTHS:
        X_slice = slice_to_length(X_val, length)
        X_cnn = X_slice[..., np.newaxis]

        y_prob = variable_model.predict(
            X_cnn, batch_size=64, verbose=0
        ).ravel()

        pr_auc = float(average_precision_score(y_val, y_prob))
        roc_auc = float(roc_auc_score(y_val, y_prob))

        tuned = metrics_at_threshold(y_val, y_prob, VALIDATION_THRESHOLD)
        default = metrics_at_threshold(y_val, y_prob, DEFAULT_THRESHOLD)

        row = {
            "length": length,
            "n_samples": n_val,
            "n_positives": n_pos,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            **{f"tuned_{k}": v for k, v in tuned.items()},
            **{f"default_{k}": v for k, v in default.items()},
        }
        rows.append(row)

        print(
            f"  Length {length:5d}: "
            f"PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}  "
            f"tuned@{VALIDATION_THRESHOLD} "
            f"P={tuned['precision']:.3f} "
            f"R={tuned['recall']:.3f} "
            f"F1={tuned['f1']:.3f}  "
            f"(pos={n_pos})"
        )

    import pandas as pd
    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 72)
    print("VALIDATION COMPLETE")
    print("=" * 72)
    print(f"Results saved: {OUTPUT_CSV}")
    print("\nSummary:")
    print(results_df.to_string(index=False))
    print("\nNote: The held-out Kaggle test set was NOT used in this validation.")
    print("      All metrics are computed on the stratified validation split only.")
    print()


if __name__ == "__main__":
    main()
