from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

BATCH_SIZE = 64

RF_PARAMS = {
    "n_estimators": 500,
    "class_weight": "balanced",
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_FEATURES_PATH = DATA_DIR / "train_features.csv"
TEST_FEATURES_PATH = DATA_DIR / "test_features.csv"

TRAIN_LABELS_PATH = DATA_DIR / "train_labels.npy"
TEST_LABELS_PATH = DATA_DIR / "test_labels.npy"

TRAIN_SEQUENCES_PATH = DATA_DIR / "train_sequences.npy"
TEST_SEQUENCES_PATH = DATA_DIR / "test_sequences.npy"

RF_MODEL_PATH = MODEL_DIR / "random_forest.joblib"
CNN_MODEL_PATH = MODEL_DIR / "cnn_1d.keras"

COMPARISON_CSV_PATH = MODEL_DIR / "model_comparison.csv"
THRESHOLD_CSV_PATH = MODEL_DIR / "model_threshold_analysis.csv"

ROC_PLOT_PATH = MODEL_DIR / "model_comparison_roc.png"
PR_PLOT_PATH = MODEL_DIR / "model_comparison_precision_recall.png"

COMPARISON_TEXT_PATH = MODEL_DIR / "model_comparison_report.txt"


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# HELPERS
# ============================================================

def validate_binary_labels(y, name):
    """Validate binary 0/1 labels."""

    if y.ndim != 1:
        raise ValueError(
            f"{name} must be 1-dimensional. Got {y.shape}."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    unique = np.unique(y)

    if not np.array_equal(unique, np.array([0, 1])):
        raise ValueError(
            f"{name} must contain [0, 1]. Found {unique}."
        )


def load_random_forest_artifact(path):
    """
    Load the RF artifact safely.

    Supports:
      1. A raw RandomForestClassifier
      2. A dictionary containing the estimator
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Random Forest model not found:\n{path}"
        )

    artifact = load(path)

    print(
        f"\nRF artifact type: {type(artifact).__name__}"
    )

    # Case 1: raw sklearn estimator
    if hasattr(artifact, "predict_proba"):
        print("RF artifact contains a direct estimator.")
        return artifact

    # Case 2: wrapped dictionary
    if isinstance(artifact, dict):

        print(
            "RF artifact is a dictionary."
        )

        # Most likely structure from our training pipeline.
        preferred_keys = [
            "model",
            "classifier",
            "estimator",
            "random_forest",
            "rf",
        ]

        for key in preferred_keys:
            candidate = artifact.get(key)

            if candidate is not None and hasattr(
                candidate,
                "predict_proba",
            ):
                print(
                    f"Extracted RF estimator from key: '{key}'"
                )

                return candidate

        # Generic fallback:
        # search every dictionary value.
        for key, value in artifact.items():

            if hasattr(value, "predict_proba"):
                print(
                    f"Extracted RF estimator from key: '{key}'"
                )

                return value

        raise TypeError(
            "The random_forest.joblib dictionary does not "
            "contain an object with predict_proba(). "
            f"Available keys: {list(artifact.keys())}"
        )

    raise TypeError(
        "Unsupported Random Forest artifact type: "
        f"{type(artifact)}"
    )


def validate_feature_data(
    train_features,
    test_features,
):
    """Validate RF feature matrices."""

    if train_features.empty:
        raise ValueError(
            "Training feature file is empty."
        )

    if test_features.empty:
        raise ValueError(
            "Test feature file is empty."
        )

    if list(train_features.columns) != list(
        test_features.columns
    ):
        raise ValueError(
            "Train/test feature columns do not match."
        )

    if train_features.isnull().any().any():
        raise ValueError(
            "Training features contain missing values."
        )

    if test_features.isnull().any().any():
        raise ValueError(
            "Test features contain missing values."
        )

    train_values = train_features.to_numpy()
    test_values = test_features.to_numpy()

    if not np.isfinite(train_values).all():
        raise ValueError(
            "Training features contain NaN/Inf."
        )

    if not np.isfinite(test_values).all():
        raise ValueError(
            "Test features contain NaN/Inf."
        )

    suspicious = {
        "label",
        "labels",
        "target",
        "y",
        "class",
    }

    lower_columns = {
        column.lower()
        for column in train_features.columns
    }

    leakage_columns = suspicious.intersection(
        lower_columns
    )

    if leakage_columns:
        raise ValueError(
            "Target leakage detected in RF features: "
            f"{sorted(leakage_columns)}"
        )


def find_best_f1_threshold(
    y_true,
    probabilities,
):
    """
    Select threshold using validation data only.
    """

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    if len(thresholds) == 0:
        return 0.5, 0.0

    f1_scores = (
        2.0
        * precision[:-1]
        * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    best_idx = int(
        np.argmax(f1_scores)
    )

    return (
        float(thresholds[best_idx]),
        float(f1_scores[best_idx]),
    )


def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):
    """Calculate threshold and ranking metrics."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "specificity": specificity,
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def print_metrics(
    model_name,
    dataset_name,
    metrics,
):
    """Print evaluation metrics."""

    print("\n" + "=" * 72)
    print(
        f"{model_name} — {dataset_name}"
    )
    print("=" * 72)

    print(
        f"Threshold          : "
        f"{metrics['threshold']:.4f}"
    )

    print(
        f"Accuracy           : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy  : "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Precision          : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall             : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1                 : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Specificity        : "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"ROC-AUC            : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC             : "
        f"{metrics['pr_auc']:.4f}"
    )

    print("\nConfusion matrix:")

    print(
        np.array(
            [
                [
                    metrics["true_negatives"],
                    metrics["false_positives"],
                ],
                [
                    metrics["false_negatives"],
                    metrics["true_positives"],
                ],
            ]
        )
    )


# ============================================================
# START
# ============================================================

print("=" * 72)
print("ExoSift — RF vs 1D CNN Model Comparison")
print("=" * 72)


# ============================================================
# LOAD LABELS
# ============================================================

print("\nLoading labels...")

y_train = np.load(
    TRAIN_LABELS_PATH
)

y_test = np.load(
    TEST_LABELS_PATH
)

validate_binary_labels(
    y_train,
    "y_train",
)

validate_binary_labels(
    y_test,
    "y_test",
)

print(
    f"Training labels: {y_train.shape}"
)

print(
    f"Test labels    : {y_test.shape}"
)

print("\nTraining distribution:")

for label, count in zip(
    *np.unique(
        y_train,
        return_counts=True,
    )
):
    name = (
        "Planet"
        if label == 1
        else "Non-planet"
    )

    print(
        f"  {name:12s}: {count}"
    )

print("\nTest distribution:")

for label, count in zip(
    *np.unique(
        y_test,
        return_counts=True,
    )
):
    name = (
        "Planet"
        if label == 1
        else "Non-planet"
    )

    print(
        f"  {name:12s}: {count}"
    )


# ============================================================
# RECREATE EXACT CNN/RF VALIDATION SPLIT
# ============================================================

print(
    "\nRecreating stratified validation split..."
)

all_train_indices = np.arange(
    len(y_train)
)

train_idx, val_idx = train_test_split(
    all_train_indices,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train,
)

y_train_split = y_train[train_idx]
y_val = y_train[val_idx]

print(
    f"Training split indices  : "
    f"{len(train_idx)}"
)

print(
    f"Validation split indices: "
    f"{len(val_idx)}"
)

print(
    f"Validation planets      : "
    f"{np.sum(y_val == 1)}"
)

print(
    f"Validation non-planets  : "
    f"{np.sum(y_val == 0)}"
)


# ============================================================
# LOAD RF FEATURES
# ============================================================

print(
    "\nLoading RF feature data..."
)

train_features = pd.read_csv(
    TRAIN_FEATURES_PATH
)

test_features = pd.read_csv(
    TEST_FEATURES_PATH
)

validate_feature_data(
    train_features,
    test_features,
)

X_rf_train = train_features.to_numpy(
    dtype=np.float32
)

X_rf_test = test_features.to_numpy(
    dtype=np.float32
)

print(
    f"RF training shape: "
    f"{X_rf_train.shape}"
)

print(
    f"RF test shape    : "
    f"{X_rf_test.shape}"
)


# ============================================================
# RF VALIDATION MODEL
# ============================================================

print(
    "\nTraining temporary RF for "
    "validation threshold selection..."
)

rf_validation = RandomForestClassifier(
    **RF_PARAMS
)

rf_validation.fit(
    X_rf_train[train_idx],
    y_train_split,
)

rf_val_probabilities = (
    rf_validation
    .predict_proba(
        X_rf_train[val_idx]
    )[:, 1]
)


# ============================================================
# RF VALIDATION THRESHOLD
# ============================================================

rf_best_threshold, rf_best_f1 = (
    find_best_f1_threshold(
        y_val,
        rf_val_probabilities,
    )
)

print(
    f"\nRF best validation threshold: "
    f"{rf_best_threshold:.4f}"
)

print(
    f"RF best validation F1: "
    f"{rf_best_f1:.4f}"
)


# ============================================================
# LOAD FINAL RF
# ============================================================

print(
    "\nLoading final RF model..."
)

rf_final = load_random_forest_artifact(
    RF_MODEL_PATH
)

# Safety check before proceeding.

if not hasattr(
    rf_final,
    "predict_proba",
):
    raise TypeError(
        "Loaded RF model does not expose predict_proba()."
    )

if not hasattr(
    rf_final,
    "classes_",
):
    raise TypeError(
        "Loaded RF model appears to be unfitted."
    )

rf_test_probabilities = (
    rf_final
    .predict_proba(
        X_rf_test
    )[:, 1]
)

print(
    "✓ Final RF successfully loaded."
)

print(
    "✓ RF test probabilities generated."
)


# ============================================================
# LOAD CNN
# ============================================================

print("\nLoading CNN...")

if not CNN_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"CNN model not found:\n"
        f"{CNN_MODEL_PATH}"
    )

cnn = tf.keras.models.load_model(
    CNN_MODEL_PATH
)

print("✓ CNN successfully loaded.")


# ============================================================
# LOAD CNN SEQUENCES
# ============================================================

print(
    "\nLoading CNN sequences..."
)

X_sequences = np.load(
    TRAIN_SEQUENCES_PATH
)

X_test_sequences = np.load(
    TEST_SEQUENCES_PATH
)

if not np.isfinite(
    X_sequences
).all():
    raise ValueError(
        "Training CNN sequences contain NaN/Inf."
    )

if not np.isfinite(
    X_test_sequences
).all():
    raise ValueError(
        "Test CNN sequences contain NaN/Inf."
    )

if len(X_sequences) != len(y_train):
    raise ValueError(
        "Training sequence count does not "
        "match training labels."
    )

if len(X_test_sequences) != len(y_test):
    raise ValueError(
        "Test sequence count does not "
        "match test labels."
    )

X_sequences = X_sequences[
    ...,
    np.newaxis,
]

X_test_sequences = X_test_sequences[
    ...,
    np.newaxis,
]

print(
    f"CNN validation input: "
    f"{X_sequences[val_idx].shape}"
)

print(
    f"CNN test input      : "
    f"{X_test_sequences.shape}"
)


# ============================================================
# CNN VALIDATION PREDICTIONS
# ============================================================

print(
    "\nGenerating CNN validation predictions..."
)

cnn_val_probabilities = (
    cnn.predict(
        X_sequences[val_idx],
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    .ravel()
)


# ============================================================
# CNN TEST PREDICTIONS
# ============================================================

print(
    "Generating CNN test predictions..."
)

cnn_test_probabilities = (
    cnn.predict(
        X_test_sequences,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    .ravel()
)


# ============================================================
# CNN VALIDATION THRESHOLD
# ============================================================

cnn_best_threshold, cnn_best_f1 = (
    find_best_f1_threshold(
        y_val,
        cnn_val_probabilities,
    )
)

print(
    f"\nCNN best validation threshold: "
    f"{cnn_best_threshold:.4f}"
)

print(
    f"CNN best validation F1: "
    f"{cnn_best_f1:.4f}"
)


# ============================================================
# VALIDATION METRICS
# ============================================================

rf_val_default = calculate_metrics(
    y_val,
    rf_val_probabilities,
    0.5,
)

rf_val_tuned = calculate_metrics(
    y_val,
    rf_val_probabilities,
    rf_best_threshold,
)

cnn_val_default = calculate_metrics(
    y_val,
    cnn_val_probabilities,
    0.5,
)

cnn_val_tuned = calculate_metrics(
    y_val,
    cnn_val_probabilities,
    cnn_best_threshold,
)


# ============================================================
# TEST METRICS
# ============================================================

rf_test_default = calculate_metrics(
    y_test,
    rf_test_probabilities,
    0.5,
)

rf_test_tuned = calculate_metrics(
    y_test,
    rf_test_probabilities,
    rf_best_threshold,
)

cnn_test_default = calculate_metrics(
    y_test,
    cnn_test_probabilities,
    0.5,
)

cnn_test_tuned = calculate_metrics(
    y_test,
    cnn_test_probabilities,
    cnn_best_threshold,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print_metrics(
    "Random Forest",
    "Validation — Threshold 0.50",
    rf_val_default,
)

print_metrics(
    "Random Forest",
    "Validation — Tuned Threshold",
    rf_val_tuned,
)

print_metrics(
    "1D CNN",
    "Validation — Threshold 0.50",
    cnn_val_default,
)

print_metrics(
    "1D CNN",
    "Validation — Tuned Threshold",
    cnn_val_tuned,
)

print_metrics(
    "Random Forest",
    "Held-Out Test — Threshold 0.50",
    rf_test_default,
)

print_metrics(
    "Random Forest",
    "Held-Out Test — Validation-Selected Threshold",
    rf_test_tuned,
)

print_metrics(
    "1D CNN",
    "Held-Out Test — Threshold 0.50",
    cnn_test_default,
)

print_metrics(
    "1D CNN",
    "Held-Out Test — Validation-Selected Threshold",
    cnn_test_tuned,
)


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

comparison_rows = [
    {
        "model": "Random Forest",
        "dataset": "Validation",
        "threshold_type": "Default",
        **rf_val_default,
    },
    {
        "model": "Random Forest",
        "dataset": "Validation",
        "threshold_type": "Tuned",
        **rf_val_tuned,
    },
    {
        "model": "1D CNN",
        "dataset": "Validation",
        "threshold_type": "Default",
        **cnn_val_default,
    },
    {
        "model": "1D CNN",
        "dataset": "Validation",
        "threshold_type": "Tuned",
        **cnn_val_tuned,
    },
    {
        "model": "Random Forest",
        "dataset": "Held-Out Test",
        "threshold_type": "Default",
        **rf_test_default,
    },
    {
        "model": "Random Forest",
        "dataset": "Held-Out Test",
        "threshold_type": "Validation-Selected",
        **rf_test_tuned,
    },
    {
        "model": "1D CNN",
        "dataset": "Held-Out Test",
        "threshold_type": "Default",
        **cnn_test_default,
    },
    {
        "model": "1D CNN",
        "dataset": "Held-Out Test",
        "threshold_type": "Validation-Selected",
        **cnn_test_tuned,
    },
]

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    COMPARISON_CSV_PATH,
    index=False,
)


# ============================================================
# THRESHOLD TABLE
# ============================================================

threshold_df = pd.DataFrame(
    [
        {
            "model": "Random Forest",
            "validation_best_threshold":
                rf_best_threshold,
            "validation_best_f1":
                rf_best_f1,
        },
        {
            "model": "1D CNN",
            "validation_best_threshold":
                cnn_best_threshold,
            "validation_best_f1":
                cnn_best_f1,
        },
    ]
)

threshold_df.to_csv(
    THRESHOLD_CSV_PATH,
    index=False,
)


# ============================================================
# ROC CURVE
# ============================================================

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_test_probabilities,
)

cnn_fpr, cnn_tpr, _ = roc_curve(
    y_test,
    cnn_test_probabilities,
)

plt.figure(figsize=(8, 6))

plt.plot(
    rf_fpr,
    rf_tpr,
    label=(
        f"Random Forest "
        f"(AUC={rf_test_default['roc_auc']:.4f})"
    ),
)

plt.plot(
    cnn_fpr,
    cnn_tpr,
    label=(
        f"1D CNN "
        f"(AUC={cnn_test_default['roc_auc']:.4f})"
    ),
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random baseline",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ExoSift — RF vs 1D CNN ROC Curve"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    ROC_PLOT_PATH,
    dpi=220,
)

plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

rf_precision, rf_recall, _ = (
    precision_recall_curve(
        y_test,
        rf_test_probabilities,
    )
)

cnn_precision, cnn_recall, _ = (
    precision_recall_curve(
        y_test,
        cnn_test_probabilities,
    )
)

test_prevalence = np.mean(
    y_test
)

plt.figure(figsize=(8, 6))

plt.plot(
    rf_recall,
    rf_precision,
    label=(
        f"Random Forest "
        f"(PR-AUC={rf_test_default['pr_auc']:.4f})"
    ),
)

plt.plot(
    cnn_recall,
    cnn_precision,
    label=(
        f"1D CNN "
        f"(PR-AUC={cnn_test_default['pr_auc']:.4f})"
    ),
)

plt.axhline(
    test_prevalence,
    linestyle="--",
    label=(
        f"Positive prevalence "
        f"({test_prevalence:.4f})"
    ),
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "ExoSift — RF vs 1D CNN Precision-Recall"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PR_PLOT_PATH,
    dpi=220,
)

plt.close()


# ============================================================
# TEXT REPORT
# ============================================================

report = f"""
ExoSift — Model Comparison Report
=================================

Dataset
-------
Training samples : {len(y_train)}
Test samples     : {len(y_test)}
Training planets : {np.sum(y_train == 1)}
Test planets     : {np.sum(y_test == 1)}

Evaluation design
-----------------
The Random Forest and 1D CNN are evaluated on the same
held-out Kaggle test set.

Operating thresholds are selected only from validation data.

The test set contains only five positive planet examples,
so test metrics must be interpreted cautiously.

Held-Out Test — Default Threshold 0.50
--------------------------------------

Random Forest:
  ROC-AUC  : {rf_test_default["roc_auc"]:.4f}
  PR-AUC   : {rf_test_default["pr_auc"]:.4f}
  Precision: {rf_test_default["precision"]:.4f}
  Recall   : {rf_test_default["recall"]:.4f}
  F1       : {rf_test_default["f1"]:.4f}

1D CNN:
  ROC-AUC  : {cnn_test_default["roc_auc"]:.4f}
  PR-AUC   : {cnn_test_default["pr_auc"]:.4f}
  Precision: {cnn_test_default["precision"]:.4f}
  Recall   : {cnn_test_default["recall"]:.4f}
  F1       : {cnn_test_default["f1"]:.4f}

Validation-Selected Threshold
-----------------------------

Random Forest:
  Threshold: {rf_best_threshold:.4f}
  Test F1  : {rf_test_tuned["f1"]:.4f}
  Test Recall: {rf_test_tuned["recall"]:.4f}

1D CNN:
  Threshold: {cnn_best_threshold:.4f}
  Test F1  : {cnn_test_tuned["f1"]:.4f}
  Test Recall: {cnn_test_tuned["recall"]:.4f}

Interpretation
--------------

PR-AUC, recall and F1 are emphasized because the dataset
is extremely imbalanced.

Accuracy is not used as the primary headline metric.

The comparison is evidence about performance on this
specific Kaggle dataset and split, not proof of real-world
exoplanet discovery capability.
""".strip()

COMPARISON_TEXT_PATH.write_text(
    report,
    encoding="utf-8",
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("MODEL COMPARISON COMPLETE")
print("=" * 72)

print("\nSaved artifacts:")

print(
    f"  ✓ {COMPARISON_CSV_PATH}"
)

print(
    f"  ✓ {THRESHOLD_CSV_PATH}"
)

print(
    f"  ✓ {ROC_PLOT_PATH}"
)

print(
    f"  ✓ {PR_PLOT_PATH}"
)

print(
    f"  ✓ {COMPARISON_TEXT_PATH}"
)

print("\nHeld-out test headline metrics:")

print(
    f"  Random Forest:"
    f" ROC-AUC={rf_test_default['roc_auc']:.4f},"
    f" PR-AUC={rf_test_default['pr_auc']:.4f}"
)

print(
    f"  1D CNN:"
    f" ROC-AUC={cnn_test_default['roc_auc']:.4f},"
    f" PR-AUC={cnn_test_default['pr_auc']:.4f}"
)

print(
    "\nExoSift model comparison finished successfully."
)