"""
ExoSift — Random Forest Baseline

Pipeline:
    Engineered features
        ↓
    Stratified validation split
        ↓
    Random Forest
        ↓
    Validation evaluation
        ↓
    Held-out Kaggle test evaluation
        ↓
    Save model + feature importance + plots
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Use a non-interactive backend so plots never block training.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ================================================================
# PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = MODEL_DIR / "plots"

TRAIN_FEATURES_PATH = PROCESSED_DIR / "train_features.csv"
TEST_FEATURES_PATH = PROCESSED_DIR / "test_features.csv"

TRAIN_LABELS_PATH = PROCESSED_DIR / "train_labels.npy"
TEST_LABELS_PATH = PROCESSED_DIR / "test_labels.npy"

MODEL_PATH = MODEL_DIR / "random_forest.joblib"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "random_forest_feature_importance.csv"

VALIDATION_CM_PATH = PLOT_DIR / "random_forest_validation_confusion_matrix.png"
VALIDATION_PR_PATH = PLOT_DIR / "random_forest_validation_precision_recall.png"

TEST_CM_PATH = PLOT_DIR / "random_forest_test_confusion_matrix.png"
TEST_PR_PATH = PLOT_DIR / "random_forest_test_precision_recall.png"


# ================================================================
# RANDOM FOREST CONFIGURATION
# ================================================================

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

N_ESTIMATORS = 500
MIN_SAMPLES_LEAF = 2
MAX_FEATURES = "sqrt"


# ================================================================
# DIRECTORY SETUP
# ================================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def load_features_and_labels(feature_path, label_path, dataset_name):
    """Load engineered features and labels."""

    features = pd.read_csv(feature_path)
    labels = np.load(label_path)

    if len(features) != len(labels):
        raise ValueError(
            f"{dataset_name}: feature rows ({len(features)}) "
            f"do not match label count ({len(labels)})."
        )

    if features.isnull().any().any():
        raise ValueError(f"{dataset_name}: feature matrix contains NaN values.")

    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError(
            f"{dataset_name}: feature matrix contains infinite values."
        )

    return features, labels


def print_dataset_summary(train_features, train_labels, test_features, test_labels):
    """Print dataset dimensions and class distribution."""

    print()
    print("Training dataset:", train_features.shape)
    print("Test dataset    :", test_features.shape)

    print()
    print("Training planets    :", int(np.sum(train_labels == 1)))
    print("Training non-planets:", int(np.sum(train_labels == 0)))

    print()
    print("Test planets        :", int(np.sum(test_labels == 1)))
    print("Test non-planets    :", int(np.sum(test_labels == 0)))


def evaluate_model(model, X, y, dataset_name):
    """
    Evaluate model using both thresholded predictions and probability scores.
    """

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)

    roc_auc = roc_auc_score(y, probabilities)
    pr_auc = average_precision_score(y, probabilities)

    cm = confusion_matrix(y, predictions)

    print()
    print("=" * 65)
    print(f"📊 {dataset_name.upper()} RESULTS")
    print("=" * 65)

    print()
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    print()
    print("Classification Report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=["Non-Planet", "Planet"],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "probabilities": probabilities,
        "predictions": predictions,
        "confusion_matrix": cm,
    }


def save_confusion_matrix(cm, output_path, title):
    """Save confusion matrix plot without opening a window."""

    fig, ax = plt.subplots(figsize=(8, 7))

    image = ax.imshow(cm, interpolation="nearest")

    ax.set_title(title, fontsize=18, pad=15)
    ax.set_xlabel("Predicted label", fontsize=14)
    ax.set_ylabel("True label", fontsize=14)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Non-Planet", "Planet"], fontsize=12)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Non-Planet", "Planet"], fontsize=12)

    # Write values inside each cell.
    threshold = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=18,
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.colorbar(image, ax=ax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_curve(y, probabilities, output_path, title):
    """Save precision-recall curve without opening a window."""

    precision, recall, _ = precision_recall_curve(y, probabilities)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recall, precision, linewidth=2)

    positive_rate = np.mean(y)

    ax.axhline(
        y=positive_rate,
        linestyle="--",
        linewidth=1.5,
        label=f"Positive-class prevalence = {positive_rate:.4f}",
    )

    ax.set_title(title, fontsize=18, pad=15)
    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(model, feature_names):
    """Save Random Forest feature importances."""

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    print()
    print("=" * 65)
    print("🔍 TOP RANDOM FOREST FEATURES")
    print("=" * 65)

    print()
    print(importance_df.head(15).to_string(index=False))

    return importance_df


def save_model_bundle(model, feature_names):
    """
    Save the trained model together with metadata needed for later inference.
    """

    model_bundle = {
        "model": model,
        "feature_names": list(feature_names),
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "max_features": MAX_FEATURES,
    }

    joblib.dump(model_bundle, MODEL_PATH)


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print("=" * 65)
    print("🔭 EXOSIFT — RANDOM FOREST BASELINE")
    print("=" * 65)

    # ------------------------------------------------------------
    # LOAD FEATURES
    # ------------------------------------------------------------

    print()
    print("Loading engineered features...")

    train_features, train_labels = load_features_and_labels(
        TRAIN_FEATURES_PATH,
        TRAIN_LABELS_PATH,
        "TRAIN",
    )

    test_features, test_labels = load_features_and_labels(
        TEST_FEATURES_PATH,
        TEST_LABELS_PATH,
        "TEST",
    )

    print_dataset_summary(
        train_features,
        train_labels,
        test_features,
        test_labels,
    )

    # ------------------------------------------------------------
    # FEATURE NAMES
    # ------------------------------------------------------------

    feature_names = train_features.columns.tolist()

    if train_features.columns.tolist() != test_features.columns.tolist():
        raise ValueError(
            "Train and test feature columns do not match."
        )

    # Convert to numpy arrays.
    X = train_features.to_numpy(dtype=np.float32)
    y = np.asarray(train_labels, dtype=np.int64)

    X_test = test_features.to_numpy(dtype=np.float32)
    y_test = np.asarray(test_labels, dtype=np.int64)

    # ------------------------------------------------------------
    # STRATIFIED VALIDATION SPLIT
    # ------------------------------------------------------------

    print()
    print("=" * 65)
    print("Creating stratified validation split...")
    print("=" * 65)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print()
    print("Training samples   :", len(X_train))
    print("Validation samples :", len(X_val))

    print("Training planets   :", int(np.sum(y_train == 1)))
    print("Validation planets :", int(np.sum(y_val == 1)))

    # ------------------------------------------------------------
    # BUILD MODEL
    # ------------------------------------------------------------

    print()
    print("=" * 65)
    print("🌲 TRAINING RANDOM FOREST")
    print("=" * 65)

    print()
    print("Trees           :", N_ESTIMATORS)
    print("Class weighting : balanced")
    print("Features        :", X_train.shape[1])

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    print()
    print("Fitting model...")

    model.fit(X_train, y_train)

    print("✅ Random Forest training complete.")

    # ------------------------------------------------------------
    # VALIDATION EVALUATION
    # ------------------------------------------------------------

    validation_results = evaluate_model(
        model,
        X_val,
        y_val,
        "Validation",
    )

    # ------------------------------------------------------------
    # VALIDATION PLOTS
    # ------------------------------------------------------------

    save_confusion_matrix(
        validation_results["confusion_matrix"],
        VALIDATION_CM_PATH,
        "ExoSift — Random Forest Validation",
    )

    save_precision_recall_curve(
        y_val,
        validation_results["probabilities"],
        VALIDATION_PR_PATH,
        "ExoSift — Random Forest Validation Precision-Recall",
    )

    print()
    print("Validation plots saved:")
    print(" -", VALIDATION_CM_PATH)
    print(" -", VALIDATION_PR_PATH)

    # ------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------------

    save_feature_importance(
        model,
        feature_names,
    )

    # ------------------------------------------------------------
    # FINAL MODEL
    # ------------------------------------------------------------

    print()
    print("=" * 65)
    print("🌲 TRAINING FINAL RANDOM FOREST")
    print("=" * 65)

    print()
    print("Refitting on ALL training data...")

    final_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    final_model.fit(X, y)

    print("✅ Final Random Forest training complete.")

    # ------------------------------------------------------------
    # FINAL TEST EVALUATION
    # ------------------------------------------------------------

    test_results = evaluate_model(
        final_model,
        X_test,
        y_test,
        "Held-out Kaggle Test",
    )

    # ------------------------------------------------------------
    # TEST PLOTS
    # ------------------------------------------------------------

    save_confusion_matrix(
        test_results["confusion_matrix"],
        TEST_CM_PATH,
        "ExoSift — Random Forest Held-out Test",
    )

    save_precision_recall_curve(
        y_test,
        test_results["probabilities"],
        TEST_PR_PATH,
        "ExoSift — Random Forest Test Precision-Recall",
    )

    print()
    print("Test plots saved:")
    print(" -", TEST_CM_PATH)
    print(" -", TEST_PR_PATH)

    # ------------------------------------------------------------
    # SAVE FINAL MODEL
    # ------------------------------------------------------------

    save_model_bundle(
        final_model,
        feature_names,
    )

    # Recalculate feature importance for the final model.
    final_importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": final_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    final_importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    # ------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------

    print()
    print("=" * 65)
    print("💾 EXOSIFT ARTIFACTS")
    print("=" * 65)

    print()
    print("Model:")
    print(" -", MODEL_PATH)

    print()
    print("Feature importance:")
    print(" -", FEATURE_IMPORTANCE_PATH)

    print()
    print("Plots:")
    print(" -", VALIDATION_CM_PATH)
    print(" -", VALIDATION_PR_PATH)
    print(" -", TEST_CM_PATH)
    print(" -", TEST_PR_PATH)

    print()
    print("=" * 65)
    print("✅ RANDOM FOREST BASELINE COMPLETE")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()