from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SEQUENCE_PATH = DATA_DIR / "train_sequences.npy"
TRAIN_LABEL_PATH = DATA_DIR / "train_labels.npy"

TEST_SEQUENCE_PATH = DATA_DIR / "test_sequences.npy"
TEST_LABEL_PATH = DATA_DIR / "test_labels.npy"

MODEL_PATH = MODEL_DIR / "cnn_1d.keras"

RANDOM_STATE = 42

VALIDATION_SIZE = 0.20

BATCH_SIZE = 64
EPOCHS = 60

LEARNING_RATE = 3e-4

L2_REG = 1e-4


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 72)
print("ExoSift — Optimized 1D CNN Training")
print("=" * 72)

print("\nLoading training sequences...")

X = np.load(TRAIN_SEQUENCE_PATH)
y = np.load(TRAIN_LABEL_PATH)

print(f"Training X shape: {X.shape}")
print(f"Training y shape: {y.shape}")
print(f"X dtype: {X.dtype}")
print(f"y dtype: {y.dtype}")


# ============================================================
# SAFETY CHECKS
# ============================================================

if X.ndim != 2:
    raise ValueError(
        f"Expected X shape (samples, timesteps), got {X.shape}"
    )

if y.ndim != 1:
    raise ValueError(
        f"Expected y shape (samples,), got {y.shape}"
    )

if len(X) != len(y):
    raise ValueError(
        f"X/y sample mismatch: {len(X)} vs {len(y)}"
    )

if not np.isfinite(X).all():
    raise ValueError("Training sequences contain NaN or infinite values.")

if not np.isfinite(y).all():
    raise ValueError("Training labels contain NaN or infinite values.")

unique_labels = np.unique(y)

if not np.array_equal(unique_labels, np.array([0, 1])):
    raise ValueError(
        f"Expected labels [0, 1], found {unique_labels}"
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

unique, counts = np.unique(y, return_counts=True)

print("\nTraining class distribution:")

for label, count in zip(unique, counts):
    name = "Planet" if label == 1 else "Non-planet"
    percentage = count / len(y) * 100

    print(
        f"  {name:12s}: {count:5d} "
        f"({percentage:.3f}%)"
    )


# ============================================================
# ADD CNN CHANNEL DIMENSION
# ============================================================

# Conv1D input:
# (samples, timesteps, channels)

X = X[..., np.newaxis]

print(f"\nCNN input shape: {X.shape}")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nDataset split:")
print(f"  Training   : {X_train.shape}")
print(f"  Validation : {X_val.shape}")

print("\nTraining labels:")

for label, count in zip(*np.unique(y_train, return_counts=True)):
    name = "Planet" if label == 1 else "Non-planet"
    print(f"  {name:12s}: {count}")

print("\nValidation labels:")

for label, count in zip(*np.unique(y_val, return_counts=True)):
    name = "Planet" if label == 1 else "Non-planet"
    print(f"  {name:12s}: {count}")


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("\nClass weights:")

for cls, weight in class_weights.items():
    name = "Planet" if cls == 1 else "Non-planet"
    print(f"  {name:12s}: {weight:.4f}")


# ============================================================
# BUILD CNN
# ============================================================

print("\nBuilding optimized 1D CNN...")

regularizer = tf.keras.regularizers.l2(L2_REG)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=(X_train.shape[1], 1)
        ),

        # ====================================================
        # BLOCK 1
        # ====================================================

        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=15,
            padding="same",
            kernel_regularizer=regularizer,
            use_bias=False,
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ReLU(),

        tf.keras.layers.MaxPooling1D(
            pool_size=4
        ),

        tf.keras.layers.Dropout(0.15),

        # ====================================================
        # BLOCK 2
        # ====================================================

        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=11,
            padding="same",
            kernel_regularizer=regularizer,
            use_bias=False,
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ReLU(),

        tf.keras.layers.MaxPooling1D(
            pool_size=4
        ),

        tf.keras.layers.Dropout(0.20),

        # ====================================================
        # BLOCK 3
        # ====================================================

        tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=7,
            padding="same",
            kernel_regularizer=regularizer,
            use_bias=False,
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ReLU(),

        tf.keras.layers.MaxPooling1D(
            pool_size=4
        ),

        tf.keras.layers.Dropout(0.25),

        # ====================================================
        # DUAL GLOBAL POOLING
        # ====================================================

        tf.keras.layers.GlobalAveragePooling1D(),

        # ====================================================
        # CLASSIFIER
        # ====================================================

        tf.keras.layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=regularizer,
        ),

        tf.keras.layers.Dropout(0.40),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid",
        ),
    ]
)


# ============================================================
# COMPILE
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    clipnorm=1.0,
)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

        tf.keras.metrics.AUC(
            name="roc_auc",
            curve="ROC",
        ),

        tf.keras.metrics.AUC(
            name="pr_auc",
            curve="PR",
        ),
    ],
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nModel summary:\n")

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_pr_auc",
    mode="max",
    patience=10,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_pr_auc",
    mode="max",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1,
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_pr_auc",
    mode="max",
    save_best_only=True,
    verbose=1,
)

terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 72)
print("Starting CNN training")
print("=" * 72)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    shuffle=True,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint,
        terminate_on_nan,
    ],
    verbose=1,
)


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print("\nLoading best checkpoint...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# PREDICTION HELPER
# ============================================================

def evaluate_predictions(
    y_true,
    probabilities,
    threshold,
    title,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    cm = confusion_matrix(
        y_true,
        predictions,
    )

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    print(f"Threshold : {threshold:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "Non-planet",
                "Planet",
            ],
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
    }


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

val_probabilities = model.predict(
    X_val,
    batch_size=BATCH_SIZE,
    verbose=0,
).ravel()


# ============================================================
# VALIDATION — DEFAULT THRESHOLD
# ============================================================

val_default_metrics = evaluate_predictions(
    y_val,
    val_probabilities,
    threshold=0.5,
    title="CNN Validation Results — Threshold 0.50",
)


# ============================================================
# FIND VALIDATION F1 THRESHOLD
# ============================================================

precision_curve, recall_curve, thresholds = (
    precision_recall_curve(
        y_val,
        val_probabilities,
    )
)

if len(thresholds) > 0:
    f1_scores = (
        2
        * precision_curve[:-1]
        * recall_curve[:-1]
        / (
            precision_curve[:-1]
            + recall_curve[:-1]
            + 1e-12
        )
    )

    best_index = int(
        np.argmax(f1_scores)
    )

    best_threshold = float(
        thresholds[best_index]
    )

    best_val_f1 = float(
        f1_scores[best_index]
    )

else:
    best_threshold = 0.5
    best_val_f1 = 0.0


print("\n" + "=" * 72)
print("Validation Threshold Analysis")
print("=" * 72)

print(
    f"Best validation F1 threshold: "
    f"{best_threshold:.4f}"
)

print(
    f"Best validation F1: "
    f"{best_val_f1:.4f}"
)

print(
    "\nImportant: this threshold is selected only "
    "from validation data."
)


# ============================================================
# VALIDATION — TUNED THRESHOLD
# ============================================================

val_tuned_metrics = evaluate_predictions(
    y_val,
    val_probabilities,
    threshold=best_threshold,
    title="CNN Validation Results — Validation-Optimized Threshold",
)


# ============================================================
# VALIDATION CONFUSION MATRIX
# ============================================================

val_predictions = (
    val_probabilities >= best_threshold
).astype(int)

cm = confusion_matrix(
    y_val,
    val_predictions,
)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title(
    "ExoSift CNN — Validation Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["Non-planet", "Planet"],
)

plt.yticks(
    [0, 1],
    ["Non-planet", "Planet"],
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    MODEL_DIR / "cnn_validation_confusion_matrix.png",
    dpi=200,
)

plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    recall_curve,
    precision_curve,
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "ExoSift CNN — Validation Precision-Recall"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    MODEL_DIR / "cnn_validation_precision_recall.png",
    dpi=200,
)

plt.close()


# ============================================================
# TEST SET
# ============================================================

print("\nLoading untouched Kaggle test set...")

X_test = np.load(TEST_SEQUENCE_PATH)
y_test = np.load(TEST_LABEL_PATH)

if X_test.ndim != 2:
    raise ValueError(
        f"Expected test X shape (samples, timesteps), got {X_test.shape}"
    )

if len(X_test) != len(y_test):
    raise ValueError(
        "Test X/y sample mismatch."
    )

if not np.isfinite(X_test).all():
    raise ValueError(
        "Test sequences contain NaN or infinite values."
    )

X_test = X_test[..., np.newaxis]

print(f"Test X shape: {X_test.shape}")
print(f"Test y shape: {y_test.shape}")

print("\nGenerating test predictions...")

test_probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=0,
).ravel()


# ============================================================
# TEST — DEFAULT 0.50
# ============================================================

test_default_metrics = evaluate_predictions(
    y_test,
    test_probabilities,
    threshold=0.5,
    title="CNN Held-Out Test Results — Threshold 0.50",
)


# ============================================================
# TEST — VALIDATION-SELECTED THRESHOLD
# ============================================================

test_tuned_metrics = evaluate_predictions(
    y_test,
    test_probabilities,
    threshold=best_threshold,
    title="CNN Held-Out Test Results — Validation-Selected Threshold",
)


# ============================================================
# SAVE VALIDATION / TEST PREDICTIONS
# ============================================================

validation_predictions = pd.DataFrame(
    {
        "actual_label": y_val,
        "predicted_probability": val_probabilities,
        "predicted_label_default_0_5": (
            val_probabilities >= 0.5
        ).astype(int),
        "predicted_label_tuned": val_predictions,
    }
)

validation_predictions.to_csv(
    MODEL_DIR / "cnn_validation_predictions.csv",
    index=False,
)


test_predictions = pd.DataFrame(
    {
        "actual_label": y_test,
        "predicted_probability": test_probabilities,
        "predicted_label_default_0_5": (
            test_probabilities >= 0.5
        ).astype(int),
        "predicted_label_tuned": (
            test_probabilities >= best_threshold
        ).astype(int),
    }
)

test_predictions.to_csv(
    MODEL_DIR / "cnn_test_predictions.csv",
    index=False,
)


# ============================================================
# TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    MODEL_DIR / "cnn_training_history.csv",
    index=False,
)


# ============================================================
# TRAINING LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss",
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "ExoSift CNN — Training vs Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    MODEL_DIR / "cnn_training_loss.png",
    dpi=200,
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("CNN TRAINING COMPLETE")
print("=" * 72)

print("\nBest model:")
print(f"  {MODEL_PATH}")

print("\nSaved artifacts:")
print("  ✓ cnn_1d.keras")
print("  ✓ cnn_validation_confusion_matrix.png")
print("  ✓ cnn_validation_precision_recall.png")
print("  ✓ cnn_training_history.csv")
print("  ✓ cnn_training_loss.png")
print("  ✓ cnn_validation_predictions.csv")
print("  ✓ cnn_test_predictions.csv")

print("\nKey validation metrics:")
print(
    f"  PR-AUC  : "
    f"{val_tuned_metrics['pr_auc']:.4f}"
)

print(
    f"  ROC-AUC : "
    f"{val_tuned_metrics['roc_auc']:.4f}"
)

print(
    f"  F1      : "
    f"{val_tuned_metrics['f1']:.4f}"
)

print(
    f"  Recall  : "
    f"{val_tuned_metrics['recall']:.4f}"
)

print(
    f"  Precision: "
    f"{val_tuned_metrics['precision']:.4f}"
)

print("\nExoSift CNN stage finished.")