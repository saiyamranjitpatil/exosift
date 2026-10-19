"""
ExoSift — Feature Engineering

Creates 34 engineered features from preprocessed stellar light curves.

Outputs:
    data/processed/train_features.csv
    data/processed/test_features.csv
    data/processed/train_sequences.npy
    data/processed/test_sequences.npy
    data/processed/train_labels.npy
    data/processed/test_labels.npy
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

from preprocessing import preprocess_light_curve


# ================================================================
# PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "raw" / "exoTrain.csv"
TEST_PATH = BASE_DIR / "data" / "raw" / "exoTest.csv"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# EXPECTED FEATURE SET
# ================================================================

FEATURE_NAMES = [
    "mean",
    "std",
    "mad",
    "minimum",
    "maximum",
    "range",
    "q01",
    "q05",
    "q10",
    "q25",
    "q75",
    "q90",
    "q95",
    "q99",
    "iqr",
    "skewness",
    "kurtosis",
    "mean_absolute_difference",
    "difference_std",
    "maximum_absolute_difference",
    "fraction_below_1",
    "fraction_below_2",
    "fraction_below_3",
    "longest_dip_below_1",
    "longest_dip_below_2",
    "negative_area",
    "number_of_negative_peaks",
    "strongest_negative_peak",
    "average_negative_peak_prominence",
    "fraction_above_1",
    "fraction_above_2",
    "autocorr_lag_1",
    "autocorr_lag_5",
    "autocorr_lag_10",
]


# ================================================================
# BASIC HELPERS
# ================================================================

def robust_mad(values):
    """Median absolute deviation."""

    values = np.asarray(values, dtype=np.float64)

    median = np.median(values)

    return np.median(np.abs(values - median))


def safe_autocorrelation(sequence, lag):
    """Calculate lagged autocorrelation safely."""

    sequence = np.asarray(sequence, dtype=np.float64)

    if lag >= len(sequence):
        return 0.0

    x = sequence[:-lag]
    y = sequence[lag:]

    x_std = np.std(x)
    y_std = np.std(y)

    if x_std == 0 or y_std == 0:
        return 0.0

    correlation = np.corrcoef(x, y)[0, 1]

    if not np.isfinite(correlation):
        return 0.0

    return float(correlation)


def longest_run(mask):
    """Return the longest consecutive True run."""

    mask = np.asarray(mask, dtype=bool)

    longest = 0
    current = 0

    for value in mask:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return int(longest)


# ================================================================
# FEATURE EXTRACTION
# ================================================================

def extract_features(processed_flux):
    """
    Extract 34 engineered features from one processed light curve.
    """

    x = np.asarray(processed_flux, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("Light curve must be one-dimensional.")

    if len(x) == 0:
        raise ValueError("Light curve is empty.")

    if not np.isfinite(x).all():
        raise ValueError("Processed light curve contains NaN or infinite values.")

    # ------------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------------

    mean_value = float(np.mean(x))
    std_value = float(np.std(x))

    mad_value = float(robust_mad(x))

    minimum = float(np.min(x))
    maximum = float(np.max(x))
    value_range = maximum - minimum

    q01 = float(np.percentile(x, 1))
    q05 = float(np.percentile(x, 5))
    q10 = float(np.percentile(x, 10))
    q25 = float(np.percentile(x, 25))

    q75 = float(np.percentile(x, 75))
    q90 = float(np.percentile(x, 90))
    q95 = float(np.percentile(x, 95))
    q99 = float(np.percentile(x, 99))

    iqr = q75 - q25

    # ------------------------------------------------------------
    # Distribution shape
    # ------------------------------------------------------------

    skewness = float(skew(x, bias=False))

    kurtosis_value = float(kurtosis(x, bias=False))

    # scipy can theoretically return NaN for constant sequences.
    if not np.isfinite(skewness):
        skewness = 0.0

    if not np.isfinite(kurtosis_value):
        kurtosis_value = 0.0

    # ------------------------------------------------------------
    # Difference / temporal features
    # ------------------------------------------------------------

    differences = np.diff(x)

    if len(differences) > 0:

        mean_absolute_difference = float(
            np.mean(np.abs(differences))
        )

        difference_std = float(
            np.std(differences)
        )

        maximum_absolute_difference = float(
            np.max(np.abs(differences))
        )

    else:

        mean_absolute_difference = 0.0
        difference_std = 0.0
        maximum_absolute_difference = 0.0

    # ------------------------------------------------------------
    # Negative excursion / transit-like features
    # ------------------------------------------------------------

    fraction_below_1 = float(np.mean(x < -1.0))
    fraction_below_2 = float(np.mean(x < -2.0))
    fraction_below_3 = float(np.mean(x < -3.0))

    longest_dip_below_1 = longest_run(x < -1.0)
    longest_dip_below_2 = longest_run(x < -2.0)

    negative_area = float(np.sum(np.maximum(-x, 0.0)))

    # ------------------------------------------------------------
    # Negative peak detection
    # ------------------------------------------------------------

    negative_signal = -x

    peaks, peak_properties = find_peaks(
        negative_signal,
        prominence=0.25,
    )

    number_of_negative_peaks = int(len(peaks))

    if number_of_negative_peaks > 0:

        negative_prominences = peak_properties.get(
            "prominences",
            np.zeros(number_of_negative_peaks),
        )

        strongest_negative_peak = float(
            np.max(negative_signal[peaks])
        )

        average_negative_peak_prominence = float(
            np.mean(negative_prominences)
        )

    else:

        strongest_negative_peak = 0.0
        average_negative_peak_prominence = 0.0

    # ------------------------------------------------------------
    # Positive excursion features
    # ------------------------------------------------------------

    fraction_above_1 = float(np.mean(x > 1.0))
    fraction_above_2 = float(np.mean(x > 2.0))

    # ------------------------------------------------------------
    # Autocorrelation
    # ------------------------------------------------------------

    autocorr_lag_1 = safe_autocorrelation(x, 1)
    autocorr_lag_5 = safe_autocorrelation(x, 5)
    autocorr_lag_10 = safe_autocorrelation(x, 10)

    # ------------------------------------------------------------
    # Build exact feature dictionary
    # ------------------------------------------------------------

    features = {
        "mean": mean_value,
        "std": std_value,
        "mad": mad_value,
        "minimum": minimum,
        "maximum": maximum,
        "range": value_range,
        "q01": q01,
        "q05": q05,
        "q10": q10,
        "q25": q25,
        "q75": q75,
        "q90": q90,
        "q95": q95,
        "q99": q99,
        "iqr": iqr,
        "skewness": skewness,
        "kurtosis": kurtosis_value,
        "mean_absolute_difference": mean_absolute_difference,
        "difference_std": difference_std,
        "maximum_absolute_difference": maximum_absolute_difference,
        "fraction_below_1": fraction_below_1,
        "fraction_below_2": fraction_below_2,
        "fraction_below_3": fraction_below_3,
        "longest_dip_below_1": longest_dip_below_1,
        "longest_dip_below_2": longest_dip_below_2,
        "negative_area": negative_area,
        "number_of_negative_peaks": number_of_negative_peaks,
        "strongest_negative_peak": strongest_negative_peak,
        "average_negative_peak_prominence": average_negative_peak_prominence,
        "fraction_above_1": fraction_above_1,
        "fraction_above_2": fraction_above_2,
        "autocorr_lag_1": autocorr_lag_1,
        "autocorr_lag_5": autocorr_lag_5,
        "autocorr_lag_10": autocorr_lag_10,
    }

    # ------------------------------------------------------------
    # Hard safety check against accidental leakage
    # ------------------------------------------------------------

    if set(features.keys()) != set(FEATURE_NAMES):
        raise RuntimeError(
            "Feature set mismatch. Possible accidental feature addition/removal."
        )

    return features


# ================================================================
# DATASET PROCESSING
# ================================================================

def process_dataset(csv_path, dataset_name):
    """
    Load one dataset, preprocess every light curve, extract features,
    and return feature matrix, labels, and processed sequences.
    """

    print()
    print(f"Loading {dataset_name} dataset...")

    df = pd.read_csv(csv_path)

    if "LABEL" not in df.columns:
        raise ValueError(
            f"{dataset_name}: LABEL column not found."
        )

    # ------------------------------------------------------------
    # Extract labels FIRST.
    #
    # LABEL=1 -> non-planet -> internal 0
    # LABEL=2 -> planet     -> internal 1
    # ------------------------------------------------------------

    original_labels = df["LABEL"].to_numpy(dtype=np.int64)

    labels = np.where(original_labels == 2, 1, 0).astype(np.int64)

    # ------------------------------------------------------------
    # CRITICAL:
    # Remove LABEL before constructing the flux matrix.
    # ------------------------------------------------------------

    flux_df = df.drop(columns=["LABEL"])

    # Extra safety: ensure ONLY flux columns remain.
    if any(column.lower() in {"label", "target", "y", "class"} for column in flux_df.columns):
        raise RuntimeError(
            f"{dataset_name}: target-like column leaked into flux data."
        )

    flux = flux_df.to_numpy(dtype=np.float64)

    print(f"{dataset_name}: raw shape = {flux.shape}")
    print(f"{dataset_name}: labels    = {labels.shape}")

    if flux.shape[1] != 3197:
        print(
            f"⚠️ Warning: expected 3197 flux columns, "
            f"found {flux.shape[1]}."
        )

    print(
        f"{dataset_name}: planets     = {int(np.sum(labels == 1))}"
    )
    print(
        f"{dataset_name}: non-planets = {int(np.sum(labels == 0))}"
    )

    # ------------------------------------------------------------
    # Process every light curve
    # ------------------------------------------------------------

    processed_sequences = []
    feature_rows = []

    total = len(flux)

    print()
    print(f"Building {dataset_name} features...")

    for index, raw_flux in enumerate(flux):

        processed = preprocess_light_curve(raw_flux)

        features = extract_features(processed)

        processed_sequences.append(processed)
        feature_rows.append(features)

        if (index + 1) % 100 == 0 or index + 1 == total:
            print(f"{dataset_name}: {index + 1}/{total}")

    # ------------------------------------------------------------
    # Construct feature dataframe
    # ------------------------------------------------------------

    features_df = pd.DataFrame(
        feature_rows,
        columns=FEATURE_NAMES,
    )

    # ------------------------------------------------------------
    # HARD LEAKAGE CHECKS
    # ------------------------------------------------------------

    forbidden_columns = {
        "label",
        "LABEL",
        "target",
        "TARGET",
        "class",
        "CLASS",
        "y",
    }

    leaked = forbidden_columns.intersection(features_df.columns)

    if leaked:
        raise RuntimeError(
            f"❌ DATA LEAKAGE DETECTED — target column(s) present: {leaked}"
        )

    if len(features_df.columns) != 34:
        raise RuntimeError(
            f"Expected exactly 34 engineered features, "
            f"found {len(features_df.columns)}."
        )

    if features_df.columns.tolist() != FEATURE_NAMES:
        raise RuntimeError(
            "Feature column ordering does not match FEATURE_NAMES."
        )

    if features_df.isnull().any().any():
        raise RuntimeError(
            f"{dataset_name}: feature matrix contains missing values."
        )

    if not np.isfinite(features_df.to_numpy(dtype=np.float64)).all():
        raise RuntimeError(
            f"{dataset_name}: feature matrix contains infinite values."
        )

    # ------------------------------------------------------------
    # Convert processed sequences
    # ------------------------------------------------------------

    processed_sequences = np.asarray(
        processed_sequences,
        dtype=np.float32,
    )

    # ------------------------------------------------------------
    # Final shape checks
    # ------------------------------------------------------------

    if len(features_df) != len(labels):
        raise RuntimeError(
            f"{dataset_name}: number of feature rows and labels differ."
        )

    if len(processed_sequences) != len(labels):
        raise RuntimeError(
            f"{dataset_name}: number of sequences and labels differ."
        )

    return features_df, processed_sequences, labels


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print("=" * 65)
    print("🔭 EXOSIFT — FEATURE ENGINEERING")
    print("=" * 65)

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    train_features, train_sequences, train_labels = process_dataset(
        TRAIN_PATH,
        "TRAIN",
    )

    # ------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------

    test_features, test_sequences, test_labels = process_dataset(
        TEST_PATH,
        "TEST",
    )

    # ------------------------------------------------------------
    # SAVE TRAIN
    # ------------------------------------------------------------

    train_features.to_csv(
        PROCESSED_DIR / "train_features.csv",
        index=False,
    )

    np.save(
        PROCESSED_DIR / "train_sequences.npy",
        train_sequences,
    )

    np.save(
        PROCESSED_DIR / "train_labels.npy",
        train_labels,
    )

    # ------------------------------------------------------------
    # SAVE TEST
    # ------------------------------------------------------------

    test_features.to_csv(
        PROCESSED_DIR / "test_features.csv",
        index=False,
    )

    np.save(
        PROCESSED_DIR / "test_sequences.npy",
        test_sequences,
    )

    np.save(
        PROCESSED_DIR / "test_labels.npy",
        test_labels,
    )

    # ------------------------------------------------------------
    # FINAL VALIDATION
    # ------------------------------------------------------------

    print()
    print("=" * 65)
    print("TRAIN VALIDATION")
    print("=" * 65)

    print("Stars     :", len(train_features))
    print("Features  :", len(train_features.columns))
    print("Sequences :", train_sequences.shape)
    print("Missing   :", int(train_features.isnull().sum().sum()))
    print(
        "Infinite  :",
        int(
            (~np.isfinite(train_features.to_numpy(dtype=float))).sum()
        ),
    )
    print("Planet    :", int(np.sum(train_labels == 1)))
    print("Non-planet:", int(np.sum(train_labels == 0)))

    print()
    print("=" * 65)
    print("TEST VALIDATION")
    print("=" * 65)

    print("Stars     :", len(test_features))
    print("Features  :", len(test_features.columns))
    print("Sequences :", test_sequences.shape)
    print("Missing   :", int(test_features.isnull().sum().sum()))
    print(
        "Infinite  :",
        int(
            (~np.isfinite(test_features.to_numpy(dtype=float))).sum()
        ),
    )
    print("Planet    :", int(np.sum(test_labels == 1)))
    print("Non-planet:", int(np.sum(test_labels == 0)))

    # ------------------------------------------------------------
    # PRINT FEATURE NAMES
    # ------------------------------------------------------------

    print()
    print("=" * 65)
    print("FEATURES")
    print("=" * 65)

    for index, name in enumerate(train_features.columns, start=1):
        print(f"{index:02d}. {name}")

    # ------------------------------------------------------------
    # FINAL LEAKAGE CONFIRMATION
    # ------------------------------------------------------------

    print()
    print("=" * 65)
    print("🔐 DATA LEAKAGE CHECK")
    print("=" * 65)

    print()
    print("LABEL present in train features :", "LABEL" in train_features.columns)
    print("label present in train features :", "label" in train_features.columns)

    if (
        "LABEL" in train_features.columns
        or "label" in train_features.columns
    ):
        raise RuntimeError(
            "❌ FAILED: target leakage remains in train features."
        )

    if (
        "LABEL" in test_features.columns
        or "label" in test_features.columns
    ):
        raise RuntimeError(
            "❌ FAILED: target leakage remains in test features."
        )

    print()
    print("✅ No target column found in feature matrices.")

    print()
    print("=" * 65)
    print("✅ FEATURE ENGINEERING COMPLETE")
    print("=" * 65)

    print()
    print("Saved:")
    print(" -", PROCESSED_DIR / "train_features.csv")
    print(" -", PROCESSED_DIR / "test_features.csv")
    print(" -", PROCESSED_DIR / "train_sequences.npy")
    print(" -", PROCESSED_DIR / "test_sequences.npy")
    print(" -", PROCESSED_DIR / "train_labels.npy")
    print(" -", PROCESSED_DIR / "test_labels.npy")
    print()


if __name__ == "__main__":
    main()