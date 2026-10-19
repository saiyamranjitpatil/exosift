from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# EXOSIFT — PREPROCESSING CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "raw" / "exoTrain.csv"

# Rolling window used to estimate the long-term baseline.
# Must be an odd number.
TREND_WINDOW = 101

# Robust outlier threshold.
# Larger value = less aggressive outlier removal.
OUTLIER_THRESHOLD = 7.0

# Final safety limit after normalization.
NORMALIZED_CLIP = 5.0


# ============================================================
# PREPROCESS ONE LIGHT CURVE
# ============================================================

def preprocess_light_curve(
    flux,
    trend_window=TREND_WINDOW,
    outlier_threshold=OUTLIER_THRESHOLD,
    normalized_clip=NORMALIZED_CLIP,
):
    """
    Clean and normalize one stellar light curve.

    Pipeline
    --------
    1. Validate the input.
    2. Handle NaN / infinite values.
    3. Estimate the local baseline.
    4. Detect extreme outliers using MAD.
    5. Interpolate over extreme outliers.
    6. Re-estimate the baseline.
    7. Remove the long-term trend.
    8. Robustly normalize the signal.
    9. Clip only extreme final values.

    Parameters
    ----------
    flux : array-like
        One-dimensional sequence of flux measurements.

    trend_window : int
        Rolling-median window used to estimate the long-term trend.

    outlier_threshold : float
        Threshold for robust outlier detection.

    normalized_clip : float
        Maximum absolute value allowed after normalization.

    Returns
    -------
    np.ndarray
        Cleaned, detrended and normalized light curve.
    """

    # --------------------------------------------------------
    # 1. Convert input to NumPy array
    # --------------------------------------------------------

    flux = np.asarray(flux, dtype=np.float64)

    if flux.ndim != 1:
        raise ValueError("Expected a 1D light curve.")

    if len(flux) < 3:
        raise ValueError("Light curve contains too few data points.")

    # Make sure the trend window is valid.
    trend_window = int(trend_window)

    if trend_window < 3:
        raise ValueError("trend_window must be at least 3.")

    if trend_window % 2 == 0:
        trend_window += 1

    if trend_window > len(flux):
        trend_window = len(flux)

        if trend_window % 2 == 0:
            trend_window -= 1

    # --------------------------------------------------------
    # 2. Handle NaN / infinite values
    # --------------------------------------------------------

    finite_mask = np.isfinite(flux)

    if not finite_mask.all():

        valid_indices = np.flatnonzero(finite_mask)

        if len(valid_indices) < 2:
            raise ValueError(
                "Light curve does not contain enough valid values."
            )

        invalid_indices = np.flatnonzero(~finite_mask)

        flux[invalid_indices] = np.interp(
            invalid_indices,
            valid_indices,
            flux[valid_indices],
        )

    # Keep a working copy.
    cleaned = flux.copy()

    # --------------------------------------------------------
    # 3. Estimate local baseline
    # --------------------------------------------------------

    baseline = (
        pd.Series(cleaned)
        .rolling(
            window=trend_window,
            center=True,
            min_periods=1,
        )
        .median()
        .to_numpy()
    )

    # --------------------------------------------------------
    # 4. Detect extreme outliers using MAD
    # --------------------------------------------------------

    residual = cleaned - baseline

    residual_median = np.median(residual)

    mad = np.median(
        np.abs(residual - residual_median)
    )

    # Avoid division by zero.
    if mad < 1e-10:
        mad = 1e-10

    robust_z = (
        np.abs(residual - residual_median)
        / (1.4826 * mad)
    )

    outlier_mask = robust_z > outlier_threshold

    # --------------------------------------------------------
    # 5. Replace extreme outliers by interpolation
    # --------------------------------------------------------

    if np.any(outlier_mask):

        good_indices = np.flatnonzero(~outlier_mask)
        bad_indices = np.flatnonzero(outlier_mask)

        if len(good_indices) >= 2:

            cleaned[bad_indices] = np.interp(
                bad_indices,
                good_indices,
                cleaned[good_indices],
            )

    # --------------------------------------------------------
    # 6. Re-estimate baseline after outlier handling
    # --------------------------------------------------------

    baseline = (
        pd.Series(cleaned)
        .rolling(
            window=trend_window,
            center=True,
            min_periods=1,
        )
        .median()
        .to_numpy()
    )

    # --------------------------------------------------------
    # 7. Flatten long-term trend
    # --------------------------------------------------------

    flattened = cleaned - baseline

    # --------------------------------------------------------
    # 8. Robust normalization
    # --------------------------------------------------------

    flattened_median = np.median(flattened)

    flattened_mad = np.median(
        np.abs(flattened - flattened_median)
    )

    if flattened_mad < 1e-10:
        flattened_mad = 1e-10

    normalized = (
        flattened - flattened_median
    ) / (1.4826 * flattened_mad)

    # --------------------------------------------------------
    # 9. Prevent extreme artifacts from dominating models
    # --------------------------------------------------------

    normalized = np.clip(
        normalized,
        -normalized_clip,
        normalized_clip,
    )

    return normalized.astype(np.float32)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():
    """
    Load the Kaggle training dataset.

    Returns
    -------
    X : np.ndarray
        Raw light curves with shape:
        (number_of_stars, number_of_flux_measurements)

    y : np.ndarray
        Binary labels:
        0 = non-planet
        1 = planet
    """

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{TRAIN_PATH}"
        )

    df = pd.read_csv(TRAIN_PATH)

    flux_columns = [
        column
        for column in df.columns
        if column.startswith("FLUX.")
    ]

    if "LABEL" not in df.columns:
        raise ValueError("Dataset does not contain a LABEL column.")

    if not flux_columns:
        raise ValueError("No FLUX columns found in the dataset.")

    X = df[flux_columns].to_numpy(
        dtype=np.float64
    )

    # Original dataset:
    # LABEL 1 = non-planet
    # LABEL 2 = planet
    #
    # Convert to:
    # 0 = non-planet
    # 1 = planet

    y = (
        df["LABEL"].to_numpy() == 2
    ).astype(np.int32)

    return X, y


# ============================================================
# PREPROCESS ENTIRE DATASET
# ============================================================

def preprocess_dataset(X):
    """
    Apply preprocessing to every light curve.

    Parameters
    ----------
    X : np.ndarray
        Raw light curves.

    Returns
    -------
    np.ndarray
        Preprocessed light curves.
    """

    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(
            "Expected X to have shape (samples, time_steps)."
        )

    processed = np.zeros(
        X.shape,
        dtype=np.float32,
    )

    total = len(X)

    for i, light_curve in enumerate(X):

        processed[i] = preprocess_light_curve(
            light_curve
        )

        # Progress update every 500 samples.
        if (
            (i + 1) % 500 == 0
            or i == total - 1
        ):
            print(
                f"Processed {i + 1}/{total} light curves"
            )

    return processed


# ============================================================
# VISUALIZATION
# ============================================================

def plot_comparison(
    raw,
    processed,
    title,
):
    """
    Plot raw and preprocessed versions
    of the same light curve.
    """

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
    )

    # --------------------------------------------------------
    # Raw signal
    # --------------------------------------------------------

    axes[0].plot(
        raw,
        linewidth=0.8,
    )

    axes[0].set_title(
        f"{title} — Raw"
    )

    axes[0].set_ylabel("Flux")
    axes[0].grid(
        True,
        alpha=0.3,
    )

    # --------------------------------------------------------
    # Processed signal
    # --------------------------------------------------------

    axes[1].plot(
        processed,
        linewidth=0.8,
    )

    axes[1].set_title(
        f"{title} — Preprocessed"
    )

    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Normalized Flux")

    axes[1].grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN — TEST THE PIPELINE
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("🔭 EXOSIFT — PREPROCESSING PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_raw, y = load_training_data()

    print(
        f"\nDataset shape: {X_raw.shape}"
    )

    print(
        f"Labels shape : {y.shape}"
    )

    print(
        f"Planet examples     : {np.sum(y == 1)}"
    )

    print(
        f"Non-planet examples : {np.sum(y == 0)}"
    )

    # --------------------------------------------------------
    # Select one example from each class
    # --------------------------------------------------------

    planet_indices = np.flatnonzero(y == 1)
    non_planet_indices = np.flatnonzero(y == 0)

    if len(planet_indices) == 0:
        raise ValueError(
            "No planet examples found."
        )

    if len(non_planet_indices) == 0:
        raise ValueError(
            "No non-planet examples found."
        )

    planet_index = planet_indices[0]
    non_planet_index = non_planet_indices[0]

    planet_raw = X_raw[planet_index]
    non_planet_raw = X_raw[non_planet_index]

    # --------------------------------------------------------
    # Process examples
    # --------------------------------------------------------

    print("\nProcessing planet example...")

    planet_processed = preprocess_light_curve(
        planet_raw
    )

    print(
        "Processing non-planet example..."
    )

    non_planet_processed = preprocess_light_curve(
        non_planet_raw
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    print("\nPREPROCESSING CHECKS")
    print("-" * 60)

    print(
        "Planet processed min/max:"
        f" {planet_processed.min():.3f}"
        f" / {planet_processed.max():.3f}"
    )

    print(
        "Non-planet processed min/max:"
        f" {non_planet_processed.min():.3f}"
        f" / {non_planet_processed.max():.3f}"
    )

    print(
        "Planet finite values:"
        f" {np.isfinite(planet_processed).all()}"
    )

    print(
        "Non-planet finite values:"
        f" {np.isfinite(non_planet_processed).all()}"
    )

    # --------------------------------------------------------
    # Visual comparison
    # --------------------------------------------------------

    plot_comparison(
        planet_raw,
        planet_processed,
        "ExoSift — Planet Candidate",
    )

    plot_comparison(
        non_planet_raw,
        non_planet_processed,
        "ExoSift — Non-Planet",
    )

    print()
    print("=" * 60)
    print("✅ PREPROCESSING TEST COMPLETE")
    print("=" * 60)