from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf
import lightkurve as lk

from preprocessing import preprocess_light_curve


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "cnn_1d.keras"

VARIABLE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "cnn_1d_variable.keras"
)

OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "real_star_predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Original CNN training length.
REFERENCE_LENGTH = 3197

# We allow shorter real sequences at inference.
# Below this, the time context becomes too short for
# a meaningful V1 comparison.
MIN_INFERENCE_LENGTH = 1024

# When a contiguous real run is longer than the
# reference training length, analyze overlapping
# 3197-point windows.
WINDOW_STRIDE = REFERENCE_LENGTH // 2

# Large temporal gaps split independent sequences.
# Normal Kepler cadence is ~29.4 minutes.
MAX_GAP_MULTIPLIER = 3.0

# Threshold selected from validation data.
DEFAULT_THRESHOLD = 0.0939

BATCH_SIZE = 64

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "ExoSift — Real Kepler star inference "
            "using NASA/MAST and Lightkurve."
        )
    )

    parser.add_argument(
        "--kic",
        type=int,
        required=True,
        help="Kepler Input Catalog ID.",
    )

    parser.add_argument(
        "--quarter",
        type=int,
        default=None,
        help=(
            "Optional Kepler quarter "
            "(0–17)."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Planet score threshold. "
            f"Default: {DEFAULT_THRESHOLD}"
        ),
    )

    parser.add_argument(
        "--all-windows",
        action="store_true",
        help=(
            "Analyze every valid window in the "
            "best usable real-data segment."
        ),
    )

    return parser.parse_args()


# ============================================================
# VALIDATION
# ============================================================

def validate_arguments(
    kic,
    quarter,
    threshold,
):
    if kic <= 0:
        raise ValueError(
            "KIC ID must be positive."
        )

    if quarter is not None:
        if not 0 <= quarter <= 17:
            raise ValueError(
                "Kepler quarter must be between 0 and 17."
            )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )


# ============================================================
# LOAD ORIGINAL MODEL
# ============================================================

def load_reference_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained CNN was not found:\n"
            f"{MODEL_PATH}"
        )

    print(
        "\nLoading trained ExoSift CNN..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "✓ Trained CNN loaded."
    )

    print(
        f"Reference model input shape: "
        f"{model.input_shape}"
    )

    expected = (
        None,
        REFERENCE_LENGTH,
        1,
    )

    if model.input_shape != expected:
        raise ValueError(
            "\nUnexpected trained model input shape.\n"
            f"Expected: {expected}\n"
            f"Got:      {model.input_shape}"
        )

    return model


# ============================================================
# BUILD VARIABLE-LENGTH INFERENCE MODEL
# ============================================================

def build_variable_length_model(
    reference_model,
):
    """
    Recreate the exact CNN architecture but allow
    arbitrary sequence length during inference.

    We then copy the learned weights from the trained
    3197-point model.

    No retraining occurs.
    """

    print(
        "\nPreparing variable-length CNN..."
    )

    regularizer = tf.keras.regularizers.l2(
        1e-4
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(None, 1)
            ),

            # Block 1
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

            tf.keras.layers.Dropout(
                0.15
            ),

            # Block 2
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

            tf.keras.layers.Dropout(
                0.20
            ),

            # Block 3
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

            tf.keras.layers.Dropout(
                0.25
            ),

            # Global average pooling allows
            # variable sequence lengths.
            tf.keras.layers.GlobalAveragePooling1D(),

            tf.keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=regularizer,
            ),

            tf.keras.layers.Dropout(
                0.40
            ),

            tf.keras.layers.Dense(
                1,
                activation="sigmoid",
            ),
        ]
    )

    # --------------------------------------------------------
    # Verify weight compatibility.
    # --------------------------------------------------------

    reference_weights = (
        reference_model.get_weights()
    )

    new_weights = (
        model.get_weights()
    )

    if len(reference_weights) != len(
        new_weights
    ):
        raise RuntimeError(
            "Reference and variable CNN architectures "
            "have different numbers of weight tensors."
        )

    for index, (
        ref_weight,
        new_weight,
    ) in enumerate(
        zip(
            reference_weights,
            new_weights,
        )
    ):

        if ref_weight.shape != new_weight.shape:
            raise RuntimeError(
                "Weight shape mismatch at tensor "
                f"{index}: "
                f"{ref_weight.shape} vs "
                f"{new_weight.shape}"
            )

    # --------------------------------------------------------
    # Copy exact learned weights.
    # --------------------------------------------------------

    model.set_weights(
        reference_weights
    )

    print(
        "✓ Exact trained CNN weights transferred."
    )

    print(
        f"Variable CNN input shape: "
        f"{model.input_shape}"
    )

    # --------------------------------------------------------
    # Save adapter.
    # --------------------------------------------------------

    model.save(
        VARIABLE_MODEL_PATH
    )

    print(
        f"✓ Variable-length model saved:\n"
        f"  {VARIABLE_MODEL_PATH}"
    )

    return model


# ============================================================
# SEARCH MAST
# ============================================================

def search_kepler(
    kic,
    quarter=None,
):
    target = f"KIC {kic}"

    print("\n" + "=" * 72)
    print("NASA / MAST SEARCH")
    print("=" * 72)

    print(
        f"Target: {target}"
    )

    if quarter is not None:
        print(
            f"Quarter: Q{quarter:02d}"
        )

    else:
        print(
            "Quarter: all available"
        )

    search_kwargs = {
        "target": target,
        "mission": "Kepler",
        "exptime": "long",
    }

    if quarter is not None:
        search_kwargs[
            "quarter"
        ] = quarter

    results = lk.search_lightcurve(
        **search_kwargs
    )

    if len(results) == 0:
        raise RuntimeError(
            f"No Kepler long-cadence data found "
            f"for {target}."
        )

    print(
        f"\nProducts found: {len(results)}"
    )

    for i in range(
        min(len(results), 20)
    ):
        print(
            f"\n[{i}]"
        )

        print(
            results[i]
        )

    return results


# ============================================================
# QUARTER IDENTIFICATION
# ============================================================

def extract_quarter(product):
    match = re.search(
        r"Quarter\s*(\d+)",
        str(product),
        flags=re.IGNORECASE,
    )

    if match:
        return int(
            match.group(1)
        )

    return None


# ============================================================
# DOWNLOAD
# ============================================================

def download_lightcurve(
    product,
):
    try:
        lc = product.download(
            quality_bitmask="default"
        )

    except Exception as exc:
        print(
            f"✗ Download failed: {exc}"
        )

        return None

    if lc is None:
        print(
            "✗ No light curve returned."
        )

        return None

    return lc


# ============================================================
# EXTRACT FLUX
# ============================================================

def extract_flux(lc):
    column_names = [
        str(name).lower()
        for name in lc.colnames
    ]

    # Prefer pipeline-corrected data.
    preferred = [
        "pdcsap_flux",
        "sap_flux",
    ]

    for preferred_name in preferred:

        if preferred_name in column_names:

            actual_name = lc.colnames[
                column_names.index(
                    preferred_name
                )
            ]

            flux = np.asarray(
                lc[actual_name].value,
                dtype=np.float64,
            )

            return (
                flux,
                actual_name,
            )

    flux = np.asarray(
        lc.flux.value,
        dtype=np.float64,
    )

    return (
        flux,
        "flux",
    )


# ============================================================
# CLEAN TIME / FLUX
# ============================================================

def clean_time_flux(
    time,
    flux,
):
    time = np.asarray(
        time,
        dtype=np.float64,
    )

    flux = np.asarray(
        flux,
        dtype=np.float64,
    )

    if len(time) != len(flux):
        raise ValueError(
            "Time and flux lengths do not match."
        )

    valid = (
        np.isfinite(time)
        & np.isfinite(flux)
    )

    time = time[valid]
    flux = flux[valid]

    if len(time) < MIN_INFERENCE_LENGTH:
        raise ValueError(
            f"Only {len(time)} valid points remain; "
            f"minimum is {MIN_INFERENCE_LENGTH}."
        )

    # Chronological order.
    order = np.argsort(time)

    time = time[order]
    flux = flux[order]

    # Remove duplicate timestamps.
    time, indices = np.unique(
        time,
        return_index=True,
    )

    flux = flux[indices]

    return (
        time,
        flux,
    )


# ============================================================
# FIND REAL CONTIGUOUS RUNS
# ============================================================

def find_contiguous_runs(
    time,
):
    if len(time) < 2:
        return []

    gaps = np.diff(time)

    median_gap = float(
        np.median(gaps)
    )

    if (
        not np.isfinite(median_gap)
        or median_gap <= 0
    ):
        return []

    gap_limit = (
        median_gap
        * MAX_GAP_MULTIPLIER
    )

    break_indices = np.where(
        gaps > gap_limit
    )[0]

    starts = np.concatenate(
        [
            np.array([0]),
            break_indices + 1,
        ]
    )

    ends = np.concatenate(
        [
            break_indices,
            np.array(
                [len(time) - 1]
            ),
        ]
    )

    runs = []

    for start, end in zip(
        starts,
        ends,
    ):

        length = (
            end
            - start
            + 1
        )

        if length >= MIN_INFERENCE_LENGTH:

            run_gaps = gaps[
                start:end
            ]

            runs.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "length": int(length),
                    "median_gap": float(
                        np.median(
                            run_gaps
                        )
                    )
                    if len(run_gaps)
                    else median_gap,
                    "max_gap": float(
                        np.max(
                            run_gaps
                        )
                    )
                    if len(run_gaps)
                    else 0.0,
                }
            )

    runs.sort(
        key=lambda item:
            item["length"],
        reverse=True,
    )

    return runs


# ============================================================
# CREATE WINDOWS
# ============================================================

def create_windows_for_run(
    run,
    all_windows,
):
    length = run[
        "length"
    ]

    # --------------------------------------------------------
    # Long enough for original model window.
    # --------------------------------------------------------

    if length >= REFERENCE_LENGTH:

        if not all_windows:
            return [
                {
                    "start": 0,
                    "end": REFERENCE_LENGTH,
                }
            ]

        max_start = (
            length
            - REFERENCE_LENGTH
        )

        starts = list(
            range(
                0,
                max_start + 1,
                WINDOW_STRIDE,
            )
        )

        # Always include final window.
        if (
            not starts
            or starts[-1] != max_start
        ):
            starts.append(
                max_start
            )

        return [
            {
                "start": int(start),
                "end": int(
                    start
                    + REFERENCE_LENGTH
                ),
            }
            for start in sorted(
                set(starts)
            )
        ]

    # --------------------------------------------------------
    # Shorter real run.
    #
    # Use ALL real observations.
    # No interpolation.
    # --------------------------------------------------------

    return [
        {
            "start": 0,
            "end": length,
        }
    ]


# ============================================================
# PREPROCESS WINDOW
# ============================================================

def preprocess_window(
    flux_window,
):
    if len(flux_window) < (
        MIN_INFERENCE_LENGTH
    ):
        raise ValueError(
            "Inference window is too short."
        )

    if not np.isfinite(
        flux_window
    ).all():
        raise ValueError(
            "Window contains NaN or Inf."
        )

    processed = preprocess_light_curve(
        flux_window
    )

    processed = np.asarray(
        processed,
        dtype=np.float32,
    )

    if processed.ndim != 1:
        raise ValueError(
            "Processed light curve must be 1D."
        )

    if len(processed) != len(
        flux_window
    ):
        raise ValueError(
            "Preprocessing changed sequence length."
        )

    if not np.isfinite(
        processed
    ).all():
        raise ValueError(
            "Processed sequence contains NaN/Inf."
        )

    return processed


# ============================================================
# PREDICT WINDOWS
# ============================================================

def predict_windows(
    model,
    time,
    flux,
    run,
    all_windows,
    threshold,
):
    run_start = run[
        "start"
    ]

    run_end = (
        run["end"]
        + 1
    )

    run_time = time[
        run_start:
        run_end
    ]

    run_flux = flux[
        run_start:
        run_end
    ]

    windows = create_windows_for_run(
        run,
        all_windows,
    )

    processed_sequences = []
    metadata = []

    for number, window in enumerate(
        windows,
        start=1,
    ):

        start = window[
            "start"
        ]

        end = window[
            "end"
        ]

        time_window = run_time[
            start:end
        ]

        flux_window = run_flux[
            start:end
        ]

        processed = preprocess_window(
            flux_window
        )

        processed_sequences.append(
            processed
        )

        metadata.append(
            {
                "window_number": number,
                "start_index": int(
                    start
                ),
                "end_index": int(
                    end - 1
                ),
                "length": int(
                    len(
                        processed
                    )
                ),
                "start_time": float(
                    time_window[0]
                ),
                "end_time": float(
                    time_window[-1]
                ),
                "duration_days": float(
                    time_window[-1]
                    - time_window[0]
                ),
            }
        )

    # Because all windows within one run have the
    # same length, they can be batched safely.
    X = np.stack(
        processed_sequences
    )

    X = X[
        ...,
        np.newaxis,
    ]

    print(
        f"\nRunning CNN on "
        f"{len(X)} window(s)..."
    )

    probabilities = (
        model.predict(
            X,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        .ravel()
    )

    results = []

    for meta, probability, processed, raw in zip(
        metadata,
        probabilities,
        processed_sequences,
        [
            run_flux[
                item["start"]:
                item["end"]
            ]
            for item in windows
        ],
    ):

        probability = float(
            probability
        )

        results.append(
            {
                **meta,
                "planet_score":
                    probability,
                "candidate":
                    bool(
                        probability
                        >= threshold
                    ),
                "time":
                    run_time[
                        meta[
                            "start_index"
                        ]:
                        meta[
                            "end_index"
                        ] + 1
                    ],
                "raw_flux":
                    raw,
                "processed_flux":
                    processed,
            }
        )

    return results


# ============================================================
# SAVE WINDOW CSV
# ============================================================

def save_window_csv(
    kic,
    quarter,
    result,
):
    q_part = (
        f"_Q{quarter:02d}"
        if quarter is not None
        else "_Qunknown"
    )

    filename = (
        f"KIC_{kic}"
        f"{q_part}"
        f"_window_"
        f"{result['window_number']:02d}.csv"
    )

    path = (
        OUTPUT_DIR
        / filename
    )

    df = pd.DataFrame(
        {
            "time":
                result["time"],
            "raw_flux":
                result["raw_flux"],
            "processed_flux":
                result["processed_flux"],
        }
    )

    df.to_csv(
        path,
        index=False,
    )

    return path


# ============================================================
# SAVE BEST PLOT
# ============================================================

def save_best_plot(
    kic,
    quarter,
    result,
    threshold,
):
    q_part = (
        f"_Q{quarter:02d}"
        if quarter is not None
        else "_Qunknown"
    )

    path = (
        OUTPUT_DIR
        / (
            f"KIC_{kic}"
            f"{q_part}"
            f"_best_window.png"
        )
    )

    probability = result[
        "planet_score"
    ]

    status = (
        "PLANET CANDIDATE"
        if probability >= threshold
        else "LIKELY NON-PLANET"
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
    )

    axes[0].plot(
        result["time"],
        result["raw_flux"],
        linewidth=0.7,
    )

    axes[0].set_title(
        f"KIC {kic} — Real Kepler Flux"
    )

    axes[0].set_xlabel(
        "Time (BTJD)"
    )

    axes[0].set_ylabel(
        "Flux"
    )

    axes[0].grid(
        True,
        alpha=0.3,
    )

    axes[1].plot(
        result["processed_flux"],
        linewidth=0.8,
    )

    axes[1].set_title(
        f"ExoSift CNN Input — {status}\n"
        f"Planet score: "
        f"{probability:.6f} | "
        f"Threshold: "
        f"{threshold:.6f}"
    )

    axes[1].set_xlabel(
        "Model timestep"
    )

    axes[1].set_ylabel(
        "Processed flux"
    )

    axes[1].grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return path


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_arguments()

    validate_arguments(
        args.kic,
        args.quarter,
        args.threshold,
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "ExoSift — REAL KEPLER INFERENCE"
    )

    print(
        "=" * 72
    )

    print(
        "\nInference policy:"
    )

    print(
        "  ✓ Real Kepler observations only"
    )

    print(
        "  ✓ No interpolation"
    )

    print(
        "  ✓ No synthetic cadences"
    )

    print(
        "  ✓ Large gaps split into separate runs"
    )

    print(
        "  ✓ Original trained CNN weights retained"
    )

    print(
        "  ✓ Variable sequence length allowed at inference"
    )

    # --------------------------------------------------------
    # Load trained model.
    # --------------------------------------------------------

    reference_model = (
        load_reference_model()
    )

    model = build_variable_length_model(
        reference_model
    )

    # --------------------------------------------------------
    # Search MAST.
    # --------------------------------------------------------

    search_results = search_kepler(
        args.kic,
        args.quarter,
    )

    # --------------------------------------------------------
    # Find best real-data segment across products.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "SEARCHING FOR BEST REAL-DATA SEGMENT"
    )

    print(
        "=" * 72
    )

    candidates = []

    for product_index in range(
        len(search_results)
    ):

        product = search_results[
            product_index
        ]

        quarter = extract_quarter(
            product
        )

        label = (
            f"Q{quarter:02d}"
            if quarter is not None
            else "Unknown quarter"
        )

        print(
            f"\n[{product_index + 1}/"
            f"{len(search_results)}] "
            f"Checking {label}"
        )

        lc = download_lightcurve(
            product
        )

        if lc is None:
            continue

        try:
            time = np.asarray(
                lc.time.value,
                dtype=np.float64,
            )

            flux, flux_source = (
                extract_flux(lc)
            )

            time, flux = (
                clean_time_flux(
                    time,
                    flux,
                )
            )

            runs = find_contiguous_runs(
                time
            )

        except Exception as exc:

            print(
                f"  ✗ Rejected: {exc}"
            )

            continue

        if not runs:
            print(
                "  ✗ No usable real-data run."
            )

            continue

        for run in runs:

            candidates.append(
                {
                    "product": product,
                    "lightcurve": lc,
                    "quarter": quarter,
                    "time": time,
                    "flux": flux,
                    "flux_source":
                        flux_source,
                    "run": run,
                }
            )

        print(
            f"  ✓ Usable runs: "
            f"{len(runs)}"
        )

        print(
            f"  ✓ Longest run: "
            f"{runs[0]['length']} real cadences"
        )

    if not candidates:
        raise RuntimeError(
            "\nNo suitable real-data segment found.\n\n"
            f"Required minimum: "
            f"{MIN_INFERENCE_LENGTH} real cadences.\n"
            "Try another KIC or a specific quarter."
        )

    # --------------------------------------------------------
    # Select the longest real segment.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item:
            item["run"]["length"],
        reverse=True,
    )

    selected = candidates[0]

    selected_quarter = (
        selected["quarter"]
    )

    selected_run = (
        selected["run"]
    )

    selected_time = (
        selected["time"]
    )

    selected_flux = (
        selected["flux"]
    )

    selected_flux_source = (
        selected["flux_source"]
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "BEST REAL-DATA SEGMENT"
    )

    print(
        "=" * 72
    )

    if selected_quarter is not None:
        print(
            f"Quarter: Q{selected_quarter:02d}"
        )

    print(
        f"Real observations: "
        f"{selected_run['length']}"
    )

    print(
        f"Median cadence gap: "
        f"{selected_run['median_gap']:.6f} days"
    )

    print(
        f"Maximum internal gap: "
        f"{selected_run['max_gap']:.6f} days"
    )

    print(
        f"Flux source: "
        f"{selected_flux_source}"
    )

    # --------------------------------------------------------
    # Run CNN.
    # --------------------------------------------------------

    results = predict_windows(
        model=model,
        time=selected_time,
        flux=selected_flux,
        run=selected_run,
        all_windows=args.all_windows,
        threshold=args.threshold,
    )

    # --------------------------------------------------------
    # Print window results.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "WINDOW RESULTS"
    )

    print(
        "=" * 72
    )

    for result in results:

        print(
            f"Window "
            f"{result['window_number']:02d} | "
            f"Length: "
            f"{result['length']:4d} | "
            f"Score: "
            f"{result['planet_score']:.6f} | "
            f"{'CANDIDATE' if result['candidate'] else 'non-candidate'}"
        )

    # --------------------------------------------------------
    # Pick strongest window.
    # --------------------------------------------------------

    best_result = max(
        results,
        key=lambda item:
            item["planet_score"],
    )

    candidate_count = sum(
        result["candidate"]
        for result in results
    )

    # --------------------------------------------------------
    # Save files.
    # --------------------------------------------------------

    saved_csv = []

    for result in results:

        saved_csv.append(
            save_window_csv(
                args.kic,
                selected_quarter,
                result,
            )
        )

    plot_path = save_best_plot(
        args.kic,
        selected_quarter,
        best_result,
        args.threshold,
    )

    final_result = {
        "kic": int(args.kic),
        "quarter": (
            int(selected_quarter)
            if selected_quarter is not None
            else None
        ),
        "flux_source":
            selected_flux_source,
        "real_cadences":
            int(
                selected_run[
                    "length"
                ]
            ),
        "cnn_reference_length":
            REFERENCE_LENGTH,
        "inference_sequence_length":
            int(
                best_result[
                    "length"
                ]
            ),
        "median_cadence_gap_days":
            float(
                selected_run[
                    "median_gap"
                ]
            ),
        "maximum_internal_gap_days":
            float(
                selected_run[
                    "max_gap"
                ]
            ),
        "windows_analyzed":
            int(len(results)),
        "candidate_windows":
            int(candidate_count),
        "threshold":
            float(args.threshold),
        "best_planet_score":
            float(
                best_result[
                    "planet_score"
                ]
            ),
        "prediction":
            (
                "PLANET CANDIDATE"
                if best_result[
                    "planet_score"
                ] >= args.threshold
                else
                "LIKELY NON-PLANET"
            ),
        "best_window": {
            "number":
                int(
                    best_result[
                        "window_number"
                    ]
                ),
            "start_time":
                float(
                    best_result[
                        "start_time"
                    ]
                ),
            "end_time":
                float(
                    best_result[
                        "end_time"
                    ]
                ),
            "duration_days":
                float(
                    best_result[
                        "duration_days"
                    ]
                ),
            "score":
                float(
                    best_result[
                        "planet_score"
                    ]
                ),
        },
        "model_note": (
            "The original fixed-length CNN was trained "
            "on 3197-point sequences. For real-star "
            "inference, the same learned convolutional "
            "weights are used with variable-length input. "
            "No retraining or artificial interpolation is "
            "performed."
        ),
        "scientific_warning": (
            "This is an ML candidate score, not an "
            "astronomical confirmation of an exoplanet."
        ),
    }

    q_part = (
        f"_Q{selected_quarter:02d}"
        if selected_quarter is not None
        else "_Qunknown"
    )

    json_path = (
        OUTPUT_DIR
        / (
            f"KIC_{args.kic}"
            f"{q_part}"
            f"_result.json"
        )
    )

    json_path.write_text(
        json.dumps(
            final_result,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final output.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "EXOSIFT REAL-STAR RESULT"
    )

    print(
        "=" * 72
    )

    print(
        f"KIC ID              : "
        f"{args.kic}"
    )

    print(
        f"Quarter             : "
        f"Q{selected_quarter:02d}"
        if selected_quarter is not None
        else "Unknown"
    )

    print(
        f"Flux source         : "
        f"{selected_flux_source}"
    )

    print(
        f"Real observations   : "
        f"{selected_run['length']}"
    )

    print(
        f"CNN input used      : "
        f"{best_result['length']}"
    )

    print(
        f"Best planet score   : "
        f"{best_result['planet_score']:.6f}"
    )

    print(
        f"Threshold           : "
        f"{args.threshold:.6f}"
    )

    print(
        f"Candidate windows   : "
        f"{candidate_count}/{len(results)}"
    )

    print(
        "\nPrediction:"
    )

    if (
        best_result[
            "planet_score"
        ]
        >= args.threshold
    ):
        print(
            "  🔭 PLANET CANDIDATE"
        )

    else:
        print(
            "  ○ LIKELY NON-PLANET"
        )

    print(
        "\nSaved:"
    )

    for path in saved_csv:
        print(
            f"  ✓ {path}"
        )

    print(
        f"  ✓ {plot_path}"
    )

    print(
        f"  ✓ {json_path}"
    )

    print(
        "\nImportant:"
    )

    print(
        "  No interpolated samples were supplied to the CNN."
    )

    print(
        "  Large Kepler gaps were not treated as "
        "continuous observations."
    )

    print(
        "  The output is a candidate score, not a "
        "confirmed astronomical discovery."
    )

    print(
        "\nExoSift real-star inference finished successfully."
    )


# ============================================================
# SCORE SEGMENT (for Streamlit app reuse)
# ============================================================

def score_segment(
    time,
    flux,
    threshold=DEFAULT_THRESHOLD,
    model=None,
):
    """
    Preprocess a light-curve segment and score it with the variable-length CNN.

    Parameters
    ----------
    time : array-like
        Observation times in days.
    flux : array-like
        Flux measurements.
    threshold : float
        Planet score threshold.
    model : tf.keras.Model or None
        Loaded variable-length CNN. If None, loads cnn_1d_variable.keras.

    Returns
    -------
    dict with keys:
        planet_score : float
        length : int
        threshold : float
        is_candidate : bool
    """
    import tensorflow as tf

    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    if model is None:
        model = tf.keras.models.load_model(VARIABLE_MODEL_PATH)

    processed = preprocess_window(flux)

    # Variable-length CNN accepts any length >= MIN_INFERENCE_LENGTH.
    input_seq = processed[np.newaxis, :, np.newaxis].astype(np.float32)
    planet_score = float(model.predict(input_seq, verbose=0)[0, 0])

    return {
        "planet_score": planet_score,
        "length": len(processed),
        "threshold": float(threshold),
        "is_candidate": planet_score >= threshold,
    }


if __name__ == "__main__":
    main()