"""
ExoSift — Candidate Ranking & Explainability (V2, Priority B4)

Provides two explainability mechanisms:

1. CNN saliency maps: gradients of the planet-output probability wrt
   each input flux point, highlighting which parts of the light curve
   drive the CNN's classification.

2. Composite candidate ranking: combines CNN score, RF score, BLS SDE,
   transit depth, and repeatability into a single ranked list.

Uses only the V1 models (cnn_1d.keras, random_forest.joblib) and V2
periodicity outputs — no retraining.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from joblib import load

from phase_fold import fold_light_curve
from repeated_transit import analyze_repeated_transits


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

CNN_MODEL_PATH = MODEL_DIR / "cnn_1d.keras"
RF_MODEL_PATH = MODEL_DIR / "random_forest.joblib"
RF_FEATURE_IMPORTANCE_PATH = MODEL_DIR / "random_forest_feature_importance.csv"


# ============================================================
# CNN SALIENCY MAPS
# ============================================================

def compute_saliency(
    model,
    flux_sequence,
    class_index=0,
):
    """
    Compute saliency map for a CNN input sequence.

    The saliency map is the absolute gradient of the output probability
    wrt each input flux point. High values = input points that most
    influence the classification.

    Parameters
    ----------
    model : tf.keras.Model
        Trained 1D CNN (expects input shape (batch, timesteps, 1)).
    flux_sequence : array-like
        1-D flux sequence (timesteps,). Will be reshaped to (1, T, 1).
    class_index : int
        Output index to differentiate (0 for sigmoid output).

    Returns
    -------
    dict with keys:
        saliency : (T,) array of absolute gradients per input point
        normalized_saliency : (T,) array scaled to [0, 1]
        planet_probability : float, model output for this input
    """
    flux_sequence = np.asarray(flux_sequence, dtype=np.float64)
    if flux_sequence.ndim != 1:
        raise ValueError("flux_sequence must be 1-D.")

    input_seq = flux_sequence[np.newaxis, :, np.newaxis]  # (1, T, 1)
    input_tensor = tf.convert_to_tensor(input_seq, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        prediction = model(input_tensor, training=False)
        # Sigmoid output: single value per sample.
        loss = prediction[:, class_index]

    gradients = tape.gradient(loss, input_tensor)
    if gradients is None:
        raise RuntimeError("Gradient computation returned None.")

    saliency = np.abs(gradients.numpy()[0, :, 0])  # (T,)

    # Normalize to [0, 1] for comparison across sequences.
    s_max = saliency.max()
    s_min = saliency.min()
    if s_max > s_min:
        normalized = (saliency - s_min) / (s_max - s_min)
    else:
        normalized = np.zeros_like(saliency)

    planet_probability = float(prediction.numpy()[0, 0])

    return {
        "saliency": saliency,
        "normalized_saliency": normalized,
        "planet_probability": planet_probability,
    }


def load_cnn_model():
    """Load the trained 1D CNN."""
    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"CNN model not found: {CNN_MODEL_PATH}")
    return tf.keras.models.load_model(CNN_MODEL_PATH)


# ============================================================
# COMPOSITE CANDIDATE RANKING
# ============================================================

def rank_candidate(
    cnn_probability,
    rf_probability,
    bls_sde,
    transit_depth,
    repeatability_score,
    weights=None,
):
    """
    Compute a composite candidate score (0-1) from multiple signals.

    Parameters
    ----------
    cnn_probability : float
        CNN planet probability (0-1).
    rf_probability : float
        Random Forest planet probability (0-1).
    bls_sde : float
        Box Least Squares Signal Detection Efficiency (>= 0).
    transit_depth : float
        Relative transit depth (>= 0).
    repeatability_score : float or None
        Repeated-transit consistency score (0-1), or None.
    weights : dict or None
        Weights for each component. Defaults give equal weight to
        ML scores and equal weight to transit signals.

    Returns
    -------
    float : composite score in [0, 1].
    dict : per-component contribution breakdown.
    """
    if weights is None:
        weights = {
            "cnn": 0.25,
            "rf": 0.25,
            "bls": 0.20,
            "depth": 0.15,
            "repeatability": 0.15,
        }

    # Normalize BLS SDE to [0, 1] using a saturating transform.
    # SDE > 10 is strong; SDE ~ 5-10 is moderate.
    bls_norm = min(bls_sde / 10.0, 1.0) if bls_sde is not None else 0.0

    # Normalize transit depth: typical transits are 0.005-0.02 (0.5-2%).
    # Cap at 0.05 (5%) for normalization.
    depth_norm = min(transit_depth / 0.05, 1.0) if transit_depth is not None else 0.0

    rep = repeatability_score if repeatability_score is not None else 0.0

    components = {
        "cnn": float(cnn_probability) if cnn_probability is not None else 0.0,
        "rf": float(rf_probability) if rf_probability is not None else 0.0,
        "bls": bls_norm,
        "depth": depth_norm,
        "repeatability": float(rep),
    }

    composite = sum(
        weights[k] * components[k] for k in weights
    )

    return float(composite), components


def rank_candidates(candidate_list):
    """
    Rank a list of candidate dicts by composite score.

    Each candidate dict should contain keys: cnn_probability,
    rf_probability, bls_sde, transit_depth, repeatability_score.

    Returns
    -------
    list of (candidate_dict, composite_score, components) sorted
    by composite_score descending.
    """
    ranked = []
    for cand in candidate_list:
        score, components = rank_candidate(
            cnn_probability=cand.get("cnn_probability"),
            rf_probability=cand.get("rf_probability"),
            bls_sde=cand.get("bls_sde"),
            transit_depth=cand.get("transit_depth"),
            repeatability_score=cand.get("repeatability_score"),
        )
        ranked.append((cand, score, components))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# ============================================================
# RF FEATURE IMPORTANCE
# ============================================================

def load_rf_feature_importance(top_n=None):
    """Load RF feature importance CSV, optionally top N."""
    if not RF_FEATURE_IMPORTANCE_PATH.exists():
        return None
    import pandas as pd
    df = pd.read_csv(RF_FEATURE_IMPORTANCE_PATH)
    if top_n is not None:
        df = df.head(top_n)
    return df


# ============================================================
# MAIN — SELF-TEST ON SYNTHETIC DATA
# ============================================================

def _synthetic_flux_sequence(n_points=3197, seed=42):
    rng = np.random.default_rng(seed)
    # Mostly flat with a small dip (transit-like).
    flux = np.ones(n_points) + rng.normal(0, 0.002, size=n_points)
    flux[1000:1050] -= 0.01  # 50-point dip
    return flux


if __name__ == "__main__":
    print("=" * 72)
    print("EXOSIFT — EXPLAINABILITY SELF-TEST")
    print("=" * 72)

    # --- CNN saliency ---
    try:
        model = load_cnn_model()
        flux_seq = _synthetic_flux_sequence()
        saliency_result = compute_saliency(model, flux_seq)
        print(f"\nCNN saliency:")
        print(f"  Planet probability: {saliency_result['planet_probability']:.4f}")
        print(f"  Saliency shape: {saliency_result['saliency'].shape}")
        print(f"  Max saliency at point: {int(np.argmax(saliency_result['saliency']))}")
    except FileNotFoundError as exc:
        print(f"\nCNN saliency: skipped ({exc})")

    # --- Composite ranking ---
    print(f"\nComposite ranking examples:")
    candidates = [
        {"name": "Strong candidate", "cnn_probability": 0.9, "rf_probability": 0.8,
         "bls_sde": 15.0, "transit_depth": 0.01, "repeatability_score": 0.8},
        {"name": "Weak candidate", "cnn_probability": 0.3, "rf_probability": 0.2,
         "bls_sde": 3.0, "transit_depth": 0.002, "repeatability_score": 0.4},
    ]
    for name, cnn, rf, bls, depth, rep in [
        ("Strong candidate", 0.9, 0.8, 15.0, 0.01, 0.8),
        ("Weak candidate", 0.3, 0.2, 3.0, 0.002, 0.4),
    ]:
        score, components = rank_candidate(cnn, rf, bls, depth, rep)
        print(f"  {name}: composite={score:.3f}  "
              f"cnn={components['cnn']:.2f} rf={components['rf']:.2f} "
              f"bls={components['bls']:.2f} depth={components['depth']:.2f} "
              f"rep={components['repeatability']:.2f}")

    # --- RF feature importance ---
    fi = load_rf_feature_importance(top_n=5)
    if fi is not None:
        print(f"\nTop 5 RF features:")
        for _, row in fi.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    else:
        print("\nRF feature importance: not found (run train_rf.py first).")

    print("\nSelf-test complete.")
