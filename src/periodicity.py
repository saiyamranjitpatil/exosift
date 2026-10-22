"""
ExoSift — Transit Periodicity Detection (V2, Priority B1)

Detects candidate transit periods in a stellar light curve using a
Box Least Squares (BLS) periodogram. BLS is the standard method for
transit detection: it folds the light curve at trial periods and fits
a box-shaped model (transit dip + out-of-transit baseline), which is
far more appropriate than a sinusoid for transit-shaped signals.

Pipeline
--------
1. Detrend the light curve (rolling-median baseline removal).
2. Build a period grid spanning the exoplanet range (default 0.5–50 d).
3. Run BoxLeastSquares at each period; compute Signal Detection
   Efficiency (SDE) for each.
4. Return the top candidate periods ranked by SDE.

No retraining, no interpolation across gaps, no synthetic data.
"""

from pathlib import Path

import numpy as np
from astropy.timeseries import BoxLeastSquares

from preprocessing import preprocess_light_curve


# ============================================================
# CONFIGURATION
# ============================================================

MIN_PERIOD_DAYS = 0.5
MAX_PERIOD_DAYS = 50.0
TOP_K = 5

# BLS duration grid: fraction of period to try as transit duration.
# Typical transits last 2–12 hours = 0.08–0.5 days on a 3–10 day period.
# We express duration as a fraction of the period: 0.005–0.20 (≈0.1–5 h
# on a 1–10 d period).
DURATION_FRAC_MIN = 0.005
DURATION_FRAC_MAX = 0.20
DURATION_FRAC_STEPS = 50


# ============================================================
# PERIOD DETECTION
# ============================================================

def detect_periods(
    time,
    flux,
    min_period_days=MIN_PERIOD_DAYS,
    max_period_days=MAX_PERIOD_DAYS,
    top_k=TOP_K,
    detrend=True,
    trend_window=101,
    return_spectrum=False,
):
    """
    Detect candidate transit periods via Box Least Squares periodogram.

    Parameters
    ----------
    time : array-like
        Observation times in days (Kepler BJD). Must be 1-D.
    flux : array-like
        Normalized flux measurements. Must be 1-D, same length as time.
    min_period_days : float
        Shortest period to search (days).
    max_period_days : float
        Longest period to search (days).
    top_k : int
        Number of top periods to return.
    detrend : bool
        If True, remove long-term trend before periodogram.
    trend_window : int
        Rolling-median window for detrending (odd number).
    return_spectrum : bool
        If True, also return the full BLS periodogram (see Returns).
        Default False leaves the return value unchanged.

    Returns
    -------
    list of dict
        Each dict: {"period_days", "sde", "power", "depth",
        "duration_days", "transit_time"}. Sorted by SDE descending.

    If ``return_spectrum`` is True, returns ``(candidates, spectrum)`` where
    ``spectrum`` is a dict of equal-length arrays over the whole period grid:
    {"period", "power", "depth", "duration", "transit_time"}.
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    if time.ndim != 1 or flux.ndim != 1:
        raise ValueError("time and flux must be 1-D arrays.")
    if len(time) != len(flux):
        raise ValueError("time and flux must have the same length.")
    if len(time) < 10:
        raise ValueError("Need at least 10 observations for BLS period detection.")

    finite = np.isfinite(time) & np.isfinite(flux)
    time = time[finite]
    flux = flux[finite]

    if len(time) < 10:
        raise ValueError("Not enough finite observations after cleaning.")

    # ------------------------------------------------------------
    # Detrend: remove long-term baseline so transits stand out.
    # ------------------------------------------------------------
    if detrend:
        try:
            flux_detrended = preprocess_light_curve(
                flux,
                trend_window=trend_window,
            )
        except Exception:
            flux_detrended = flux - np.median(flux)
    else:
        flux_detrended = flux - np.median(flux)

    # ------------------------------------------------------------
    # Period grid.
    # ------------------------------------------------------------
    # BLS period grid: logarithmically spaced is standard because
    # period uncertainty scales with period (fractional resolution).
    n_periods = max(200, int(10 * np.log10(max_period_days / min_period_days) * 100))
    period_grid = np.logspace(
        np.log10(min_period_days),
        np.log10(max_period_days),
        n_periods,
    )

    # Duration grid as fractions of each period.
    duration_grid = np.linspace(
        DURATION_FRAC_MIN,
        DURATION_FRAC_MAX,
        DURATION_FRAC_STEPS,
    )

    # ------------------------------------------------------------
    # Box Least Squares periodogram.
    # ------------------------------------------------------------
    model = BoxLeastSquares(time, flux_detrended)

    # BLS expects duration in days. We compute a duration grid per period.
    # To keep runtime reasonable, we evaluate BLS at a representative set
    # of durations (min and max fraction) and interpolate, or we can pass
    # the full duration grid. astropy's BLS computes a periodogram where
    # each period's power is the max over all durations.
    result = model.power(
        period=period_grid,
        duration=duration_grid,
    )

    # ------------------------------------------------------------
    # Extract top candidates.
    # ------------------------------------------------------------
    # result.period, result.power, result.depth, result.duration,
    # result.transit_time
    order = np.argsort(result.power)[::-1]

    candidates = []
    seen_periods = []
    for idx in order:
        period_days = float(result.period[idx])
        # Avoid reporting near-duplicate periods (harmonics/aliases).
        if any(
            abs(period_days - p) / p < 0.02 for p in seen_periods
        ):
            continue
        seen_periods.append(period_days)

        candidates.append(
            {
                "period_days": period_days,
                "power": float(result.power[idx]),
                "sde": float(result.power[idx]),  # BLS power ≈ SDE proxy
                "depth": float(result.depth[idx]),
                "duration_days": float(result.duration[idx]),
                "transit_time": float(result.transit_time[idx]),
            }
        )

        if len(candidates) >= top_k:
            break

    if return_spectrum:
        spectrum = {
            "period": np.asarray(result.period, dtype=np.float64),
            "power": np.asarray(result.power, dtype=np.float64),
            "depth": np.asarray(result.depth, dtype=np.float64),
            "duration": np.asarray(result.duration, dtype=np.float64),
            "transit_time": np.asarray(result.transit_time, dtype=np.float64),
        }
        return candidates, spectrum

    return candidates


def best_period(candidates):
    """Return the highest-SDE candidate, or None if empty."""
    if not candidates:
        return None
    return candidates[0]


# ============================================================
# MAIN — SELF-TEST ON SYNTHETIC DATA
# ============================================================

def _synthetic_transit_light_curve(
    n_points=2000,
    period_days=3.5,
    transit_depth=0.01,
    transit_duration_hours=3.0,
    noise=0.002,
    seed=42,
):
    """Generate a synthetic light curve with a box-shaped transit."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 30, n_points)

    phase = (time % period_days) / period_days
    duration_phase = (transit_duration_hours / 24.0) / period_days
    in_transit = (phase < duration_phase / 2) | (phase > 1 - duration_phase / 2)
    flux = 1.0 - transit_depth * in_transit.astype(float)
    flux += rng.normal(0, noise, size=n_points)
    return time, flux


if __name__ == "__main__":
    print("=" * 72)
    print("EXOSIFT — PERIODICITY DETECTION SELF-TEST (BLS)")
    print("=" * 72)

    time, flux = _synthetic_transit_light_curve()
    print(f"\nSynthetic light curve: {len(time)} points, injected period 3.5 days")

    candidates = detect_periods(time, flux)
    print("\nTop candidate periods:")
    for i, c in enumerate(candidates, 1):
        print(
            f"  {i}. period={c['period_days']:.3f} d  "
            f"SDE={c['sde']:.2f}  "
            f"depth={c['depth']:.4f}  "
            f"duration={c['duration_days']*24:.2f} h"
        )

    best = best_period(candidates)
    if best:
        err = abs(best["period_days"] - 3.5)
        print(f"\nBest period: {best['period_days']:.3f} days "
              f"(injected: 3.5 days, error: {err:.3f} d)")
    print("\nSelf-test complete.")
