"""
ExoSift — Phase Folding (V2, Priority B2)

Folds a stellar light curve on a detected transit period to reveal the
average transit shape. Phase folding aligns all transits on top of each
other by computing orbital phase = (time - transit_time) % period / period,
then binning in phase to produce a mean transit profile.

This does not detect planets by itself; it visualizes and characterizes
a period already found by the periodicity detector (B1).
"""

from pathlib import Path

import numpy as np

from periodicity import detect_periods, best_period


# ============================================================
# CONFIGURATION
# ============================================================

N_PHASE_BINS = 200  # Number of phase bins for the folded profile


# ============================================================
# PHASE FOLDING
# ============================================================

def fold_light_curve(
    time,
    flux,
    period_days,
    transit_time=None,
    n_bins=N_PHASE_BINS,
    center_on_transit=True,
    detrend=True,
):
    """
    Fold a light curve on a given period and bin by orbital phase.

    Parameters
    ----------
    time : array-like
        Observation times in days (Kepler BJD). Must be 1-D.
    flux : array-like
        Normalized flux measurements. Must be 1-D, same length as time.
    period_days : float
        Folding period in days.
    transit_time : float or None
        Reference transit midpoint (days). If None, uses time[0] as the
        phase zero point (transits may not align perfectly but the shape
        is still recoverable).
    n_bins : int
        Number of phase bins for the mean profile.
    center_on_transit : bool
        If True, phase 0.0 = transit midpoint. If False, phase 0.0 =
        transit_time (same thing when transit_time is provided).
    detrend : bool
        If True, remove long-term trend before folding.

    Returns
    -------
    dict with keys:
        phase_bin_centers : (n_bins,) array of phase values in [0, 1)
        mean_flux : (n_bins,) array of mean flux per phase bin
        std_flux : (n_bins,) array of std flux per phase bin
        n_per_bin : (n_bins,) array of point counts per phase bin
        phase : (N,) array of per-point phase values
        folded_flux : (N,) array of per-point flux in phase order
        period_days : the folding period used
        transit_depth : estimated depth = 1 - min(mean_flux) (if bins exist)
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    if time.ndim != 1 or flux.ndim != 1:
        raise ValueError("time and flux must be 1-D arrays.")
    if len(time) != len(flux):
        raise ValueError("time and flux must have the same length.")
    if len(time) < 4:
        raise ValueError("Need at least 4 observations for phase folding.")
    if period_days <= 0:
        raise ValueError("period_days must be positive.")

    finite = np.isfinite(time) & np.isfinite(flux)
    time = time[finite]
    flux = flux[finite]

    if len(time) < 4:
        raise ValueError("Not enough finite observations after cleaning.")

    # ------------------------------------------------------------
    # Optional detrend.
    # ------------------------------------------------------------
    if detrend:
        try:
            from preprocessing import preprocess_light_curve

            flux = preprocess_light_curve(flux, trend_window=101)
        except Exception:
            flux = flux - np.median(flux)
    else:
        flux = flux - np.median(flux)

    # ------------------------------------------------------------
    # Compute orbital phase for each point.
    # ------------------------------------------------------------
    if transit_time is None:
        transit_time = time[0]

    phase = (time - transit_time) / period_days
    phase = phase - np.floor(phase)  # wrap into [0, 1)

    # Sort by phase so the folded curve is continuous.
    order = np.argsort(phase)
    phase_sorted = phase[order]
    flux_sorted = flux[order]

    # ------------------------------------------------------------
    # Bin in phase.
    # ------------------------------------------------------------
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    mean_flux = np.full(n_bins, np.nan)
    std_flux = np.full(n_bins, np.nan)
    n_per_bin = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (phase_sorted >= bin_edges[i]) & (phase_sorted < bin_edges[i + 1])
        if i == n_bins - 1:
            # Include the right edge on the last bin.
            mask = (phase_sorted >= bin_edges[i]) & (phase_sorted <= bin_edges[i + 1])
        count = int(np.count_nonzero(mask))
        n_per_bin[i] = count
        if count > 0:
            bin_flux = flux_sorted[mask]
            mean_flux[i] = float(np.mean(bin_flux))
            std_flux[i] = float(np.std(bin_flux)) if count > 1 else 0.0

    # ------------------------------------------------------------
    # Estimate transit depth from the folded profile.
    # ------------------------------------------------------------
    valid = ~np.isnan(mean_flux)
    transit_depth = None
    if np.any(valid):
        out_of_transit = float(np.nanmedian(mean_flux[valid]))
        min_flux = float(np.nanmin(mean_flux[valid]))
        transit_depth = out_of_transit - min_flux

    return {
        "phase_bin_centers": bin_centers,
        "mean_flux": mean_flux,
        "std_flux": std_flux,
        "n_per_bin": n_per_bin,
        "phase": phase_sorted,
        "folded_flux": flux_sorted,
        "period_days": float(period_days),
        "transit_time": float(transit_time),
        "transit_depth": transit_depth,
    }


def fold_best_period(
    time,
    flux,
    top_period=None,
    **fold_kwargs,
):
    """
    Convenience: detect periods then fold on the best one.

    If top_period is provided, folds on that period instead of detecting.
    Returns (fold_result, candidates).
    """
    if top_period is None:
        candidates = detect_periods(time, flux)
        best = best_period(candidates)
        if best is None:
            raise ValueError("No period detected; cannot fold.")
        period_days = best["period_days"]
        transit_time = best.get("transit_time")
    else:
        candidates = detect_periods(time, flux)
        period_days = top_period
        transit_time = None

    fold_result = fold_light_curve(
        time,
        flux,
        period_days,
        transit_time=transit_time,
        **fold_kwargs,
    )
    return fold_result, candidates


# ============================================================
# MAIN — SELF-TEST ON SYNTHETIC DATA
# ============================================================

def _synthetic_transit_light_curve(
    n_points=3000,
    period_days=3.5,
    transit_depth=0.01,
    transit_duration_hours=3.0,
    noise=0.002,
    seed=42,
):
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 60, n_points)  # 60 days, ~17 transits
    phase = (time % period_days) / period_days
    duration_phase = (transit_duration_hours / 24.0) / period_days
    in_transit = (phase < duration_phase / 2) | (phase > 1 - duration_phase / 2)
    flux = 1.0 - transit_depth * in_transit.astype(float)
    flux += rng.normal(0, noise, size=n_points)
    return time, flux


if __name__ == "__main__":
    print("=" * 72)
    print("EXOSIFT — PHASE FOLDING SELF-TEST")
    print("=" * 72)

    time, flux = _synthetic_transit_light_curve()
    print(f"\nSynthetic light curve: {len(time)} points, period=3.5 d")

    # Detect period first.
    from periodicity import detect_periods, best_period
    candidates = detect_periods(time, flux, detrend=False)
    best = best_period(candidates)
    print(f"Detected period: {best['period_days']:.3f} days "
          f"(injected: 3.5 days)")

    # Fold on detected period (data already flat; no detrend needed).
    result = fold_light_curve(
        time,
        flux,
        best["period_days"],
        transit_time=best.get("transit_time"),
        detrend=False,
    )

    print(f"\nFolded profile: {len(result['phase_bin_centers'])} phase bins")
    print(f"Transit depth (from folded profile): {result['transit_depth']:.5f}"
          f" (injected: 0.010)")

    # Show the deepest bin (transit center).
    valid = ~np.isnan(result["mean_flux"])
    min_idx = int(np.nanargmin(result["mean_flux"][valid]))
    min_phase = result["phase_bin_centers"][valid][min_idx]
    min_flux = result["mean_flux"][valid][min_idx]
    print(f"Transit center: phase={min_phase:.3f}, flux={min_flux:.5f}")
    print(f"Points in transit bin: {result['n_per_bin'][valid][min_idx]}")

    print("\nSelf-test complete.")
