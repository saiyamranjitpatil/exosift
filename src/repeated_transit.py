"""
ExoSift — Repeated-Transit Evidence (V2, Priority B3)

Evaluates whether a candidate transit signal repeats consistently across
a light curve. A genuine planet produces transits at regular intervals
with consistent depth, duration, and timing. False positives (eclipsing
binaries, instrumental artifacts, stellar variability) typically show
inconsistent or non-periodic behavior.

Pipeline
--------
1. Fold the light curve on the candidate period (via src.phase_fold).
2. Identify individual transit windows (phase bins near transit center).
3. Extract each transit event from the raw (unfolded) light curve.
4. Measure depth, duration, and midpoint for each event.
5. Compute consistency: scatter in depth/duration/timing across events.
6. Return a repeatability score (0-1) and per-event diagnostics.

No model retraining; purely diagnostic on the light curve itself.
"""

from pathlib import Path

import numpy as np

from phase_fold import fold_light_curve


# ============================================================
# CONFIGURATION
# ============================================================

# Fraction of period around transit center to search for each event.
TRANSIT_WINDOW_FRAC = 0.25

# Minimum number of transits required for a repeatability assessment.
MIN_TRANSITS = 3


# ============================================================
# REPEATED-TRANSIT ANALYSIS
# ============================================================

def analyze_repeated_transits(
    time,
    flux,
    period_days,
    transit_time=None,
    min_transits=MIN_TRANSITS,
):
    """
    Analyze whether transit signals repeat consistently.

    Parameters
    ----------
    time : array-like
        Observation times in days.
    flux : array-like
        Normalized flux measurements.
    period_days : float
        Candidate transit period (days).
    transit_time : float or None
        Reference transit midpoint. If None, inferred from phase folding.
    min_transits : int
        Minimum number of transits required.

    Returns
    -------
    dict with keys:
        n_transits : number of transit events detected
        repeatability_score : 0-1, higher = more consistent (None if < min_transits)
        depth_scatter : std of measured depths (relative to mean depth)
        duration_scatter : std of measured durations (relative to mean)
        timing_scatter_days : std of transit midpoint offsets (days)
        events : list of per-event dicts (index, center_time, depth,
                 duration_days, snr)
        folded_profile : result from fold_light_curve
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    if len(time) < 10:
        raise ValueError("Need at least 10 observations.")

    # ------------------------------------------------------------
    # Phase-fold to find transit center and profile.
    # ------------------------------------------------------------
    folded = fold_light_curve(
        time,
        flux,
        period_days,
        transit_time=transit_time,
        detrend=False,
    )
    phase = folded["phase"]
    folded_flux = folded["folded_flux"]
    mean_flux = folded["mean_flux"]
    bin_centers = folded["phase_bin_centers"]

    if transit_time is None:
        transit_time = folded["transit_time"]

    # ------------------------------------------------------------
    # Identify transit center phase (deepest bin).
    # ------------------------------------------------------------
    valid = ~np.isnan(mean_flux)
    if not np.any(valid):
        raise ValueError("No valid phase bins from folding.")

    transit_center_idx = int(np.nanargmin(mean_flux[valid]))
    transit_center_phase = bin_centers[valid][transit_center_idx]

    # ------------------------------------------------------------
    # Find individual transit events in the raw light curve.
    # ------------------------------------------------------------
    # Expected transit midpoints: transit_time + k * period
    t_min, t_max = time.min(), time.max()
    k_min = int(np.floor((t_min - transit_time) / period_days)) - 1
    k_max = int(np.ceil((t_max - transit_time) / period_days)) + 1

    window_half_width = TRANSIT_WINDOW_FRAC * period_days

    events = []
    for k in range(k_min, k_max + 1):
        expected_center = transit_time + k * period_days
        # Only consider events within the observed time range.
        if expected_center < t_min - window_half_width or \
           expected_center > t_max + window_half_width:
            continue

        # Extract points in the transit window.
        mask = np.abs(time - expected_center) <= window_half_width
        if np.count_nonzero(mask) < 3:
            continue

        event_time = time[mask]
        event_flux = flux[mask]

        # Measure depth: median of out-of-window flux minus min in window.
        # Use points outside the central 1/3 of window as local baseline.
        center_mask = np.abs(event_time - expected_center) <= window_half_width / 3
        if np.count_nonzero(center_mask) < 2:
            continue

        local_baseline = np.median(event_flux[~center_mask]) \
            if np.count_nonzero(~center_mask) > 0 else np.median(event_flux)
        transit_flux = event_flux[center_mask]
        depth = local_baseline - np.min(transit_flux)
        min_idx = int(np.argmin(transit_flux))
        event_center = event_time[min_idx]

        # Duration: width where flux < baseline - depth/2 (half-depth).
        threshold = local_baseline - depth / 2.0
        below = event_flux < threshold
        if np.count_nonzero(below) > 0:
            below_times = event_time[below]
            duration_days = float(below_times.max() - below_times.min())
        else:
            duration_days = 0.0

        # Signal-to-noise: depth / std of local baseline.
        baseline_scatter = np.std(event_flux[~center_mask]) \
            if np.count_nonzero(~center_mask) > 1 else np.std(event_flux)
        snr = depth / baseline_scatter if baseline_scatter > 0 else 0.0

        events.append({
            "transit_index": k,
            "center_time": float(event_center),
            "expected_center": float(expected_center),
            "timing_offset_days": float(event_center - expected_center),
            "depth": float(depth),
            "duration_days": duration_days,
            "snr": float(snr),
            "n_points": int(np.count_nonzero(mask)),
        })

    n_transits = len(events)

    # ------------------------------------------------------------
    # Consistency metrics.
    # ------------------------------------------------------------
    if n_transits < min_transits:
        repeatability_score = None
        depth_scatter = None
        duration_scatter = None
        timing_scatter_days = None
    else:
        depths = np.array([e["depth"] for e in events])
        durations = np.array([e["duration_days"] for e in events])
        offsets = np.array([e["timing_offset_days"] for e in events])

        mean_depth = np.mean(depths)
        mean_duration = np.mean(durations)

        depth_scatter = float(np.std(depths) / mean_depth) if mean_depth > 0 else float("inf")
        duration_scatter = float(np.std(durations) / mean_duration) if mean_duration > 0 else float("inf")
        timing_scatter_days = float(np.std(offsets))

        # Repeatability score: combination of consistency metrics.
        # Lower scatter = higher score. Each component scaled to [0, 1].
        depth_consistency = 1.0 / (1.0 + depth_scatter)
        duration_consistency = 1.0 / (1.0 + duration_scatter)
        timing_consistency = 1.0 / (1.0 + timing_scatter_days / period_days)

        repeatability_score = float(
            (depth_consistency + duration_consistency + timing_consistency) / 3.0
        )

    return {
        "n_transits": n_transits,
        "repeatability_score": repeatability_score,
        "depth_scatter": depth_scatter,
        "duration_scatter": duration_scatter,
        "timing_scatter_days": timing_scatter_days,
        "period_days": float(period_days),
        "transit_time": float(transit_time),
        "events": events,
        "folded_profile": folded,
    }


def is_repeatable(result, score_threshold=0.7):
    """
    Heuristic: is the signal repeatable enough to be a planet candidate?

    Returns (bool, reason_string).
    """
    if result["n_transits"] < MIN_TRANSITS:
        return False, f"Only {result['n_transits']} transits (need >= {MIN_TRANSITS})"

    score = result["repeatability_score"]
    if score is None:
        return False, "No repeatability score computed"

    if score >= score_threshold:
        return True, f"Repeatability score {score:.2f} >= {score_threshold}"
    else:
        return False, f"Repeatability score {score:.2f} < {score_threshold}"


# ============================================================
# MAIN — SELF-TEST ON SYNTHETIC DATA
# ============================================================

def _synthetic_transit_light_curve(
    n_points=4000,
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


def _inconsistent_transit_light_curve(
    n_points=4000,
    period_days=3.5,
    base_depth=0.01,
    noise=0.002,
    seed=43,
):
    """Transits with varying depth (inconsistent = likely false positive)."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 60, n_points)
    phase = (time % period_days) / period_days
    duration_phase = (3.0 / 24.0) / period_days
    in_transit = (phase < duration_phase / 2) | (phase > 1 - duration_phase / 2)
    # Depth varies randomly per transit.
    depths = base_depth * (0.5 + rng.random(len(time)) * 0.01)  # small variation
    flux = 1.0 - depths * in_transit.astype(float)
    flux += rng.normal(0, noise, size=n_points)
    return time, flux


if __name__ == "__main__":
    print("=" * 72)
    print("EXOSIFT — REPEATED-TRANSIT EVIDENCE SELF-TEST")
    print("=" * 72)

    # --- Consistent signal (real planet-like) ---
    time, flux = _synthetic_transit_light_curve()
    from periodicity import detect_periods, best_period
    candidates = detect_periods(time, flux, detrend=False)
    best = best_period(candidates)

    result = analyze_repeated_transits(
        time, flux, best["period_days"], transit_time=best.get("transit_time")
    )
    print(f"\n[Consistent signal] Period: {best['period_days']:.3f} d")
    print(f"  Transits detected : {result['n_transits']}")
    print(f"  Repeatability score: {result['repeatability_score']:.3f}")
    print(f"  Depth scatter     : {result['depth_scatter']:.4f}")
    print(f"  Duration scatter  : {result['duration_scatter']:.4f}")
    print(f"  Timing scatter    : {result['timing_scatter_days']:.4f} d")
    repeatable, reason = is_repeatable(result)
    print(f"  Repeatable? {repeatable} — {reason}")

    print("\nSelf-test complete.")
