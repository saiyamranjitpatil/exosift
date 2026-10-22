"""
ExoSift — dashboard analysis pipeline (V2.1)

Pure analysis layer for the Streamlit dashboard. It contains no UI code and
reuses the existing V1/V2 modules unchanged (predict_real_star, periodicity,
phase_fold, repeated_transit, explainability, calibration).

What this module adds on top of the existing modules
----------------------------------------------------
* Keeps *every* contiguous segment that was scored (not just the best one), the
  quarter's real gaps, and the exact observation counts at each stage, so the
  UI can explain "4,070 observations in the quarter, 1,925 used by the CNN".
* Keeps the full BLS periodogram, so any peak can be inspected.
* Builds a "period view" (fold + windowed repeated-transit analysis) for any
  chosen period, and a shuffled-flux control for the repeatability metric.
* Derives data-driven caution flags from the computed numbers.

Everything is computed from the observations returned by NASA/MAST; nothing is
hard-coded or carried over from earlier runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares

from predict_real_star import (
    DEFAULT_THRESHOLD,
    MAX_GAP_MULTIPLIER,
    MIN_INFERENCE_LENGTH,
    REFERENCE_LENGTH,
    VARIABLE_MODEL_PATH,
    clean_time_flux,
    download_lightcurve,
    extract_flux,
    extract_quarter,
    find_contiguous_runs,
    score_segment,
    search_kepler,
)
from periodicity import (
    DURATION_FRAC_MAX,
    DURATION_FRAC_MIN,
    DURATION_FRAC_STEPS,
    best_period,
    detect_periods,
)
from phase_fold import fold_light_curve
from preprocessing import NORMALIZED_CLIP, OUTLIER_THRESHOLD, TREND_WINDOW, preprocess_light_curve
from repeated_transit import analyze_repeated_transits, is_repeatable
from explainability import rank_candidate

log = logging.getLogger("exosift.pipeline")

# Validated CNN input lengths (see models/variable_length_validation.csv).
VALIDATED_LENGTHS = (1024, 1925, 3197)

# Composite score weights actually used for a live target (RF has no score, so
# its 0.25 weight is dropped and the others are renormalised to sum to 1).
COMPOSITE_BASE_WEIGHTS = {"cnn": 0.25, "bls": 0.20, "depth": 0.15, "repeatability": 0.15}

# Heuristic caution thresholds (documented in docs/scientific_audit.md).
LONG_DURATION_FRACTION = 0.15
HARMONIC_TOLERANCE = 0.02
HARMONIC_LADDER_MIN_COUNT = 3
SHUFFLE_CONTROLS = 20
REPEATABLE_CUTOFF = 0.7


class NoDataError(RuntimeError):
    """MAST returned no Kepler long-cadence data for the request."""


class RetrievalError(RuntimeError):
    """The MAST search failed for a reason other than 'no data' (network, service)."""


# ============================================================
# DATA CONTAINERS
# ============================================================

@dataclass
class Segment:
    """One contiguous run of observations (no gap larger than 3x the median cadence)."""

    index: int
    start: int                      # slice start into the quarter's cleaned arrays
    stop: int                       # slice stop (exclusive)
    n_points: int
    t_start: float
    t_stop: float
    scorable: bool                  # long enough for the CNN (>= MIN_INFERENCE_LENGTH)
    planet_score: Optional[float] = None
    skipped_reason: Optional[str] = None

    @property
    def span_days(self) -> float:
        return float(self.t_stop - self.t_start)


@dataclass
class QuarterData:
    quarter: Optional[int]
    flux_source: str
    n_rows: int                     # rows returned by Lightkurve
    n_valid: int                    # after dropping non-finite / duplicate samples
    span_days: float
    median_cadence_min: float
    gaps: list[dict]                # {"t_start", "t_stop", "days"}
    segments: list[Segment]
    time: np.ndarray                # BKJD days, cleaned, sorted
    flux: np.ndarray                # relative flux (flux / median)

    @property
    def label(self) -> str:
        return "Q?" if self.quarter is None else f"Q{self.quarter:02d}"

    @property
    def n_removed(self) -> int:
        return self.n_rows - self.n_valid


@dataclass
class BLSResult:
    candidates: list[dict]
    spectrum: dict[str, np.ndarray]
    settings: dict[str, Any]
    point_sigma: float              # robust per-point scatter, relative flux
    outlier_cut_depth: float        # dips deeper than this are removed by preprocessing


@dataclass
class PeriodView:
    """Everything derived for one chosen period."""

    source: str                     # "bls_best" | "bls_peak" | "trial"
    period_days: float
    transit_time: float
    duration_days: float
    power: float
    bls_depth_sigma: float
    fold: Optional[dict] = None
    rt: Optional[dict] = None
    repeatable: Optional[tuple] = None
    fold_summary: Optional[dict] = None
    errors: dict[str, str] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    kic: int
    requested_quarter: Optional[int]
    max_products: int
    analyzed_at: str
    model_file: str
    products_found: int = 0
    products_scanned: int = 0
    quarters: list[QuarterData] = field(default_factory=list)
    best_quarter_index: Optional[int] = None
    best_segment_index: Optional[int] = None
    warnings: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    bls: Optional[BLSResult] = None
    top_view: Optional[PeriodView] = None
    null_control: Optional[dict] = None
    calibrated: Optional[float] = None
    composite: Optional[dict] = None

    @property
    def best_quarter(self) -> Optional[QuarterData]:
        return None if self.best_quarter_index is None else self.quarters[self.best_quarter_index]

    @property
    def best_segment(self) -> Optional[Segment]:
        q = self.best_quarter
        return None if q is None or self.best_segment_index is None else q.segments[self.best_segment_index]

    @property
    def scored_segments(self) -> list[tuple[QuarterData, Segment]]:
        return [(q, s) for q in self.quarters for s in q.segments if s.planet_score is not None]


# ============================================================
# QUARTER PREPARATION AND SCORING
# ============================================================

def find_all_runs(time: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) slices of every contiguous run, using the same gap rule as
    predict_real_star.find_contiguous_runs but without the minimum-length filter."""
    if len(time) < 2:
        return [(0, len(time))] if len(time) else []
    gaps = np.diff(time)
    median_gap = float(np.median(gaps))
    if not np.isfinite(median_gap) or median_gap <= 0:
        return [(0, len(time))]
    breaks = np.flatnonzero(gaps > median_gap * MAX_GAP_MULTIPLIER)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks + 1, [len(time)]])
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def find_gaps(time: np.ndarray) -> list[dict]:
    if len(time) < 2:
        return []
    gaps = np.diff(time)
    median_gap = float(np.median(gaps))
    out = []
    for i in np.flatnonzero(gaps > median_gap * MAX_GAP_MULTIPLIER):
        out.append({"t_start": float(time[i]), "t_stop": float(time[i + 1]), "days": float(gaps[i])})
    return out


def prepare_quarter(
    time_raw: np.ndarray,
    flux_raw: np.ndarray,
    quarter: Optional[int],
    flux_source: str,
    model: Any,
) -> QuarterData:
    """Clean one quarter, split it into contiguous runs and CNN-score every scorable run.

    Raises ValueError if too few valid points remain (propagated from clean_time_flux).
    """
    time_raw = np.asarray(time_raw, dtype=np.float64)
    flux_raw = np.asarray(flux_raw, dtype=np.float64)
    n_rows = len(time_raw)

    time, flux = clean_time_flux(time_raw, flux_raw)

    scorable = {(r["start"], r["end"] + 1) for r in find_contiguous_runs(time)}
    segments: list[Segment] = []
    for i, (start, stop) in enumerate(find_all_runs(time)):
        seg = Segment(
            index=i, start=start, stop=stop, n_points=stop - start,
            t_start=float(time[start]), t_stop=float(time[stop - 1]),
            scorable=(start, stop) in scorable,
        )
        if seg.scorable:
            try:
                # Raw flux goes to the CNN: its own preprocessing is scale-free, and this
                # keeps scores identical to src/predict_real_star.py.
                res = score_segment(
                    time[start:stop], flux[start:stop], threshold=DEFAULT_THRESHOLD, model=model
                )
                seg.planet_score = float(res["planet_score"])
            except ValueError as exc:
                seg.skipped_reason = f"{type(exc).__name__}: {exc}"
        else:
            seg.skipped_reason = f"run shorter than {MIN_INFERENCE_LENGTH} points"
        segments.append(seg)

    return QuarterData(
        quarter=quarter,
        flux_source=flux_source,
        n_rows=n_rows,
        n_valid=len(time),
        span_days=float(time[-1] - time[0]),
        median_cadence_min=float(np.median(np.diff(time)) * 1440.0),
        gaps=find_gaps(time),
        segments=segments,
        time=time,
        flux=flux / np.median(flux),
    )


def select_best(quarters: list[QuarterData]) -> tuple[Optional[int], Optional[int]]:
    """Highest-scoring scored segment across all quarters (earliest wins ties)."""
    best, best_score = (None, None), -np.inf
    for qi, q in enumerate(quarters):
        for si, s in enumerate(q.segments):
            if s.planet_score is not None and s.planet_score > best_score:
                best, best_score = (qi, si), s.planet_score
    return best


# ============================================================
# BLS AND PERIOD VIEWS
# ============================================================

def preprocessing_outlier_cut(flux_rel: np.ndarray) -> tuple[float, float]:
    """Replicates preprocessing.preprocess_light_curve step 4 on relative flux.

    Returns (robust per-point sigma, depth beyond which dips are treated as outliers
    and interpolated away before BLS).
    """
    flux_rel = np.asarray(flux_rel, dtype=np.float64)
    baseline = pd.Series(flux_rel).rolling(window=TREND_WINDOW, center=True, min_periods=1).median().to_numpy()
    resid = flux_rel - baseline
    mad = float(np.median(np.abs(resid - np.median(resid))))
    sigma = 1.4826 * mad
    return sigma, OUTLIER_THRESHOLD * sigma


def run_bls(time: np.ndarray, flux_rel: np.ndarray) -> BLSResult:
    candidates, spectrum = detect_periods(time, flux_rel, detrend=True, return_spectrum=True)
    sigma, cut = preprocessing_outlier_cut(flux_rel)
    settings = {
        "period_min_days": float(spectrum["period"].min()),
        "period_max_days": float(spectrum["period"].max()),
        "n_periods": int(len(spectrum["period"])),
        "period_spacing": "logarithmic",
        "duration_min_days": float(DURATION_FRAC_MIN),
        "duration_max_days": float(DURATION_FRAC_MAX),
        "duration_steps": int(DURATION_FRAC_STEPS),
        "duration_units": "absolute days, identical for every period",
        "objective": "astropy BoxLeastSquares default (log-likelihood), no flux errors supplied",
        "flux_preprocessing": f"rolling-median detrend (window {TREND_WINDOW}), outlier interpolation >{OUTLIER_THRESHOLD:g} sigma, z-score, clip +-{NORMALIZED_CLIP:g}",
        "top_k": len(candidates),
    }
    return BLSResult(candidates, spectrum, settings, sigma, cut)


def bls_at_period(time: np.ndarray, flux_rel: np.ndarray, period_days: float) -> dict:
    """BLS statistics at ONE period, using the same preprocessing and duration grid as the
    full periodogram (so a typed trial period is comparable with the spectrum)."""
    z = preprocess_light_curve(np.asarray(flux_rel, dtype=np.float64), trend_window=TREND_WINDOW)
    durations = np.linspace(DURATION_FRAC_MIN, DURATION_FRAC_MAX, DURATION_FRAC_STEPS)
    durations = durations[durations < period_days]
    if len(durations) == 0:
        raise ValueError(f"Period {period_days:g} d is shorter than the shortest BLS duration.")
    r = BoxLeastSquares(time, z).power(np.array([period_days]), durations)
    return {
        "period": float(period_days),
        "power": float(r.power[0]),
        "depth": float(r.depth[0]),
        "duration": float(r.duration[0]),
        "transit_time": float(r.transit_time[0]),
    }


def describe_fold(fold: dict, period_days: float, duration_days: float) -> dict:
    """Plain descriptors of the binned folded profile (no thresholds that claim detection)."""
    centers = np.asarray(fold["phase_bin_centers"], dtype=float)
    mean = np.asarray(fold["mean_flux"], dtype=float)
    valid = ~np.isnan(mean)
    if valid.sum() < 10:
        return {}
    phase = ((centers + 0.5) % 1.0) - 0.5           # centre the profile on phase 0
    order = np.argsort(phase)
    phase, mean = phase[order], mean[order]
    valid = ~np.isnan(mean)
    baseline = float(np.nanmedian(mean))
    i_min = int(np.nanargmin(mean))
    depth = baseline - float(mean[i_min])
    half = baseline - depth / 2.0
    below = (mean <= half) & valid
    bin_w = 1.0 / len(centers)
    # contiguous run of below-half-depth bins containing the minimum
    lo = hi = i_min
    while lo - 1 >= 0 and below[lo - 1]:
        lo -= 1
    while hi + 1 < len(mean) and below[hi + 1]:
        hi += 1
    width_phase = (hi - lo + 1) * bin_w
    outside = valid & (np.abs(phase - phase[i_min]) > max(width_phase, bin_w))
    if outside.sum() >= 5:
        o = mean[outside]
        scatter = 1.4826 * float(np.median(np.abs(o - np.median(o))))
    else:
        scatter = float("nan")
    return {
        "baseline": baseline,
        "depth_rel": depth,
        "phase_at_min": float(phase[i_min]),
        "dip_width_phase": float(width_phase),
        "bls_box_phase": float(duration_days / period_days),
        "out_of_dip_scatter": scatter,
        "depth_over_scatter": float(depth / scatter) if np.isfinite(scatter) and scatter > 0 else float("nan"),
    }


def build_view(
    time: np.ndarray,
    flux_rel: np.ndarray,
    period_days: float,
    transit_time: float,
    duration_days: float,
    power: float,
    bls_depth_sigma: float,
    source: str,
) -> PeriodView:
    view = PeriodView(source, float(period_days), float(transit_time), float(duration_days), float(power), float(bls_depth_sigma))
    try:
        view.fold = fold_light_curve(time, flux_rel, period_days, transit_time=transit_time, detrend=False)
        view.fold_summary = describe_fold(view.fold, period_days, duration_days)
    except Exception as exc:  # noqa: BLE001 - keep other stages alive
        log.exception("fold failed")
        view.errors["fold"] = f"{type(exc).__name__}: {exc}"
    try:
        view.rt = analyze_repeated_transits(time, flux_rel, period_days, transit_time=transit_time)
        view.repeatable = is_repeatable(view.rt, score_threshold=REPEATABLE_CUTOFF)
    except Exception as exc:  # noqa: BLE001
        log.exception("repeated-transit analysis failed")
        view.errors["repeated"] = f"{type(exc).__name__}: {exc}"
    return view


def view_from_candidate(time: np.ndarray, flux_rel: np.ndarray, cand: dict, source: str) -> PeriodView:
    return build_view(
        time, flux_rel, cand["period_days"], cand["transit_time"], cand["duration_days"],
        cand["power"], cand["depth"], source,
    )


def view_from_period(time: np.ndarray, flux_rel: np.ndarray, period_days: float, source: str = "trial") -> PeriodView:
    s = bls_at_period(time, flux_rel, period_days)
    return build_view(time, flux_rel, s["period"], s["transit_time"], s["duration"], s["power"], s["depth"], source)


def null_control(
    time: np.ndarray,
    flux_rel: np.ndarray,
    period_days: float,
    transit_time: float,
    n: int = SHUFFLE_CONTROLS,
    seed: int = 0,
) -> dict:
    """Repeatability score on flux randomly permuted in time (no real structure) at the
    same period and epoch. Shows what the metric gives for pure noise."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        r = analyze_repeated_transits(time, rng.permutation(flux_rel), period_days, transit_time=transit_time)
        if r["repeatability_score"] is not None:
            scores.append(float(r["repeatability_score"]))
    if not scores:
        return {"n": 0}
    a = np.asarray(scores)
    return {
        "n": int(len(a)), "min": float(a.min()), "median": float(np.median(a)),
        "mean": float(a.mean()), "max": float(a.max()),
        "pass_rate": float(np.mean(a >= REPEATABLE_CUTOFF)), "cutoff": REPEATABLE_CUTOFF,
    }


# ============================================================
# COMPOSITE (EXPERIMENTAL) AND CAUTIONS
# ============================================================

def compute_composite(cnn_score: float, bls_power: float, depth_rel: Optional[float], repeatability: Optional[float]) -> dict:
    """Composite candidate score, exactly as explainability.rank_candidate computes it,
    with the unused RF weight removed and the rest renormalised. Always experimental."""
    total = sum(COMPOSITE_BASE_WEIGHTS.values())
    weights = {k: w / total for k, w in COMPOSITE_BASE_WEIGHTS.items()}
    score, comps = rank_candidate(
        cnn_probability=cnn_score,
        rf_probability=None,
        bls_sde=bls_power,
        transit_depth=depth_rel,
        repeatability_score=repeatability,
        weights=weights,
    )
    raw = {"cnn": cnn_score, "bls": bls_power, "depth": depth_rel, "repeatability": repeatability}
    normalisation = {
        "cnn": "raw sigmoid output used as-is (not calibrated)",
        "bls": "min(BLS power / 10, 1)",
        "depth": "min(relative depth / 0.05, 1)",
        "repeatability": "score used as-is (0-1)",
    }
    rows = [
        {
            "component": k, "raw_input": raw[k], "normalisation": normalisation[k],
            "normalised": comps[k], "weight": weights[k], "contribution": weights[k] * comps[k],
        }
        for k in weights
    ]
    return {"score": float(score), "rows": rows, "weights": weights, "experimental": True}


def harmonic_relation(period: float, reference: float, tol: float = HARMONIC_TOLERANCE) -> Optional[str]:
    """Label such as '2x' or '0.5x' if `period` is a simple multiple/fraction of `reference`."""
    if reference <= 0:
        return None
    ratio = period / reference
    for m, label in [(1 / 3, "1/3x"), (0.5, "0.5x"), (2 / 3, "2/3x"), (1.0, "1x"), (1.5, "1.5x"),
                     (2.0, "2x"), (2.5, "2.5x"), (3.0, "3x"), (4.0, "4x"), (4.5, "4.5x"), (5.0, "5x")]:
        if abs(ratio - m) / m < tol:
            return label
    return None


def derive_cautions(result: AnalysisResult, view: Optional[PeriodView] = None) -> list[dict]:
    """Data-derived caution flags. Each: {code, level ('caution'|'info'), text}."""
    out: list[dict] = []
    q, seg = result.best_quarter, result.best_segment
    if q is None or seg is None:
        return out
    view = view or result.top_view

    if seg.n_points < REFERENCE_LENGTH:
        out.append({"code": "short_sequence", "level": "caution", "text": (
            f"The CNN scored {seg.n_points:,} points; it was trained on {REFERENCE_LENGTH:,}. "
            "Short-sequence performance was checked on only 7 validation planets.")})
    elif seg.n_points > REFERENCE_LENGTH:
        out.append({"code": "long_sequence", "level": "caution", "text": (
            f"The CNN scored {seg.n_points:,} points, longer than any length it was validated at "
            f"({', '.join(f'{n:,}' for n in VALIDATED_LENGTHS)}).")})

    n_scored = len(result.scored_segments)
    if n_scored > 1:
        out.append({"code": "selection_effect", "level": "caution", "text": (
            f"{n_scored} segments were scored and the highest score is shown, which biases the headline score upward.")})

    if result.bls and result.bls.candidates and view is not None:
        ref = result.bls.candidates[0]["period_days"]
        related = [c for c in result.bls.candidates[1:] if harmonic_relation(c["period_days"], ref)]
        if len(related) >= HARMONIC_LADDER_MIN_COUNT:
            out.append({"code": "harmonic_ladder", "level": "caution", "text": (
                f"{len(related)} of the other {len(result.bls.candidates) - 1} top peaks sit at simple multiples of the best period, "
                "a pattern typical of one repeating signal rather than independent detections.")})
        if view.period_days > 0 and view.duration_days / view.period_days > LONG_DURATION_FRACTION:
            out.append({"code": "long_duration", "level": "caution", "text": (
                f"The BLS box lasts {view.duration_days * 24:.1f} h, {100 * view.duration_days / view.period_days:.0f}% of the period. "
                "A dip covering a large part of the orbit is more typical of smooth variability than a transit.")})
        cut = result.bls.outlier_cut_depth
        out.append({"code": "deep_transit_blind", "level": "info", "text": (
            f"BLS preprocessing removes dips deeper than about {100 * cut:.2f}% for this star "
            f"({OUTLIER_THRESHOLD:g} sigma of its point scatter), so very deep transits would not be found.")})

    cov = seg.n_points / q.n_valid
    if cov < 0.6:
        out.append({"code": "partial_coverage", "level": "info", "text": (
            f"The CNN scored {seg.n_points:,} of {q.n_valid:,} observations ({cov:.0%}); BLS, the fold and the window analysis use all {q.n_valid:,}.")})
    if len(result.quarters) == 1:
        out.append({"code": "single_quarter", "level": "info", "text": (
            f"Only quarter {q.label} was analysed; a real planet signal should reappear in other quarters.")})
    return out


# ============================================================
# ORCHESTRATION
# ============================================================

def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def analyze_target(
    kic: int,
    quarter: Optional[int],
    max_products: int,
    model: Any,
    calibrator: Any = None,
    progress: Optional[Callable[[str], None]] = None,
) -> AnalysisResult:
    """Full pipeline for one target. Raises NoDataError / RetrievalError on search failure."""
    say = progress or (lambda _msg: None)
    result = AnalysisResult(
        kic=int(kic), requested_quarter=quarter, max_products=int(max_products),
        analyzed_at=_now_iso(), model_file=VARIABLE_MODEL_PATH.name,
    )

    say(f"Searching NASA/MAST for KIC {kic}")
    try:
        products = search_kepler(kic, quarter=quarter)
    except RuntimeError as exc:
        raise NoDataError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - network / service errors
        raise RetrievalError(f"{type(exc).__name__}: {exc}") from exc

    total = len(products)
    result.products_found = total
    n_scan = min(total, int(max_products))
    result.products_scanned = n_scan
    if total > n_scan:
        result.warnings.append(
            f"{total} quarters were found and only the first {n_scan} were scanned. "
            "Pick a specific quarter or raise 'Max quarters to scan' to cover more.")

    skipped: dict[str, list[str]] = {}

    def _skip(label: str, why: str) -> None:
        skipped.setdefault(why, []).append(label)

    for i in range(n_scan):
        product = products[i]
        q = extract_quarter(product)
        label = "Q?" if q is None else f"Q{q:02d}"
        say(f"Downloading and scoring {label} ({i + 1} of {n_scan})")
        try:
            lc = download_lightcurve(product)
            if lc is None:
                _skip(label, "download returned no light curve")
                continue
            flux_raw, flux_source = extract_flux(lc)
            qd = prepare_quarter(np.asarray(lc.time.value, dtype=np.float64), flux_raw, q, flux_source, model)
            result.quarters.append(qd)
        except Exception as exc:  # noqa: BLE001 - one bad quarter must not abort the scan
            log.exception("%s failed", label)
            _skip(label, f"{type(exc).__name__}: {exc}")

    for why, labels in skipped.items():
        result.warnings.append(f"Skipped {', '.join(labels)}: {why}")

    result.best_quarter_index, result.best_segment_index = select_best(result.quarters)
    if result.best_segment is None:
        return result

    q = result.best_quarter
    say("Running Box Least Squares period search")
    try:
        result.bls = run_bls(q.time, q.flux)
    except Exception as exc:  # noqa: BLE001
        log.exception("BLS failed")
        result.errors["periodicity"] = f"{type(exc).__name__}: {exc}"

    if result.bls and result.bls.candidates:
        say("Folding and checking repeated windows")
        top = best_period(result.bls.candidates)
        result.top_view = view_from_candidate(q.time, q.flux, top, "bls_best")
        result.errors.update({k: v for k, v in result.top_view.errors.items()})
        if result.top_view.rt is not None:
            try:
                result.null_control = null_control(q.time, q.flux, top["period_days"], top["transit_time"])
            except Exception as exc:  # noqa: BLE001
                result.errors["null_control"] = f"{type(exc).__name__}: {exc}"

    score = result.best_segment.planet_score
    if calibrator is None:
        result.errors["calibration"] = "Calibrator file not found (run src/calibration.py)."
    else:
        try:
            result.calibrated = float(calibrator.calibrate(score))
        except Exception as exc:  # noqa: BLE001
            result.errors["calibration"] = f"{type(exc).__name__}: {exc}"

    v = result.top_view
    if v and v.rt is not None and v.fold is not None:
        depth = v.fold.get("transit_depth")
        try:
            result.composite = compute_composite(
                score, result.bls.candidates[0]["power"],
                depth if depth is not None and np.isfinite(depth) else None,
                v.rt["repeatability_score"],
            )
        except Exception as exc:  # noqa: BLE001
            result.errors["composite"] = f"{type(exc).__name__}: {exc}"

    say("Done")
    return result
