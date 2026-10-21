"""
ExoSift — Interactive Kepler Target Analysis (V2, Priority B6)

Streamlit UI for end-to-end exoplanet transit candidate analysis:

1. Look up a Kepler target (KIC ID) via NASA/MAST + Lightkurve
2. Run V1 CNN inference (planet score)
3. Run V2 periodicity detection (Box Least Squares)
4. Phase fold on detected period
5. Repeated-transit evidence
6. Confidence-calibrated probability
7. Composite candidate ranking

Design notes
------------
* The whole heavy pipeline lives in ONE cached function (`run_pipeline`).
  The UI only renders its result, so moving the threshold slider or any other
  widget re-renders instantly instead of re-downloading / re-running models.
* The request is kept in `st.session_state`, so results survive reruns
  (previously every widget interaction wiped the page).
* Each analysis stage is isolated: one failing stage never kills the others.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import inspect
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional

# Ensure src/ is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from predict_real_star import (
    search_kepler,
    download_lightcurve,
    extract_flux,
    clean_time_flux,
    find_contiguous_runs,
    score_segment,
    DEFAULT_THRESHOLD,
)
from periodicity import detect_periods, best_period
from phase_fold import fold_light_curve
from repeated_transit import analyze_repeated_transits, is_repeatable
from calibration import CNNCalibrator
from explainability import rank_candidate

log = logging.getLogger("exosift.app")

# ============================================================
# CONSTANTS
# ============================================================

MAX_PLOT_POINTS = 20_000          # decimate huge (e.g. short-cadence) curves for the browser
DEFAULT_MAX_PRODUCTS = 6          # cap downloads when "all quarters" is selected
N_PLANET, N_NON_PLANET = 37, 5_050
QUARTER_RE = re.compile(r"Quarter\s*(\d+)", flags=re.IGNORECASE)
# Streamlit >= ~1.50 deprecates use_container_width in favour of width="stretch".
_HAS_WIDTH_ARG = "width" in inspect.signature(st.plotly_chart).parameters


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ExoSift — Kepler Transit Analysis",
    page_icon="🔭",
    layout="wide",
)

st.title("🔭 ExoSift — Kepler Transit Candidate Analysis")
st.markdown(
    "AI-assisted exoplanet transit detection using NASA Kepler light curves. "
    "Enter a Kepler Input Catalog (KIC) ID to analyze a real star."
)
st.warning(
    "⚠️ **Scientific caution:** Outputs are machine-learning candidate scores, "
    "not confirmed exoplanet detections. Training data is extremely imbalanced "
    f"({N_PLANET} planet / {N_NON_PLANET:,} non-planet examples). "
    "Interpret results with caution."
)


# ============================================================
# SMALL HELPERS
# ============================================================

def is_finite(x: Any) -> bool:
    """True for real, finite numbers (None, NaN, inf and non-numerics -> False)."""
    try:
        return x is not None and bool(np.isfinite(x))
    except (TypeError, ValueError):
        return False


def fmt(x: Any, spec: str = ".3f", na: str = "N/A") -> str:
    """Format a number, or return `na`. Unlike `if x`, a legitimate 0.0 is kept."""
    return format(x, spec) if is_finite(x) else na


def show_plot(fig: go.Figure) -> None:
    if _HAS_WIDTH_ARG:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def decimate(x: np.ndarray, y: np.ndarray, n: int = MAX_PLOT_POINTS):
    if len(x) <= n:
        return x, y
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return x[idx], y[idx]


def extract_quarter(product: Any) -> Optional[int]:
    match = QUARTER_RE.search(str(product))
    return int(match.group(1)) if match else None


def run_stage(name: str, fn: Callable[[], Any], errors: dict) -> Any:
    """Run one pipeline stage; on failure record the error and return None."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - UI boundary, we want to keep going
        log.exception("Stage %r failed", name)
        errors[name] = f"{type(exc).__name__}: {exc}"
        return None


@st.cache_resource(show_spinner=False)
def get_calibrator() -> CNNCalibrator:
    """Load the calibrator once per server process (raises FileNotFoundError if absent)."""
    return CNNCalibrator(method="platt")


# ============================================================
# PIPELINE (cached, no UI calls inside)
# ============================================================

def _iter_runs(time: np.ndarray, flux: np.ndarray):
    """Yield (t, f) arrays for each contiguous run.

    `find_contiguous_runs` has changed shape between versions of predict_real_star
    (different number of arguments, different return types). This adapter accepts
    the plausible variants and raises a TypeError that names what it actually got.
    """
    try:
        params = inspect.signature(find_contiguous_runs).parameters.values()
        n_positional = sum(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params
        )
    except (TypeError, ValueError):
        n_positional = 2
    runs = find_contiguous_runs(time, flux) if n_positional >= 2 else find_contiguous_runs(time)

    for run in runs or []:
        if isinstance(run, dict) and "time" in run:
            yield np.asarray(run["time"]), np.asarray(run.get("flux", flux[: len(run["time"])]))
        elif isinstance(run, slice):
            yield time[run], flux[run]
        elif isinstance(run, (tuple, list)) and len(run) == 2 and all(isinstance(r, np.ndarray) and r.ndim == 1 and r.size > 2 for r in run):
            yield np.asarray(run[0]), np.asarray(run[1])             # (time, flux) arrays
        elif isinstance(run, (tuple, list)) and len(run) == 2 and all(isinstance(r, (int, np.integer)) for r in run):
            yield time[run[0]:run[1]], flux[run[0]:run[1]]            # (start, stop) indices
        elif isinstance(run, np.ndarray) and np.issubdtype(run.dtype, np.integer):
            yield time[run], flux[run]                               # index array
        else:
            raise TypeError(f"unrecognised find_contiguous_runs() item: {type(run).__name__}")


def _scan_products(kic: int, quarter: Optional[int], max_products: int, warnings: list):
    """Download products and return the best-scoring contiguous segment (or None)."""
    results = search_kepler(kic, quarter=quarter)
    total = len(results)
    if total == 0:
        return None

    if total > max_products:
        warnings.append(
            f"{total} products found; only the first {max_products} were scanned. "
            "Pick a specific quarter or raise the limit to scan more."
        )
    products = [results[i] for i in range(min(total, max_products))]

    best = None
    skipped: dict[str, list[str]] = {}      # reason -> quarters, so identical errors collapse to one note

    def _skip(q: Optional[int], exc: Exception) -> None:
        skipped.setdefault(f"{type(exc).__name__}: {exc}", []).append(f"Q{q}" if q is not None else "?")

    for product in products:
        q = extract_quarter(product)
        try:
            lc = download_lightcurve(product)
            if lc is None:
                continue

            flux, flux_source = extract_flux(lc)
            time = np.asarray(lc.time.value, dtype=np.float64)
            time_c, flux_c = clean_time_flux(time, flux)

            for seg_t, seg_f in _iter_runs(time_c, flux_c):
                try:
                    # Always score at the default threshold; the UI re-applies the
                    # user's threshold to the raw score, so the slider never re-runs this.
                    res = score_segment(seg_t, seg_f, threshold=DEFAULT_THRESHOLD)
                except ValueError as exc:   # e.g. segment shorter than the CNN minimum: skip just this segment
                    _skip(q, exc)
                    continue
                if best is None or res["planet_score"] > best["planet_score"]:
                    best = {
                        **res,
                        "quarter": q,
                        "flux_source": flux_source,
                        "time": time_c,
                        "flux": flux_c,
                        "segment_span": (float(seg_t[0]), float(seg_t[-1])),
                    }
        except Exception as exc:  # noqa: BLE001 - one bad quarter must not abort the scan
            log.exception("Quarter %s failed", q)
            _skip(q, exc)

    for reason, quarters in skipped.items():
        warnings.append(f"Skipped {', '.join(quarters)} — {reason}")
    return best


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def run_pipeline(kic: int, quarter: Optional[int], max_products: int) -> dict:
    """Run the full V1 + V2 pipeline. Pure function of its arguments -> safely cacheable."""
    out: dict = {"best": None, "warnings": [], "errors": {}}
    errors = out["errors"]

    best = _scan_products(kic, quarter, max_products, out["warnings"])
    out["best"] = best
    if best is None:
        return out

    time, flux = best["time"], best["flux"]

    # ---- V2: periodicity ----
    def _periodicity():
        cands = detect_periods(time, flux, detrend=True)
        return cands, (best_period(cands) if cands else None)

    periodicity = run_stage("periodicity", _periodicity, errors)
    candidates, top = periodicity if periodicity else ([], None)
    out["candidates"], out["top"] = candidates, top

    # ---- V2: phase fold + repeated transits (computed ONCE, reused by ranking) ----
    fold = rt = repeat_flag = None
    if top:
        period, t0 = top["period_days"], top.get("transit_time")
        fold = run_stage(
            "fold",
            lambda: fold_light_curve(time, flux, period, transit_time=t0, detrend=False),
            errors,
        )
        rt = run_stage(
            "repeated",
            lambda: analyze_repeated_transits(time, flux, period, transit_time=t0),
            errors,
        )
        if rt is not None:
            repeat_flag = run_stage("repeated", lambda: is_repeatable(rt), errors)
    out["fold"], out["rt"], out["repeatable"] = fold, rt, repeat_flag

    # ---- V2: calibration ----
    def _calibrate():
        try:
            return float(get_calibrator().calibrate(best["planet_score"]))
        except FileNotFoundError:
            errors["calibration"] = "Calibrator not found. Run src/calibration.py to generate it."
            return None

    out["calibrated"] = run_stage("calibration", _calibrate, errors)

    # ---- V2: composite ranking ----
    if top and rt is not None:
        def _rank():
            depth = fold["transit_depth"] if fold and is_finite(fold.get("transit_depth")) else top["depth"]
            score, components = rank_candidate(
                cnn_probability=best["planet_score"],
                rf_probability=None,
                bls_sde=top["power"],
                transit_depth=depth,
                repeatability_score=rt["repeatability_score"],
            )
            return float(score), components

        out["composite"] = run_stage("ranking", _rank, errors)
    else:
        out["composite"] = None

    return out


# ============================================================
# PLOTS
# ============================================================

def _layout(fig: go.Figure, title: str, x: str, y: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def plot_light_curve(time, flux, segment_span, title) -> go.Figure:
    x, y = decimate(np.asarray(time), np.asarray(flux))
    fig = go.Figure(go.Scattergl(x=x, y=y, mode="lines", line=dict(width=1)))
    if segment_span:
        fig.add_vrect(
            x0=segment_span[0], x1=segment_span[1],
            fillcolor="orange", opacity=0.12, line_width=0,
            annotation_text="scored segment", annotation_position="top left",
        )
    return _layout(fig, title, "Time (days)", "Normalized Flux", height=400)


def plot_periodogram(candidates, best_cand, title) -> go.Figure:
    periods = [c["period_days"] for c in candidates]
    powers = [c["power"] for c in candidates]

    # Stem plot: bars at arbitrary, widely spaced periods render as hairlines.
    sx: list = []
    sy: list = []
    for p, h in zip(periods, powers):
        sx += [p, p, None]
        sy += [0, h, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color="steelblue", width=2), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=periods, y=powers, mode="markers",
        marker=dict(size=9, color=["crimson" if c is best_cand else "steelblue" for c in candidates]),
        hovertemplate="P = %{x:.4f} d<br>power = %{y:.2f}<extra></extra>",
    ))
    return _layout(fig, title, "Period (days)", "BLS Power", height=350)


def plot_folded(fold_result, title) -> go.Figure:
    phase = np.asarray(fold_result["phase_bin_centers"])
    mean_flux = np.asarray(fold_result["mean_flux"])
    valid = ~np.isnan(mean_flux)
    fig = go.Figure(go.Scatter(x=phase[valid], y=mean_flux[valid], mode="lines+markers", line=dict(width=2)))
    return _layout(fig, title, "Orbital Phase", "Normalized Flux", height=350)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Analysis Parameters")

# Inputs that trigger the (expensive) pipeline live in a form: nothing runs until Analyze is clicked.
with st.sidebar.form("analysis_form"):
    kic_id = st.number_input("KIC ID", min_value=1, value=757076, step=1,
                             help="Kepler Input Catalog identifier.")
    quarter = st.number_input("Quarter (optional)", min_value=0, max_value=17, value=None, step=1,
                              help="Leave blank to search all quarters. Q0 is a valid Kepler quarter.")
    max_products = st.number_input("Max quarters to scan", min_value=1, max_value=20,
                                   value=DEFAULT_MAX_PRODUCTS, step=1,
                                   help="Each quarter is a separate download; lower is faster.")
    submitted = st.form_submit_button("Analyze Target", type="primary")

# The threshold only relabels an already computed score, so it stays live and outside the form.
threshold = st.sidebar.slider(
    "Planet score threshold",
    min_value=0.0, max_value=1.0, value=float(DEFAULT_THRESHOLD), step=0.001,
    help="CNN threshold for planet candidate classification. Changing it does not re-run the analysis.",
)

if submitted:
    st.session_state["request"] = (
        int(kic_id),
        None if quarter is None else int(quarter),
        int(max_products),
    )


# ============================================================
# RENDERING
# ============================================================

def render_v1(best: dict, kic: int) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Light Curve")
        show_plot(plot_light_curve(
            best["time"], best["flux"], best.get("segment_span"),
            f"KIC {kic} — Q{best['quarter']} ({len(best['time']):,} points)",
        ))
    with col2:
        st.subheader("V1 CNN Inference")
        st.metric("Planet Score", f"{best['planet_score']:.6f}")
        is_candidate = best["planet_score"] >= threshold
        st.write(f"**Prediction:** {'🔭 PLANET CANDIDATE' if is_candidate else '○ Likely Non-Planet'}")
        st.write(f"**Threshold:** {threshold:.4f}")
        st.caption("Raw CNN sigmoid output, not a calibrated probability. See calibrated score below.")


def render_periodicity(res: dict) -> None:
    st.subheader("V2 — Periodicity Detection (Box Least Squares)")
    if "periodicity" in res["errors"]:
        st.warning(f"Period detection failed: {res['errors']['periodicity']}")
        return
    top = res["top"]
    if not top:
        st.info("No significant period detected.")
        return

    col3, col4 = st.columns(2)
    with col3:
        show_plot(plot_periodogram(res["candidates"], top, "BLS Periodogram — Top Periods"))
    with col4:
        st.metric("Best Period", f"{top['period_days']:.3f} days")
        st.metric("BLS Power (SDE)", f"{top['power']:.1f}")
        st.metric("Transit Depth", f"{top['depth']:.5f}")
        st.metric("Duration", f"{top['duration_days'] * 24:.2f} hours")


def render_fold(res: dict) -> None:
    if not res.get("top"):
        return
    st.subheader("V2 — Phase Folding")
    if "fold" in res["errors"]:
        st.warning(f"Phase folding failed: {res['errors']['fold']}")
        return
    fold = res["fold"]
    show_plot(plot_folded(fold, f"Phase Folded (P={res['top']['period_days']:.3f} d)"))
    st.metric("Folded Transit Depth", fmt(fold.get("transit_depth"), ".5f"))


def render_repeated(res: dict) -> None:
    if not res.get("top"):
        return
    st.subheader("V2 — Repeated-Transit Evidence")
    if "repeated" in res["errors"]:
        st.warning(f"Repeated-transit analysis failed: {res['errors']['repeated']}")
        return
    rt = res["rt"]
    c5, c6, c7 = st.columns(3)
    c5.metric("Transits Detected", rt["n_transits"])
    c6.metric("Repeatability Score", fmt(rt["repeatability_score"]))   # 0.0 is a valid score
    with c7:
        if res["repeatable"]:
            repeatable, reason = res["repeatable"]
            st.write(f"**Repeatable:** {'✅' if repeatable else '❌'}")
            st.caption(reason)


def render_calibration(best: dict, res: dict) -> None:
    st.subheader("V2 — Calibrated Probability")
    if "calibration" in res["errors"]:
        st.info(res["errors"]["calibration"])
        return
    c8, c9 = st.columns(2)
    c8.metric("Raw CNN Score", f"{best['planet_score']:.6f}")
    c9.metric("Calibrated Probability", fmt(res["calibrated"], ".6f"))
    st.caption("Calibrated via Platt scaling on V1 validation data. "
               "More conservative than the raw score due to extreme class imbalance.")


def render_ranking(res: dict) -> None:
    st.subheader("V2 — Composite Candidate Ranking")
    if "ranking" in res["errors"]:
        st.warning(f"Composite ranking failed: {res['errors']['ranking']}")
        return
    if not res["composite"]:
        st.info("Composite ranking needs a detected period and repeated-transit analysis.")
        return
    score, components = res["composite"]
    st.metric("Composite Score", f"{score:.3f}")
    rows = [
        {"Component": k, "Value": fmt(v, ".4f", na=str(v)) if isinstance(v, (int, float, np.number)) else str(v)}
        for k, v in dict(components).items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def render_landing() -> None:
    st.info("Enter a KIC ID in the sidebar and click **Analyze Target** to begin. Example: KIC 757076")
    st.markdown("""
### V2 Pipeline

| Step | Component | Description |
|------|-----------|-------------|
| 1 | NASA/MAST + Lightkurve | Fetch real Kepler observations |
| 2 | V1 CNN | Planet score from 1D CNN |
| 3 | BLS Periodogram | Detect transit period |
| 4 | Phase Folding | Reveal average transit shape |
| 5 | Repeated Transit | Check signal consistency |
| 6 | Calibration | Platt-scaled probability |
| 7 | Ranking | Composite candidate score |

### How to run
```bash
streamlit run app/streamlit_app.py
```
""")


# ============================================================
# MAIN
# ============================================================

request = st.session_state.get("request")

if request is None:
    render_landing()
else:
    kic, q, n_products = request
    with st.spinner(f"Searching NASA/MAST and analysing KIC {kic} (first run can take a minute)..."):
        try:
            result = run_pipeline(kic, q, n_products)
        except Exception as exc:  # noqa: BLE001 - network / MAST failures
            log.exception("Pipeline failed")
            st.error(f"Failed to fetch or prepare data: {type(exc).__name__}: {exc}")
            st.stop()

    best = result["best"]
    notes = result["warnings"]

    if best is None:
        run_pipeline.clear()   # never keep a failed result cached; the next click retries from scratch
        st.error("No usable Kepler data found for this target. Try a different KIC ID or quarter.")
        if notes:
            st.markdown("**Why nothing was usable:**")
            for w in notes:
                st.markdown(f"- {w}")
        st.stop()

    if notes:
        with st.expander(f"Scan notes ({len(notes)})"):
            for w in notes:
                st.markdown(f"- {w}")

    st.success(f"Found {best['real_cadences']} real observations (Q{best['quarter']}, {best['flux_source']})")

    render_v1(best, kic)
    render_periodicity(result)
    render_fold(result)
    render_repeated(result)
    render_calibration(best, result)
    render_ranking(result)