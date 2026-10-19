"""
ExoSift — Interactive Kepler Target Analysis (V2, Priority B6)

Streamlit UI for end-to-end exoplanet transit candidate analysis:

1. Look up a Kepler target (KIC ID) via NASA/MAST + Lightkurve
2. Run V1 CNN inference (planet score)
3. Run V2 periodicity detection (Box Least Squares)
4. Phase fold on detected period
5. Repeated-transit evidence
6. Confidence-calibrated probability

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

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
    REFERENCE_LENGTH,
    MIN_INFERENCE_LENGTH,
)
from periodicity import detect_periods, best_period
from phase_fold import fold_light_curve
from repeated_transit import analyze_repeated_transits, is_repeatable
from explainability import rank_candidate, CNNCalibrator, load_cnn_model, compute_saliency


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
    "(37 planet / 5,050 non-planet examples). Interpret results with caution."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Analysis Parameters")

kic_id = st.sidebar.number_input(
    "KIC ID",
    min_value=1,
    value=757076,
    help="Kepler Input Catalog identifier.",
)

quarter = st.sidebar.number_input(
    "Quarter (optional)",
    min_value=0,
    max_value=17,
    value=None,
    help="Leave blank to search all quarters.",
)

threshold = st.sidebar.slider(
    "Planet score threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(DEFAULT_THRESHOLD),
    step=0.001,
    help="CNN threshold for planet candidate classification.",
)

analyze_btn = st.sidebar.button("Analyze Target", type="primary")


# ============================================================
# HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def fetch_and_prepare(kic, quarter_val):
    """Search MAST, download, and prepare the best contiguous segment."""
    quarter_arg = int(quarter_val) if quarter_val is not None else None

    # Search.
    results = search_kepler(kic, quarter=quarter_arg)

    # Download each product and find the best contiguous run.
    best = None
    for product in results:
        lc = download_lightcurve(product)
        if lc is None:
            continue

        flux, flux_source = extract_flux(lc)
        time = np.asarray(lc.time.value, dtype=np.float64)

        time_clean, flux_clean = clean_time_flux(time, flux)
        runs = find_contiguous_runs(time_clean, flux_clean)

        if not runs:
            continue

        # Score each run, keep the best.
        for run in runs:
            result = score_segment(
                run["time"],
                run["flux"],
                threshold=threshold,
            )
            if best is None or result["planet_score"] > best["planet_score"]:
                best = {
                    **result,
                    "quarter": extract_quarter(product),
                    "flux_source": flux_source,
                    "time": time_clean,
                    "flux": flux_clean,
                    "run": run,
                }

    return best, results


def extract_quarter(product):
    import re
    match = re.search(r"Quarter\s*(\d+)", str(product), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def plot_light_curve(time, flux, title):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=time, y=flux, mode="lines", name="Flux", line=dict(width=1))
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time (days)",
        yaxis_title="Normalized Flux",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def plot_periodogram(candidates, title):
    if not candidates:
        return None
    periods = [c["period_days"] for c in candidates]
    powers = [c["power"] for c in candidates]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=periods, y=powers, name="BLS Power", marker_color="steelblue")
    )
    fig.update_layout(
        title=title,
        xaxis_title="Period (days)",
        yaxis_title="BLS Power (SDE)",
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def plot_folded(fold_result, title):
    valid = ~np.isnan(fold_result["mean_flux"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fold_result["phase_bin_centers"][valid],
            y=fold_result["mean_flux"][valid],
            mode="lines+markers",
            name="Mean flux",
            line=dict(width=2),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Orbital Phase",
        yaxis_title="Normalized Flux",
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


# ============================================================
# MAIN ANALYSIS
# ============================================================

if analyze_btn:
    with st.spinner("Searching NASA/MAST and downloading light curve..."):
        try:
            best, all_results = fetch_and_prepare(kic_id, quarter)
        except Exception as exc:
            st.error(f"Failed to fetch/prepare data: {exc}")
            best = None

    if best is None:
        st.error(
            "No usable Kepler data found for this target. "
            "Try a different KIC ID or quarter."
        )
    else:
        st.success(
            f"Found {best['real_cadences']} real observations "
            f"(Q{best['quarter']}, {best['flux_source']})"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Light Curve")
            fig_lc = plot_light_curve(
                best["time"],
                best["flux"],
                f"KIC {kic_id} — Q{best['quarter']} ({len(best['time'])} points)",
            )
            st.plotly_chart(fig_lc, use_container_width=True)

        with col2:
            st.subheader("V1 CNN Inference")
            st.metric(
                "Planet Score",
                f"{best['planet_score']:.6f}",
                delta=None,
            )
            prediction = (
                "🔭 PLANET CANDIDATE"
                if best["planet_score"] >= threshold
                else "○ Likely Non-Planet"
            )
            st.write(f"**Prediction:** {prediction}")
            st.write(f"**Threshold:** {threshold:.4f}")
            st.caption(
                "Raw CNN sigmoid output. Not a calibrated probability. "
                "See calibrated score below."
            )

        # ---- V2 Periodicity ----
        st.subheader("V2 — Periodicity Detection (Box Least Squares)")
        with st.spinner("Detecting transit periods..."):
            try:
                candidates = detect_periods(
                    best["time"],
                    best["flux"],
                    detrend=True,
                )
                best_period_cand = best_period(candidates)

                if best_period_cand:
                    col3, col4 = st.columns(2)
                    with col3:
                        fig_per = plot_periodogram(
                            candidates,
                            "BLS Periodogram — Top Periods",
                        )
                        st.plotly_chart(fig_per, use_container_width=True)

                    with col4:
                        st.metric(
                            "Best Period",
                            f"{best_period_cand['period_days']:.3f} days",
                        )
                        st.metric("BLS Power (SDE)", f"{best_period_cand['power']:.1f}")
                        st.metric("Transit Depth", f"{best_period_cand['depth']:.5f}")
                        st.metric(
                            "Duration",
                            f"{best_period_cand['duration_days']*24:.2f} hours",
                        )
                else:
                    st.info("No significant period detected.")
                    best_period_cand = None
            except Exception as exc:
                st.warning(f"Period detection failed: {exc}")
                best_period_cand = None
                candidates = []

        # ---- Phase Folding ----
        if best_period_cand:
            st.subheader("V2 — Phase Folding")
            with st.spinner("Folding light curve..."):
                try:
                    fold_result = fold_light_curve(
                        best["time"],
                        best["flux"],
                        best_period_cand["period_days"],
                        transit_time=best_period_cand.get("transit_time"),
                        detrend=False,
                    )
                    fig_fold = plot_folded(
                        fold_result,
                        f"Phase Folded (P={best_period_cand['period_days']:.3f} d)",
                    )
                    st.plotly_chart(fig_fold, use_container_width=True)
                    st.metric(
                        "Folded Transit Depth",
                        f"{fold_result['transit_depth']:.5f}"
                        if fold_result["transit_depth"]
                        else "N/A",
                    )
                except Exception as exc:
                    st.warning(f"Phase folding failed: {exc}")
                    fold_result = None

        # ---- Repeated Transit ----
        if best_period_cand:
            st.subheader("V2 — Repeated-Transit Evidence")
            with st.spinner("Analyzing transit consistency..."):
                try:
                    rt_result = analyze_repeated_transits(
                        best["time"],
                        best["flux"],
                        best_period_cand["period_days"],
                        transit_time=best_period_cand.get("transit_time"),
                    )
                    col5, col6, col7 = st.columns(3)
                    with col5:
                        st.metric("Transits Detected", rt_result["n_transits"])
                    with col6:
                        score = rt_result["repeatability_score"]
                        st.metric(
                            "Repeatability Score",
                            f"{score:.3f}" if score else "N/A",
                        )
                    with col7:
                        repeatable, reason = is_repeatable(rt_result)
                        st.write(f"**Repeatable:** {'✅' if repeatable else '❌'}")
                        st.caption(reason)
                except Exception as exc:
                    st.warning(f"Repeated-transit analysis failed: {exc}")

        # ---- Calibrated Probability ----
        st.subheader("V2 — Calibrated Probability")
        try:
            calibrator = CNNCalibrator(method="platt")
            calibrated_prob = calibrator.calibrate(best["planet_score"])
            col8, col9 = st.columns(2)
            with col8:
                st.metric("Raw CNN Score", f"{best['planet_score']:.6f}")
            with col9:
                st.metric(
                    "Calibrated Probability",
                    f"{calibrated_prob:.6f}",
                )
            st.caption(
                "Calibrated via Platt scaling on V1 validation data. "
                "More conservative than raw score due to extreme class imbalance."
            )
        except FileNotFoundError:
            st.info(
                "Calibrator not found. Run src/calibration.py to generate it."
            )

        # ---- Composite Ranking ----
        st.subheader("V2 — Composite Candidate Ranking")
        if best_period_cand:
            try:
                rt = analyze_repeated_transits(
                    best["time"],
                    best["flux"],
                    best_period_cand["period_days"],
                    transit_time=best_period_cand.get("transit_time"),
                )
                composite, components = rank_candidate(
                    cnn_probability=best["planet_score"],
                    rf_probability=None,
                    bls_sde=best_period_cand["power"],
                    transit_depth=fold_result["transit_depth"]
                    if fold_result
                    else best_period_cand["depth"],
                    repeatability_score=rt["repeatability_score"],
                )
                st.metric("Composite Score", f"{composite:.3f}")
                st.json(components)
            except Exception as exc:
                st.warning(f"Composite ranking failed: {exc}")

else:
    st.info(
        "Enter a KIC ID in the sidebar and click **Analyze Target** to begin. "
        "Example: KIC 757076"
    )

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
    """)

    st.markdown("""
    ### How to run
    ```bash
    streamlit run app/streamlit_app.py
    ```
    """)
