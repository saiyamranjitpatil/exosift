"""'Explore' dialogs. Each one adds real inspection or recomputation, not just a bigger chart."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from exosift_pipeline import (
    AnalysisResult,
    PeriodView,
    SHUFFLE_CONTROLS,
    harmonic_relation,
    null_control,
)
from predict_real_star import MIN_INFERENCE_LENGTH
from ui import charts, export, insights, state
from ui.components import key_insight


def _status(result: AnalysisResult, q, seg) -> str:
    best = result.best_segment
    if best is not None and q is result.best_quarter and seg.index == best.index:
        return "CNN input (highest score)"
    if seg.planet_score is not None:
        return "Scored (lower score)"
    return f"Not scored: shorter than {MIN_INFERENCE_LENGTH:,} points"


# ------------------------------------------------------------------ light curve
@st.dialog("Light curve explorer", width="large")
def light_curve_dialog(result: AnalysisResult) -> None:
    quarters = result.quarters
    qi = result.best_quarter_index
    if len(quarters) > 1:
        qi = st.selectbox("Quarter", list(range(len(quarters))), index=qi, format_func=lambda i: quarters[i].label)
    q = quarters[qi]
    best = result.best_segment if qi == result.best_quarter_index else None

    st.plotly_chart(charts.plot_light_curve(q, best, height=400, uirevision=f"lcd-{qi}"),
                    config=charts.PLOT_CONFIG, key=f"dlg_lc_{qi}",
                    alt=f"Relative flux of {q.label} over time, split into contiguous runs separated by gaps")
    st.caption("Drag to zoom, double-click to reset. Grey bands are real gaps (more than 3x the median cadence).")

    scored_pts = sum(s.n_points for s in q.segments if s.planet_score is not None)
    st.markdown(
        f"**Why the CNN uses fewer points than the quarter contains.** {q.label} returned {q.n_rows:,} rows; "
        f"{q.n_removed} had non-finite or duplicate samples, leaving **{q.n_valid:,}**. The CNN needs one unbroken run "
        f"of at least {MIN_INFERENCE_LENGTH:,} points and never bridges a gap, so it scores each qualifying run on "
        f"its own ({scored_pts:,} points in total across {sum(s.planet_score is not None for s in q.segments)} run(s)). "
        "The dashboard headline uses the highest-scoring run."
    )
    rows = [{
        "Run": s.index, "Points": s.n_points, "BKJD start": s.t_start, "BKJD end": s.t_stop,
        "Days": s.span_days, "CNN score": s.planet_score, "Status": _status(result, q, s),
    } for s in q.segments]
    st.dataframe(pd.DataFrame(rows), hide_index=True, alt=f"Contiguous runs of {q.label}",
                 column_config={
                     "Points": st.column_config.NumberColumn(format="%d"),
                     "BKJD start": st.column_config.NumberColumn(format="%.2f"),
                     "BKJD end": st.column_config.NumberColumn(format="%.2f"),
                     "Days": st.column_config.NumberColumn(format="%.1f"),
                     "CNN score": st.column_config.NumberColumn(format="%.6f"),
                 })
    if q.gaps:
        st.dataframe(pd.DataFrame([{"Gap starts (BKJD)": g["t_start"], "Gap ends (BKJD)": g["t_stop"], "Length (days)": g["days"]}
                                   for g in q.gaps]), hide_index=True, alt=f"Gaps in {q.label}",
                     column_config={c: st.column_config.NumberColumn(format="%.2f") for c in
                                    ("Gap starts (BKJD)", "Gap ends (BKJD)", "Length (days)")})
    if len(result.scored_segments) > 1:
        st.info(
            f"{len(result.scored_segments)} runs were scored across the scanned quarters and the highest score is the "
            "headline. Choosing the maximum of several scores biases it upward; compare the table above.",
            icon=":material/info:",
        )


# ------------------------------------------------------------------ periodogram
@st.dialog("Period explorer", width="large")
def periodogram_dialog(result: AnalysisResult, view: PeriodView) -> None:
    bls = result.bls
    top = bls.candidates[0]
    st.markdown(
        "A Box Least Squares periodogram folds the light curve at thousands of trial periods and measures how well a "
        "flat-bottomed dip fits each one. **Higher power means a better fit, not a proven planet.** Real signals also "
        "produce peaks at multiples and fractions of their true period (harmonics), which is why several top peaks "
        "often line up."
    )
    rows = []
    for i, c in enumerate(bls.candidates):
        rows.append({
            "Rank": i + 1, "Period (d)": c["period_days"], "BLS power": c["power"],
            "Power vs best": c["power"] / top["power"], "Box duration (h)": c["duration_days"] * 24.0,
            "Duration / period": c["duration_days"] / c["period_days"],
            "Relation to best peak": harmonic_relation(c["period_days"], top["period_days"]) or "-",
            "Viewing": "yes" if abs(c["period_days"] - view.period_days) / c["period_days"] < 1e-6 else "",
        })
    st.markdown("**Top peaks.** Select a row to fold the light curve on that period.")
    table = st.dataframe(
        pd.DataFrame(rows), hide_index=True, on_select="rerun", selection_mode="single-row", key="peak_table",
        alt="Top Box Least Squares peaks with period, power, duration and harmonic relation",
        column_config={
            "Period (d)": st.column_config.NumberColumn(format="%.4f"),
            "BLS power": st.column_config.NumberColumn(format="%.1f"),
            "Power vs best": st.column_config.NumberColumn(format="%.2f"),
            "Box duration (h)": st.column_config.NumberColumn(format="%.2f"),
            "Duration / period": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    picked = table.selection.rows if table and table.selection else []
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Fold on selected peak", icon=":material/stacked_line_chart:", disabled=not picked, type="primary"):
            c = bls.candidates[picked[0]]
            if c is top:
                state.set_selection(result, "default", None)
            else:
                state.set_selection(result, "peak", c["period_days"])
            st.rerun()
    with c2:
        if st.button("Reset to strongest peak", icon=":material/restart_alt:"):
            state.set_selection(result, "default", None)
            st.rerun()

    st.divider()
    q = result.best_quarter
    st.markdown("**Try your own period.** BLS statistics are recomputed at exactly this period, then the fold and window analysis follow.")
    t1, t2 = st.columns([2, 1], vertical_alignment="bottom")
    trial = t1.number_input("Trial period (days)", min_value=0.25, max_value=float(max(1.0, q.span_days / 2)),
                            value=float(round(view.period_days, 4)), step=0.001, format="%.4f", key="trial_period")
    if t2.button("Fold on this period", icon=":material/calculate:"):
        state.set_selection(result, "trial", float(trial))
        st.rerun()

    st.divider()
    s = bls.settings
    st.markdown("**Search settings (identical for every period)**")
    st.dataframe(pd.DataFrame([
        {"Setting": "Period range", "Value": f"{s['period_min_days']:g} to {s['period_max_days']:g} days, {s['n_periods']:,} {s['period_spacing']} steps"},
        {"Setting": "Box duration grid", "Value": f"{s['duration_min_days'] * 24:.2f} to {s['duration_max_days'] * 24:.2f} hours in {s['duration_steps']} steps ({s['duration_units']})"},
        {"Setting": "Power", "Value": s["objective"]},
        {"Setting": "Flux preparation", "Value": s["flux_preprocessing"]},
    ]), hide_index=True, alt="Box Least Squares search settings")
    st.warning(
        f"BLS preprocessing removes dips deeper than about {100 * bls.outlier_cut_depth:.2f}% for this star "
        f"(7 sigma of its point scatter, {100 * bls.point_sigma:.3f}% per point), so very deep transits would be missed. "
        "No significance test is attached to these peaks.",
        icon=":material/warning:",
    )


# ------------------------------------------------------------------ phase fold
@st.dialog("Phase-fold explorer", width="large")
def fold_dialog(result: AnalysisResult, view: PeriodView) -> None:
    show_raw = st.toggle("Show individual observations", value=True, key="fold_dlg_raw")
    fig = charts.plot_folded(view, height=420, uirevision=f"fd-{view.period_days:.6f}")
    if not show_raw:
        fig.data[0].visible = "legendonly"
    st.plotly_chart(fig, config=charts.PLOT_CONFIG, key="fold_dlg_chart",
                    alt=f"Relative flux folded on {view.period_days:.4f} days with binned mean and BLS box")
    d = view.fold_summary or {}
    if d:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Folded depth", f"{100 * d['depth_rel']:.4f} %", help="Median of the binned profile minus its minimum, relative to the quarter median flux.")
        m2.metric("Dip / bin scatter", f"{d['depth_over_scatter']:.1f}x" if np.isfinite(d["depth_over_scatter"]) else "n/a",
                  help="Depth divided by the robust scatter of the out-of-dip binned means. Near or below about 3 is not visually conclusive.")
        m3.metric("Dip width", f"{100 * d['dip_width_phase']:.0f} % of phase", help="Width of the contiguous bins below half depth around the minimum.")
        m4.metric("BLS box", f"{100 * d['bls_box_phase']:.0f} % of phase", help="BLS box duration divided by the period.")
    key_insight(insights.fold_insight(view))
    st.markdown(
        "**How to read this.** Every observation is placed by its position within one orbit (phase) at the chosen "
        "period, then averaged in 200 bins. A real transit appears as a flat-bottomed dip at phase 0 that stays "
        "narrow and consistent. A smooth, wide, sinusoid-like dip points to stellar variability or a harmonic of the "
        "true period. Use the P/2 and 2P control in the main view to compare neighbouring periods. "
        "Fold depth is computed on flux that is not detrended, so slow drifts can blur it."
    )


# ------------------------------------------------------------------ transit-window evidence
@st.dialog("How the window evidence is calculated", width="large")
def evidence_dialog(result: AnalysisResult, view: PeriodView) -> None:
    rt = view.rt
    st.markdown(
        "**Experimental metric. It is not used in the outcome.** For the chosen period the code lays out expected "
        "epochs across the quarter and, around each one, takes a window of plus/minus 0.25 period."
    )
    st.markdown(
        "- **What counts as an event:** any window with at least 3 points, at least 2 of them in the central third. "
        "A window is counted whatever the flux does, so the number is about how many periods fit in the data, not how many transits were seen.\n"
        "- **Per window:** depth = local baseline (median outside the central third) minus the lowest central point. "
        "Because the lowest point is taken, this is positive even for pure noise.\n"
        "- **Score:** mean of three terms, each 1/(1+x), where x is the depth scatter, the duration scatter, and the timing "
        "scatter divided by the period. The timing term is close to 1 by construction because the minimum can only move inside the window.\n"
        "- **Cutoff:** 0.70 is a convention in the code, not a derived threshold."
    )
    if rt is None:
        st.warning("The window analysis could not be computed for this period.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Windows evaluated", f"{rt['n_transits']}")
    c2.metric("Score (experimental)", f"{rt['repeatability_score']:.3f}" if rt["repeatability_score"] is not None else "n/a")
    c3.metric("Depth / duration / timing scatter",
              " / ".join("n/a" if v is None else f"{v:.2f}" for v in (rt["depth_scatter"], rt["duration_scatter"],
                                                                      (None if rt["timing_scatter_days"] is None else rt["timing_scatter_days"] / rt["period_days"]))),
              help="Depth and duration scatter are relative; timing scatter is shown as a fraction of the period.")

    cache = st.session_state.setdefault("_null_cache", {})
    nkey = (state.token(result), round(view.period_days, 6))
    if view is result.top_view and result.null_control:
        null = result.null_control
    elif nkey in cache:
        null = cache[nkey]
    else:
        with st.spinner(f"Re-running the score on {SHUFFLE_CONTROLS} shuffled copies of the flux..."):
            q = result.best_quarter
            null = null_control(q.time, q.flux, view.period_days, view.transit_time)
        cache[nkey] = null
    if null and null.get("n"):
        st.markdown("**Shuffled-flux control.** The same analysis, at the same period and epoch, on flux randomly permuted in time (no real structure):")
        n1, n2, n3 = st.columns(3)
        n1.metric("Shuffled range", f"{null['min']:.3f} to {null['max']:.3f}")
        n2.metric("Shuffled copies above 0.70", f"{100 * null['pass_rate']:.0f} %")
        n3.metric("This star", f"{rt['repeatability_score']:.3f}" if rt["repeatability_score"] is not None else "n/a")
        key_insight(insights.evidence_insight(result, view, null))
    st.plotly_chart(charts.plot_window_depths(view), config=charts.PLOT_CONFIG, key="evidence_dlg_chart",
                    alt="Depth of each period-length window across the quarter")
    st.caption("Each point is one window. Grey points have signal-to-noise below 1.")
    st.download_button("Download windows (CSV)", export.build_windows_csv(view),
                       file_name=f"exosift_KIC{result.kic}_windows.csv", mime="text/csv",
                       icon=":material/download:", on_click="ignore")
