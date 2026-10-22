"""Dashboard sections for the Analysis page."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from exosift_pipeline import AnalysisResult, PeriodView
from predict_real_star import DEFAULT_THRESHOLD, MODEL_PATH
from ui import charts, components, dialogs, export, insights, runtime, state

DEFAULT_KIC = 757076
EXAMPLE = (757076, 2, 2)
CAL_VALIDATION_CSV = MODEL_PATH.parent / "cnn_validation_predictions.csv"


# ------------------------------------------------------------------ sidebar
def _quarter_label(q: Optional[int]) -> str:
    return "Auto: scan quarters" if q is None else f"Q{q:02d}"


def sidebar_controls() -> float:
    """Target form, advanced options and the (label-only) threshold. Returns the threshold."""
    last = st.session_state.get("request")
    with st.sidebar:
        st.subheader("Analyze a target")
        st.caption("Retrieve and analyze Kepler light curves for potential exoplanet transits.")
        with st.form("target_form", border=False):
            kic = st.number_input(
                "KIC ID", min_value=1, max_value=999_999_999, value=last[0] if last else DEFAULT_KIC, step=1, format="%d",
                help="Kepler Input Catalog number of the star, for example 757076.",
            )
            options = [None] + list(range(0, 18))
            quarter = st.selectbox(
                "Quarter (optional)", options, index=options.index(last[1]) if last and last[1] in options else 0,
                format_func=_quarter_label,
                help="Kepler observed in quarters Q0 to Q17. Auto scans the available quarters (up to the limit in Advanced options).",
            )
            submitted = st.form_submit_button("Analyze target", type="primary", icon=":material/arrow_forward:", width="stretch")

        with st.expander("Advanced options", icon=":material/tune:"):
            st.number_input(
                "Max quarters to scan", min_value=1, max_value=20, value=2, step=1, key="max_products", persist_state="session",
                help="Applies when Quarter is Auto. Each quarter is a separate download, so fewer is faster.",
            )
            st.slider(
                "Screening threshold", min_value=0.0, max_value=1.0, value=float(DEFAULT_THRESHOLD), step=0.0001, format="%.4f",
                key="threshold", persist_state="session",
                help="Only relabels the CNN score. It does not re-run the model.",
            )
            st.caption(
                f"The default {DEFAULT_THRESHOLD} was chosen on validation data that contains only 7 planets. "
                "Changing it never re-runs the analysis."
            )
            st.button("Reset threshold", icon=":material/restart_alt:", type="tertiary", key="reset_threshold",
                      on_click=lambda: st.session_state.update(threshold=float(DEFAULT_THRESHOLD)))

        st.info(
            "Real Kepler observations only: no synthetic data and no resampling across gaps. "
            "Outputs are preliminary screening scores, not confirmed discoveries.",
            icon=":material/info:",
        )

    if submitted:
        request = (int(kic), quarter, int(st.session_state.get("max_products", 2)))
        runtime.clear_cached(request)          # an explicit Analyze always fetches fresh data
        st.session_state["request"] = request
        st.session_state.pop("sel", None)
    return float(st.session_state.get("threshold", DEFAULT_THRESHOLD))


# ------------------------------------------------------------------ landing
def render_landing() -> None:
    components.hero("Kepler target analysis", [("No target analyzed yet", False)], status="Preliminary screening")
    st.write("")
    c1, c2, c3 = st.columns(3)
    steps = [
        (c1, ":material/search:", "1. Choose a target", "Enter a Kepler Input Catalog (KIC) number in the sidebar, optionally with a quarter."),
        (c2, ":material/play_arrow:", "2. Analyze", "ExoSift downloads the light curve from NASA/MAST and scores it with a 1D CNN and a Box Least Squares period search."),
        (c3, ":material/monitoring:", "3. Inspect the evidence", "Review the light curve, the periodogram and the phase fold, and read what is and is not established."),
    ]
    for col, icon, title, text in steps:
        with col, st.container(border=True):
            st.markdown(f"**{icon} {title}**")
            st.caption(text)
    st.write("")
    with st.container(border=True):
        st.markdown("**Not sure where to start?**")
        st.caption("KIC 757076, Quarter 2 is the worked example used throughout the documentation.")
        if st.button("Analyze KIC 757076 (Q02)", icon=":material/rocket_launch:", type="primary"):
            runtime.clear_cached(EXAMPLE)
            st.session_state["request"] = EXAMPLE
            st.session_state.pop("sel", None)
            st.rerun()


def render_insufficient(result: AnalysisResult, threshold: float) -> None:
    out = insights.outcome(result, threshold)
    components.hero("Kepler target analysis", [(f"KIC {result.kic}", True), (_quarter_label(result.requested_quarter), False)],
                    status=out["status"], analyzed_at=_fmt_time(result.analyzed_at))
    st.write("")
    components.outcome_banner(out["tone"], out["title"], out["summary"])
    for w in result.warnings:
        st.warning(w, icon=":material/warning:")
    if result.quarters:
        rows = [{"Quarter": q.label, "Valid points": q.n_valid,
                 "Longest run": max(s.n_points for s in q.segments), "Runs": len(q.segments)} for q in result.quarters]
        st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Retrieved quarters and their longest contiguous runs")


# ------------------------------------------------------------------ header and KPIs
def _fmt_time(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%b %d, %Y, %H:%M")
    except ValueError:
        return iso


def render_header(result: AnalysisResult, threshold: float) -> None:
    out = insights.outcome(result, threshold)
    q = result.best_quarter
    chips = [(f"KIC {result.kic}", True), (f"Quarter {q.label[1:]}", False), ("NASA/MAST", False), (f"{q.flux_source.upper().replace('_FLUX', '')} flux", False)]
    left, right = st.columns([6, 1.25], vertical_alignment="top")
    with left:
        components.hero("Kepler target analysis", chips, status=out["status"], analyzed_at=_fmt_time(result.analyzed_at))
    with right:
        components.source_popover()
    st.write("")
    components.outcome_banner(out["tone"], out["title"], out["summary"])


def render_kpis(result: AnalysisResult, threshold: float) -> None:
    q, seg = result.best_quarter, result.best_segment
    top = result.bls.candidates[0] if result.bls and result.bls.candidates else None
    cols = st.columns(4)
    with cols[0], st.container(border=True):
        st.metric("CNN screening score", f"{seg.planet_score:.4f}",
                  delta=f"{'Above' if seg.planet_score >= threshold else 'Below'} threshold {threshold:.4f}",
                  delta_color="off", delta_arrow="off", help=insights.HELP["cnn_score"])
    with cols[1], st.container(border=True):
        if top:
            st.metric("Best BLS trial period", f"{top['period_days']:.3f} days",
                      delta=f"about {top['period_days'] * 24:.1f} hours, power {top['power']:.1f}",
                      delta_color="off", delta_arrow="off", help=insights.HELP["bls_period"])
        else:
            st.metric("Best BLS trial period", "n/a", help=insights.HELP["bls_period"])
    with cols[2], st.container(border=True):
        st.metric("Input coverage", f"{seg.n_points:,} of {q.n_valid:,}", delta=f"{seg.n_points / q.n_valid:.0%} of quarter observations",
                  delta_color="off", delta_arrow="off", help=insights.HELP["coverage"])
    with cols[3], st.container(border=True):
        value, sub = insights.data_quality_text(result)
        st.metric("Data quality", value, delta=sub, delta_color="off", delta_arrow="off", help=insights.HELP["quality"])


# ------------------------------------------------------------------ chart cards
def _card_header(title: str, help_text: str, button_label: str, key: str) -> bool:
    left, right = st.columns([3.2, 1.4], vertical_alignment="center")
    with left:
        st.subheader(title, help=help_text)
    with right:
        return st.button(button_label, key=key, type="tertiary", icon=":material/open_in_full:", width="stretch")


def render_lightcurve(result: AnalysisResult) -> None:
    q, seg = result.best_quarter, result.best_segment
    with st.container(border=True):
        if _card_header("Stellar light curve", "Relative flux (flux divided by the quarter median) over time. "
                        "Cyan is the segment the CNN scored highest; grey runs were too short to score.",
                        "Explore light curve", "open_lc"):
            dialogs.light_curve_dialog(result)
        st.plotly_chart(charts.plot_light_curve(q, seg, uirevision=state.token(result)), config=charts.PLOT_CONFIG,
                        key="lc_chart", alt=f"Relative flux of {q.label} over {q.span_days:.0f} days with the CNN input segment highlighted")
        components.key_insight(insights.lightcurve_insight(result))


def render_periodogram(result: AnalysisResult) -> None:
    top = result.bls.candidates[0]
    sel_period = state.selected_period(result)
    with st.container(border=True):
        view_now = state.get_view(result, st.session_state.get("fold_mult", "P")) or result.top_view
        if _card_header("BLS periodogram", "Box Least Squares power versus trial period. Click a peak to fold the light "
                        "curve on it. Peak height ranks periods; it is not a significance test.",
                        "Explore periodogram", "open_pg"):
            dialogs.periodogram_dialog(result, view_now)
        log_x = st.toggle("Log period axis", value=True, key="pg_log", persist_state="session")
        fig = charts.plot_periodogram(result.bls, sel_period if state.get_sel(result)["kind"] != "default" else top["period_days"],
                                      log_x=log_x, uirevision=f"{state.token(result)}-{log_x}")
        event = st.plotly_chart(fig, config=charts.PLOT_CONFIG, key=f"pg_chart_{st.session_state.get('pg_nonce', 0)}",
                                on_select="rerun", selection_mode="points",
                                alt="BLS power versus trial period; select a peak to fold on that period")
        try:
            pts = event.selection.points
        except (AttributeError, KeyError):
            pts = []
        if pts:
            spectrum = result.bls.spectrum["period"]
            x = float(pts[0]["x"])
            idx = int(np.argmin(np.abs(np.log(spectrum) - np.log(x))))
            period = float(spectrum[idx])
            if abs(period - top["period_days"]) / top["period_days"] < 1e-9:
                new = ("default", None)
            else:
                new = ("peak", period)
            cur = state.get_sel(result)
            if (cur["kind"], cur["period"]) != new:
                state.set_selection(result, new[0], new[1], bump_chart=False)
                st.rerun()
        st.caption("Click a peak or any point on the curve to fold the light curve on that period.")
        if state.get_sel(result)["kind"] != "default":
            if st.button("Back to strongest peak", icon=":material/restart_alt:", type="tertiary", key="pg_reset"):
                state.set_selection(result, "default", None)
                st.rerun()
        components.key_insight(insights.periodogram_insight(result, view_now))


def render_fold(result: AnalysisResult) -> Optional[PeriodView]:
    with st.container(border=True):
        left, right = st.columns([3.2, 1.4], vertical_alignment="center")
        with left:
            st.subheader("Phase-folded light curve", help="Every observation placed by its position within one orbit at "
                          "the chosen period, then averaged in 200 bins. A transit would appear as a narrow dip at phase 0.")
        st.session_state.setdefault("fold_mult", "P")
        mult = st.segmented_control("Fold period", list(state.MULTIPLIERS), key="fold_mult",
                                    label_visibility="collapsed", help="Fold on half, one or two times the selected period.")
        view = state.get_view(result, mult or "P")
        with right:
            if view is not None and view.fold is not None and st.button("Explore phase fold", key="open_fold", type="tertiary",
                                                                         icon=":material/open_in_full:", width="stretch"):
                dialogs.fold_dialog(result, view)
        if view is None:
            st.warning(f"Could not compute this period: {st.session_state.get('_view_error', 'unknown error')}", icon=":material/warning:")
            return None
        if view.fold is None:
            st.warning(f"Phase folding failed: {view.errors.get('fold', 'unknown error')}", icon=":material/warning:")
            return view
        st.caption(f"P = {view.period_days:.4f} d ({state.SOURCE_TEXT.get(view.source, view.source)}); "
                   f"box duration {view.duration_days * 24:.2f} h")
        st.plotly_chart(charts.plot_folded(view, uirevision=f"{state.token(result)}-{view.period_days:.6f}"),
                        config=charts.PLOT_CONFIG, key="fold_chart",
                        alt=f"Relative flux folded on {view.period_days:.4f} days")
        components.key_insight(insights.fold_insight(view))
    return view


def render_evidence(result: AnalysisResult, view: Optional[PeriodView]) -> None:
    with st.container(border=True):
        left, right = st.columns([3.2, 1.4], vertical_alignment="center")
        with left:
            st.subheader("Transit-window evidence", help="Experimental. Counts period-length windows and scores how "
                          "consistent their depth, duration and timing are. It is not used in the outcome.")
            st.badge("Experimental", color="orange", icon=":material/science:")
        with right:
            if view is not None and view.rt is not None and st.button("How is this calculated?", key="open_ev", type="tertiary",
                                                                      icon=":material/help:", width="stretch"):
                dialogs.evidence_dialog(result, view)
        if view is None or view.rt is None:
            st.info("The window analysis is not available for this period.", icon=":material/info:")
            return
        rt = view.rt
        c1, c2 = st.columns(2)
        c1.metric("Period windows evaluated", rt["n_transits"],
                  help="Windows of plus/minus 0.25 period around each expected epoch that have enough points. A window is "
                       "counted whatever the flux does, so this is not a count of detected transits.")
        c2.metric("Window consistency score", "n/a" if rt["repeatability_score"] is None else f"{rt['repeatability_score']:.3f}",
                  help="Mean of three consistency terms (depth, duration, timing). Experimental: shuffled-flux controls score in the same range.")
        null = result.null_control if view is result.top_view else None
        st.caption("Not used in the outcome. Definition needs validation.")
        components.key_insight(insights.evidence_insight(result, view, null) if null or view is result.top_view else
                               f"{rt['n_transits']} period-length windows were evaluated. Open 'How is this calculated?' to compare "
                               "the score with shuffled-flux controls.")


# ------------------------------------------------------------------ interpretation panel
def render_interpretation(result: AnalysisResult, threshold: float, view: Optional[PeriodView]) -> None:
    info = insights.interpretation(result, threshold, view)
    with st.container(border=True):
        st.subheader("Interpretation")
        with st.container(border=True):
            st.markdown("**:blue[:material/task_alt:] What supports this result**")
            for s in info["supports"]:
                st.markdown(f"- {s}")
        with st.container(border=True):
            st.markdown("**:orange[:material/warning:] What remains uncertain**")
            for s in info["uncertain"]:
                st.markdown(f"- {s}")
        with st.container(border=True):
            st.markdown("**:violet[:material/lightbulb:] Recommended next step**")
            st.markdown(info["next_step"])
        notes = [c["text"] for c in info["cautions"] if c["level"] == "info"]
        if notes:
            with st.expander("Context notes", icon=":material/notes:"):
                for n in notes:
                    st.markdown(f"- {n}")


# ------------------------------------------------------------------ advanced diagnostics
@st.cache_data(show_spinner=False)
def _calibration_facts() -> Optional[dict]:
    if not CAL_VALIDATION_CSV.exists():
        return None
    df = pd.read_csv(CAL_VALIDATION_CSV)
    return {"n": int(len(df)), "positives": int((df["actual_label"] == 1).sum())}


def render_diagnostics(result: AnalysisResult, view: Optional[PeriodView]) -> None:
    seg = result.best_segment
    box = st.expander("Advanced diagnostics", icon=":material/tune:", on_change="rerun", key="adv_open")
    if not box.open:
        return
    with box:
        st.caption("Experimental and technical detail. Nothing here feeds the outcome above.")
        tab_cal, tab_comp, tab_seg, tab_bls, tab_win = st.tabs(["Calibration", "Composite score", "Segments and gaps", "BLS settings", "Window events"])

        with tab_cal:
            st.badge("Experimental", color="orange", icon=":material/science:")
            if result.calibrated is None:
                st.info(result.errors.get("calibration", "Calibration is not available."), icon=":material/info:")
            else:
                c1, c2 = st.columns(2)
                c1.metric("Raw CNN score", f"{seg.planet_score:.6f}")
                c2.metric("Platt-mapped value", f"{result.calibrated:.6f}")
                cal = runtime.get_calibrator()
                if cal is not None:
                    xs = np.linspace(0.0, 1.0, 201)
                    st.plotly_chart(charts.plot_calibration_curve(xs, cal.calibrate(xs), seg.planet_score, result.calibrated),
                                    config=charts.PLOT_CONFIG, key="cal_chart", alt="Platt mapping from raw CNN score to calibrated value")
            facts = _calibration_facts()
            n_txt = f"{facts['n']:,} validation samples, only {facts['positives']} of them planets" if facts else "the validation set"
            st.markdown(
                f"**What this number is.** A Platt (logistic) mapping fitted on {n_txt}. It was fitted to the outputs of the "
                "fixed 3,197-point CNN and is applied here to the variable-length CNN at a different input length, and its fit "
                "was evaluated on the same data it was fitted on. Its output stays low because the validation base rate is "
                "under 1%. **It is not the probability that this star hosts a planet.**"
            )

        with tab_comp:
            st.badge("Experimental: methodology under review", color="orange", icon=":material/science:")
            comp = result.composite
            if not comp:
                st.info("The composite needs a period and a window analysis.", icon=":material/info:")
            else:
                st.metric("Composite score", f"{comp['score']:.3f}", help="Weighted sum of the normalised components below.")
                rows = [{"Component": r["component"], "Raw input": r["raw_input"], "Normalisation": r["normalisation"],
                         "Normalised": r["normalised"], "Weight": r["weight"], "Contribution": r["contribution"]} for r in comp["rows"]]
                st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Composite score components", column_config={
                    "Raw input": st.column_config.NumberColumn(format="%.5f"),
                    "Normalised": st.column_config.NumberColumn(format="%.4f"),
                    "Weight": st.column_config.NumberColumn(format="%.4f"),
                    "Contribution": st.column_config.NumberColumn(format="%.4f"),
                })
                st.markdown("**Formula.** Composite = sum of weight x normalised value. The RF term has no score for a live target, "
                            "so its 0.25 weight is removed and the rest are renormalised to sum to 1.")
                top = result.bls.candidates[0]
                issues = []
                if top["power"] >= 10:
                    issues.append(f"BLS power {top['power']:.1f} exceeds the arbitrary cap of 10, so that term is pinned at 1.0 and cannot discriminate.")
                if result.null_control and comp["rows"][-1]["raw_input"] is not None and comp["rows"][-1]["raw_input"] <= result.null_control["max"] + 0.05:
                    issues.append("The repeatability term scores in the same range on shuffled flux, so it carries little information.")
                issues.append("The depth term divides by an arbitrary 5% cap, which penalises small planets.")
                issues.append("The CNN term is a raw sigmoid output, not a calibrated probability.")
                for i in issues:
                    st.markdown(f"- {i}")

        with tab_seg:
            seg_csv = export.build_segments_csv(result)
            st.dataframe(pd.read_csv(io.StringIO(seg_csv)), hide_index=True, alt="Contiguous runs and CNN scores")
            st.download_button("Download segments (CSV)", seg_csv, file_name=f"exosift_KIC{result.kic}_segments.csv",
                               mime="text/csv", icon=":material/download:", on_click="ignore", key="dl_seg")

        with tab_bls:
            st.json(result.bls.settings if result.bls else {})
            if result.bls:
                st.download_button("Download periodogram (CSV)", export.build_periodogram_csv(result),
                                   file_name=f"exosift_KIC{result.kic}_periodogram.csv", mime="text/csv",
                                   icon=":material/download:", on_click="ignore", key="dl_pg")

        with tab_win:
            if view is None or view.rt is None:
                st.info("No window events for this period.", icon=":material/info:")
            else:
                st.dataframe(pd.DataFrame(view.rt["events"]), hide_index=True, alt="Per-window depth, duration and signal-to-noise",
                             column_config={"depth": st.column_config.NumberColumn(format="%.6f")})


# ------------------------------------------------------------------ footer and exports
def render_footer(result: AnalysisResult, threshold: float, view: Optional[PeriodView]) -> None:
    q, seg = result.best_quarter, result.best_segment
    out = insights.outcome(result, threshold)
    with st.container(border=True):
        left, right = st.columns([3.6, 2.4], vertical_alignment="center")
        with left:
            components.detail_strip([
                ("Target", f"KIC {result.kic}"), ("Quarter", q.label), ("Quarter observations", f"{q.n_valid:,}"),
                ("CNN input", f"{seg.n_points:,}"), ("Data source", "NASA/MAST (Lightkurve)"),
                ("Model", result.model_file), ("Analyzed", _fmt_time(result.analyzed_at)), ("Status", out["status"]),
            ])
        with right:
            base = f"exosift_KIC{result.kic}_{q.label}"
            with st.container(horizontal=True, horizontal_alignment="right"):
                st.download_button("PDF report", lambda: export.build_pdf(result, view, threshold), file_name=f"{base}.pdf",
                                   mime="application/pdf", icon=":material/picture_as_pdf:", on_click="ignore", width="content",
                                   help="Summary, key values, cautions and the three main charts.")
                st.download_button("JSON", lambda: export.build_json(result, view, threshold), file_name=f"{base}.json",
                                   mime="application/json", icon=":material/data_object:", on_click="ignore", width="content",
                                   help="Every displayed value with definitions and warnings.")
                st.download_button("CSV data", lambda: export.build_light_curve_csv(result), file_name=f"{base}_lightcurve.csv",
                                   mime="text/csv", icon=":material/download:", on_click="ignore", width="content",
                                   help="Time, relative flux, run index and whether each point was in the CNN input.")
