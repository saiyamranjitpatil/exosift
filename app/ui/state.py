"""Selected-period state shared by the periodogram, phase fold, evidence card, dialogs and exports."""

from __future__ import annotations

from typing import Optional

import numpy as np
import streamlit as st

from exosift_pipeline import AnalysisResult, PeriodView, build_view, view_from_period

MULTIPLIERS = {"P/2": 0.5, "P": 1.0, "2P": 2.0}
MAX_CACHED_VIEWS = 24


def token(result: AnalysisResult) -> str:
    return f"{result.kic}|{result.requested_quarter}|{result.analyzed_at}"


def get_sel(result: AnalysisResult) -> dict:
    """Current selection; resets to 'default' (the BLS best peak) whenever a new analysis is shown."""
    sel = st.session_state.get("sel")
    if not sel or sel.get("token") != token(result):
        sel = {"token": token(result), "kind": "default", "period": None}
        st.session_state["sel"] = sel
        st.session_state["pg_nonce"] = st.session_state.get("pg_nonce", 0) + 1
        st.session_state["fold_mult"] = "P"
    return sel


def set_selection(result: AnalysisResult, kind: str, period: Optional[float], bump_chart: bool = True) -> None:
    """kind: 'default' | 'peak' (a BLS grid period) | 'trial' (a typed period)."""
    st.session_state["sel"] = {"token": token(result), "kind": kind, "period": period}
    st.session_state["fold_mult"] = "P"
    if bump_chart:                                 # a new chart key clears the old plotly selection
        st.session_state["pg_nonce"] = st.session_state.get("pg_nonce", 0) + 1


def selected_period(result: AnalysisResult) -> float:
    sel = get_sel(result)
    return result.top_view.period_days if sel["kind"] == "default" else float(sel["period"])


def get_view(result: AnalysisResult, mult_label: str = "P") -> Optional[PeriodView]:
    """PeriodView for the selected period (x P/2, P or 2P). Cached per selection."""
    sel = get_sel(result)
    mult = MULTIPLIERS.get(mult_label, 1.0)
    if sel["kind"] == "default" and mult == 1.0:
        return result.top_view

    base = selected_period(result)
    key = (sel["token"], sel["kind"], round(base, 6), mult)
    cache = st.session_state.setdefault("_views", {})
    if key in cache:
        return cache[key]

    q = result.best_quarter
    try:
        if mult == 1.0 and sel["kind"] == "peak":
            s = result.bls.spectrum
            i = int(np.argmin(np.abs(s["period"] - base)))
            view = build_view(q.time, q.flux, float(s["period"][i]), float(s["transit_time"][i]),
                              float(s["duration"][i]), float(s["power"][i]), float(s["depth"][i]), "bls_peak")
        else:
            view = view_from_period(q.time, q.flux, base * mult, source="trial")
    except Exception as exc:  # noqa: BLE001 - invalid trial period etc.
        st.session_state["_view_error"] = f"{type(exc).__name__}: {exc}"
        return None
    cache[key] = view
    while len(cache) > MAX_CACHED_VIEWS:
        cache.pop(next(iter(cache)))
    return view


SOURCE_TEXT = {
    "bls_best": "BLS strongest peak",
    "bls_peak": "peak you selected",
    "trial": "trial period (BLS statistics recomputed at this period)",
}
