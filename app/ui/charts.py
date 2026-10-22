"""Plotly figure builders for the dashboard (no Streamlit imports)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from exosift_pipeline import BLSResult, PeriodView, QuarterData, Segment
from predict_real_star import MIN_INFERENCE_LENGTH

CYAN = "#38BDF8"
AMBER = "#F5A524"
SLATE = "#64748B"
MUTED_BLUE = "#5E7DB5"
RED = "#F87171"
GRID = "rgba(148,163,184,0.14)"
AXIS = "rgba(148,163,184,0.35)"

MAX_PLOT_POINTS = 20_000

PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def _layout(fig: go.Figure, x_title: str, y_title: str, height: int, uirevision: str = "", margin_b: int = 8) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=12, b=margin_b),
        xaxis=dict(title=x_title, gridcolor=GRID, zeroline=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS),
        yaxis=dict(title=y_title, gridcolor=GRID, zeroline=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS),
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0, font=dict(size=11)),
        dragmode="zoom",
        uirevision=uirevision or None,
    )
    return fig


# Log-axis tick labels: Plotly's automatic log ticks show bare mantissas ("5" for both 0.5
# and 5), which is ambiguous. Pin explicit, unambiguous labels over the BLS search range.
_LOG_PERIOD_TICKS = [0.5, 1, 2, 3, 5, 7, 10, 20, 30, 50]


def _decimate(x: np.ndarray, y: np.ndarray, n: int = MAX_PLOT_POINTS):
    if len(x) <= n:
        return x, y
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return x[idx], y[idx]


def plot_light_curve(
    q: QuarterData,
    best: Optional[Segment],
    height: int = 380,
    uirevision: str = "",
) -> go.Figure:
    """Full quarter in three tones so 'observations in the quarter' vs 'observations the CNN
    used' is visible: highest-scoring segment, other scored segments, unscored runs."""
    fig = go.Figure()
    shown: set[str] = set()
    for seg in q.segments:
        if best is not None and seg.index == best.index:
            name, color, width = f"CNN input: highest-scoring segment ({seg.n_points:,} points)", CYAN, 1.5
        elif seg.planet_score is not None:
            name, color, width = "Scored by CNN (lower score)", MUTED_BLUE, 1.0
        else:
            name, color, width = f"Not scored (run shorter than {MIN_INFERENCE_LENGTH:,} points)", SLATE, 1.0
        x, y = _decimate(q.time[seg.start:seg.stop], q.flux[seg.start:seg.stop])
        fig.add_trace(go.Scattergl(
            x=x, y=y, mode="lines", name=name, legendgroup=name, showlegend=name not in shown,
            line=dict(color=color, width=width),
            hovertemplate="BKJD %{x:.3f} d<br>relative flux %{y:.5f}<extra>%{fullData.name}</extra>",
        ))
        shown.add(name)

    for g in q.gaps:
        fig.add_vrect(x0=g["t_start"], x1=g["t_stop"], fillcolor="rgba(148,163,184,0.10)", line_width=0, layer="below")
    if best is not None:
        fig.add_vrect(
            x0=best.t_start, x1=best.t_stop, fillcolor=AMBER, opacity=0.10, line_width=0, layer="below",
            annotation_text=f"CNN input · {best.n_points:,} points", annotation_position="top left",
            annotation_font=dict(color=AMBER, size=11),
        )

    lo, hi = np.percentile(q.flux, [0.3, 99.7])
    pad = max(hi - lo, 1e-6) * 0.15
    # Up to 3 legend entries wrap to 2 rows in a narrow card; give them room so they aren't clipped.
    fig = _layout(fig, "Time (BKJD, days)", "Relative flux", height, uirevision, margin_b=52)
    fig.update_yaxes(range=[lo - pad, hi + pad], tickformat=".4f")
    return fig


def plot_periodogram(
    bls: BLSResult,
    selected_period: Optional[float],
    log_x: bool = True,
    height: int = 340,
    uirevision: str = "",
) -> go.Figure:
    s = bls.spectrum
    custom = np.column_stack([s["duration"] * 24.0, s["depth"]])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s["period"], y=s["power"], mode="lines+markers", name="BLS power",
        line=dict(color=CYAN, width=1.3),
        marker=dict(size=9, opacity=0.0),                       # invisible but hoverable/clickable
        customdata=custom,
        hovertemplate=("Period %{x:.4f} d<br>BLS power %{y:.1f}<br>box duration %{customdata[0]:.2f} h"
                       "<br>depth %{customdata[1]:.2f} sigma<extra></extra>"),
        selected=dict(marker=dict(opacity=0.0)), unselected=dict(marker=dict(opacity=0.0)),
    ))

    cands = bls.candidates
    cx = [c["period_days"] for c in cands]
    cy = [c["power"] for c in cands]
    colors, sizes, lines = [], [], []
    for c in cands:
        is_sel = selected_period is not None and abs(c["period_days"] - selected_period) / c["period_days"] < 1e-6
        colors.append(AMBER if is_sel else (CYAN if c is cands[0] else SLATE))
        sizes.append(14 if is_sel else 11)
        lines.append("white" if is_sel else "rgba(230,237,247,0.6)")
    fig.add_trace(go.Scatter(
        x=cx, y=cy, mode="markers+text", name="Top peaks",
        text=[str(i + 1) for i in range(len(cands))], textposition="top center", textfont=dict(size=11, color="#E6EDF7"),
        marker=dict(size=sizes, color=colors, line=dict(width=1.5, color=lines)),
        customdata=np.array([[c["duration_days"] * 24.0, c["depth"]] for c in cands]),
        hovertemplate=("Peak %{text}<br>Period %{x:.4f} d<br>BLS power %{y:.1f}<br>box duration %{customdata[0]:.2f} h"
                       "<extra></extra>"),
        selected=dict(marker=dict(opacity=1.0)), unselected=dict(marker=dict(opacity=1.0)),
    ))

    if selected_period is not None:
        pmin, pmax = float(s["period"].min()), float(s["period"].max())
        for mult, label in ((0.5, "P/2"), (2.0, "2P")):
            xp = selected_period * mult
            if pmin <= xp <= pmax:
                fig.add_vline(x=xp, line=dict(color=AMBER, width=1, dash="dot"), opacity=0.55,
                              annotation_text=label, annotation_position="top", annotation_font=dict(size=10, color=AMBER))

    fig = _layout(fig, "Trial period (days)", "BLS power", height, uirevision)
    if log_x:
        pmin, pmax = float(s["period"].min()), float(s["period"].max())
        ticks = [t for t in _LOG_PERIOD_TICKS if pmin * 0.95 <= t <= pmax * 1.05]
        fig.update_xaxes(type="log", tickvals=ticks, ticktext=[f"{t:g}" for t in ticks])
    else:
        fig.update_xaxes(type="linear")
    fig.update_layout(hovermode="closest", showlegend=False, legend=dict(y=-0.3))
    return fig


def plot_folded(view: PeriodView, height: int = 340, uirevision: str = "") -> go.Figure:
    fold = view.fold
    centers = np.asarray(fold["phase_bin_centers"], dtype=float)
    mean = np.asarray(fold["mean_flux"], dtype=float)
    npb = np.asarray(fold["n_per_bin"])
    phase = ((centers + 0.5) % 1.0) - 0.5
    order = np.argsort(phase)
    phase, mean, npb = phase[order], mean[order], npb[order]
    ok = ~np.isnan(mean)

    raw_phase = ((np.asarray(fold["phase"], dtype=float) + 0.5) % 1.0) - 0.5
    raw_flux = 1.0 + np.asarray(fold["folded_flux"], dtype=float)
    rx, ry = _decimate(raw_phase, raw_flux, 6000)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=rx, y=ry, mode="markers", name="Individual observations",
        marker=dict(size=3, color=MUTED_BLUE, opacity=0.35),
        hovertemplate="phase %{x:.3f}<br>flux %{y:.5f}<extra>observation</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=phase[ok], y=1.0 + mean[ok], mode="lines+markers", name="Binned mean",
        line=dict(color=CYAN, width=2), marker=dict(size=5, color=CYAN),
        customdata=npb[ok],
        hovertemplate="phase %{x:.3f}<br>binned flux %{y:.5f}<br>%{customdata} points<extra>binned mean</extra>",
    ))
    half = 0.5 * view.duration_days / view.period_days
    fig.add_vrect(x0=-half, x1=half, fillcolor=AMBER, opacity=0.10, line_width=0, layer="below",
                  annotation_text="BLS box", annotation_position="top left", annotation_font=dict(color=AMBER, size=11))
    fig = _layout(fig, "Orbital phase (0 = BLS transit midpoint)", "Relative flux", height, uirevision)
    fig.update_xaxes(range=[-0.5, 0.5])
    fig.update_yaxes(tickformat=".4f")
    lo, hi = np.nanpercentile(ry, [0.5, 99.5])
    pad = max(hi - lo, 1e-6) * 0.15
    fig.update_yaxes(range=[lo - pad, hi + pad])
    fig.update_layout(hovermode="closest", legend=dict(y=-0.32))
    return fig


def plot_window_depths(view: PeriodView, height: int = 300) -> go.Figure:
    """Per-window depth (relative flux) and signal-to-noise for the repeated-window analysis."""
    ev = view.rt["events"]
    t = np.array([e["center_time"] for e in ev])
    depth = np.array([e["depth"] for e in ev]) * 100.0
    snr = np.array([e["snr"] for e in ev])
    low = snr < 1.0
    fig = go.Figure()
    for mask, name, color in ((~low, "Window SNR >= 1", CYAN), (low, "Window SNR < 1", SLATE)):
        fig.add_trace(go.Scatter(
            x=t[mask], y=depth[mask], mode="markers", name=name, marker=dict(size=6, color=color),
            customdata=snr[mask],
            hovertemplate="BKJD %{x:.2f} d<br>window depth %{y:.4f}%<br>SNR %{customdata:.2f}<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig = _layout(fig, "Window position (BKJD, days)", "Window depth (% of flux)", height)
    fig.update_layout(hovermode="closest", legend=dict(y=-0.3))
    return fig


def plot_calibration_curve(mapping_x: np.ndarray, mapping_y: np.ndarray, raw_score: float, calibrated: float, height: int = 280) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mapping_x, y=mapping_y, mode="lines", name="Platt mapping", line=dict(color=CYAN, width=2),
                             hovertemplate="raw %{x:.3f}<br>mapped %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[raw_score], y=[calibrated], mode="markers", name="This segment",
                             marker=dict(size=12, color=AMBER, line=dict(width=1.5, color="white")),
                             hovertemplate="raw %{x:.4f}<br>mapped %{y:.4f}<extra>this segment</extra>"))
    fig = _layout(fig, "Raw CNN score", "Platt-mapped value", height)
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(legend=dict(y=-0.3), hovermode="closest")
    return fig
