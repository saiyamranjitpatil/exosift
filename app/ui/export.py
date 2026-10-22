"""Report and data exports built from the same AnalysisResult the dashboard shows."""

from __future__ import annotations

import io
import json
import math
import textwrap
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from exosift_pipeline import AnalysisResult, PeriodView, derive_cautions  # noqa: E402
from ui import insights  # noqa: E402

DISCLAIMER = (
    "ExoSift outputs are machine-learning screening scores and signal-analysis statistics, not confirmed "
    "exoplanet detections. The CNN was trained on 37 planet / 5,050 non-planet examples; its short-sequence "
    "behaviour was checked on only 7 validation planets."
)

DEFINITIONS = {
    "cnn_score": "Raw sigmoid output of the variable-length 1D CNN for the highest-scoring contiguous segment. Not a probability.",
    "threshold": "Validation-selected label cut-off (0.0939). It only decides the screening label.",
    "bls_power": "Box Least Squares log-likelihood power at a trial period (astropy default objective, no flux errors). Ranks periods; not an SDE and not a significance.",
    "folded_depth": "Median of the folded binned profile minus its minimum, as a fraction of the quarter's median flux.",
    "windows": "Number of period-length windows (plus/minus 0.25 period around each expected epoch) with enough points. A window is counted whatever the flux does; it is not a detected transit.",
    "repeatability_score": "Mean of 1/(1+x) for depth scatter, duration scatter and timing scatter/period across windows. Experimental: shuffled-flux controls score in the same range.",
    "calibrated_value": "Platt-scaled raw score, fitted on 7 validation planets for the 3,197-point model. Not a real-world probability.",
    "composite": "Weighted mix of the above (renormalised without the RF term). Experimental; methodology under review.",
}


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def build_report_dict(result: AnalysisResult, view: Optional[PeriodView], threshold: float) -> dict:
    q, seg = result.best_quarter, result.best_segment
    out = insights.outcome(result, threshold)
    view = view or result.top_view
    rep: dict[str, Any] = {
        "schema": "exosift-report/1",
        "disclaimer": DISCLAIMER,
        "target": {"kic": result.kic, "requested_quarter": result.requested_quarter,
                   "source": "NASA/MAST via Lightkurve", "time_system": "BKJD (BJD - 2454833)"},
        "analyzed_at": result.analyzed_at,
        "model_file": result.model_file,
        "outcome": {"key": out["key"], "title": out["title"], "summary": out["summary"]},
        "screening_threshold": threshold,
        "warnings": result.warnings,
        "stage_errors": result.errors,
        "definitions": DEFINITIONS,
    }
    if seg is None:
        return _clean(rep)

    rep["observations"] = {
        "quarter": q.label, "flux_column": q.flux_source,
        "rows_returned": q.n_rows, "valid_observations": q.n_valid, "removed": q.n_removed,
        "span_days": q.span_days, "median_cadence_minutes": q.median_cadence_min,
        "gaps": q.gaps,
        "segments": [{"index": s.index, "points": s.n_points, "bkjd_start": s.t_start, "bkjd_stop": s.t_stop,
                      "scored": s.planet_score is not None, "planet_score": s.planet_score,
                      "note": s.skipped_reason} for s in q.segments],
        "cnn_input": {"segment_index": seg.index, "points": seg.n_points, "bkjd_start": seg.t_start,
                      "bkjd_stop": seg.t_stop, "fraction_of_valid": seg.n_points / q.n_valid},
        "quarters_scanned": [x.label for x in result.quarters],
    }
    rep["cnn"] = {"raw_score": seg.planet_score, "above_threshold": bool(seg.planet_score >= threshold),
                  "segments_scored": len(result.scored_segments),
                  "note": "Highest score among scored segments; a probability it is not."}
    if result.bls:
        rep["bls"] = {"settings": result.bls.settings, "candidates": result.bls.candidates,
                      "point_sigma_relative": result.bls.point_sigma,
                      "outlier_cut_depth_relative": result.bls.outlier_cut_depth}
    if view is not None:
        v: dict[str, Any] = {"source": view.source, "period_days": view.period_days, "bls_power": view.power,
                             "box_duration_hours": view.duration_days * 24.0,
                             "folded_depth_percent": None if not view.fold or view.fold.get("transit_depth") is None
                             else 100.0 * view.fold["transit_depth"],
                             "fold_summary": view.fold_summary}
        if view.rt:
            v["windows"] = {"evaluated": view.rt["n_transits"], "repeatability_score": view.rt["repeatability_score"],
                            "experimental": True}
        rep["period_view"] = v
    if result.null_control:
        rep["shuffled_flux_control"] = result.null_control
    rep["experimental"] = {"calibrated_value": result.calibrated, "composite": result.composite}
    rep["cautions"] = derive_cautions(result, result.top_view)
    return _clean(rep)


def build_json(result: AnalysisResult, view: Optional[PeriodView], threshold: float) -> str:
    return json.dumps(build_report_dict(result, view, threshold), indent=2)


def build_light_curve_csv(result: AnalysisResult) -> str:
    q, best = result.best_quarter, result.best_segment
    seg_id = np.full(len(q.time), -1, dtype=int)
    for s in q.segments:
        seg_id[s.start:s.stop] = s.index
    df = pd.DataFrame({
        "bkjd_days": q.time, "relative_flux": q.flux, "segment_index": seg_id,
        "in_cnn_input": seg_id == (best.index if best else -2),
    })
    return df.to_csv(index=False)


def build_periodogram_csv(result: AnalysisResult) -> str:
    s = result.bls.spectrum
    return pd.DataFrame({
        "period_days": s["period"], "bls_power": s["power"], "box_duration_days": s["duration"],
        "bls_depth_sigma": s["depth"], "transit_time_bkjd": s["transit_time"],
    }).to_csv(index=False)


def build_segments_csv(result: AnalysisResult) -> str:
    rows = []
    for q in result.quarters:
        for s in q.segments:
            rows.append({"quarter": q.label, "segment_index": s.index, "points": s.n_points,
                         "bkjd_start": s.t_start, "bkjd_stop": s.t_stop, "span_days": s.span_days,
                         "scored": s.planet_score is not None, "planet_score": s.planet_score, "note": s.skipped_reason})
    return pd.DataFrame(rows).to_csv(index=False)


def build_windows_csv(view: PeriodView) -> str:
    return pd.DataFrame(view.rt["events"]).to_csv(index=False)


def _wrap(text: str, width: int = 98) -> str:
    return "\n".join(textwrap.fill(p, width) for p in text.split("\n"))


def build_pdf(result: AnalysisResult, view: Optional[PeriodView], threshold: float) -> bytes:
    q, seg = result.best_quarter, result.best_segment
    view = view or result.top_view
    out = insights.outcome(result, threshold)
    interp = insights.interpretation(result, threshold, view) if seg else None
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, "ExoSift analysis report", fontsize=20, fontweight="bold")
        fig.text(0.07, 0.925, f"KIC {result.kic}  |  {q.label if q else 'n/a'}  |  analyzed {result.analyzed_at}", fontsize=9, color="#444")
        y = 0.89
        fig.text(0.07, y, out["title"], fontsize=14, fontweight="bold")
        fig.text(0.07, y - 0.012, _wrap(out["summary"]), fontsize=9, va="top")
        y -= 0.08
        if seg is not None:
            rows = [
                ("Observations in quarter", f"{q.n_valid:,} valid of {q.n_rows:,} rows; {len(q.gaps)} gap(s)"),
                ("Observations used by CNN", f"{seg.n_points:,} (segment BKJD {seg.t_start:.2f} to {seg.t_stop:.2f})"),
                ("CNN raw score / threshold", f"{seg.planet_score:.4f} / {threshold:.4f}"),
            ]
            if view is not None:
                rows += [("BLS best period (power)", f"{result.bls.candidates[0]['period_days']:.4f} d ({result.bls.candidates[0]['power']:.1f})")]
                if view.fold and view.fold.get("transit_depth") is not None:
                    rows.append(("Folded depth at viewed period", f"{100 * view.fold['transit_depth']:.4f} % (P = {view.period_days:.4f} d)"))
                if view.rt:
                    rows.append(("Period windows / score (experimental)", f"{view.rt['n_transits']} / {view.rt['repeatability_score']:.3f}"))
            for k, v in rows:
                fig.text(0.07, y, k, fontsize=9, color="#444")
                fig.text(0.42, y, v, fontsize=9)
                y -= 0.022
            y -= 0.01
            fig.text(0.07, y, "Cautions and uncertainty", fontsize=11, fontweight="bold")
            y -= 0.02
            for item in interp["uncertain"]:
                txt = _wrap("- " + item, 100)
                fig.text(0.07, y, txt, fontsize=8.5, va="top")
                y -= 0.017 * (txt.count("\n") + 1) + 0.006
            y -= 0.01
            fig.text(0.07, y, "Recommended next step", fontsize=11, fontweight="bold")
            y -= 0.02
            txt = _wrap(interp["next_step"], 100)
            fig.text(0.07, y, txt, fontsize=8.5, va="top")
            y -= 0.017 * (txt.count("\n") + 1) + 0.02
        fig.text(0.07, y, "Definitions", fontsize=11, fontweight="bold")
        y -= 0.02
        for k, v in DEFINITIONS.items():
            txt = _wrap(f"{k}: {v}", 108)
            fig.text(0.07, y, txt, fontsize=7.5, va="top", color="#222")
            y -= 0.0135 * (txt.count("\n") + 1) + 0.005
        fig.text(0.07, 0.04, _wrap(DISCLAIMER, 112), fontsize=7.5, color="#555", va="bottom")
        pdf.savefig(fig)
        plt.close(fig)

        if seg is not None:
            fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69))
            ax = axes[0]
            for s in q.segments:
                color = "#0284C7" if s.index == seg.index else ("#7C93C9" if s.planet_score is not None else "#9CA3AF")
                ax.plot(q.time[s.start:s.stop], q.flux[s.start:s.stop], lw=0.8, color=color)
            ax.axvspan(seg.t_start, seg.t_stop, color="#F5A524", alpha=0.12)
            ax.set(title=f"{q.label} light curve (blue = CNN input segment, grey = not scored)", xlabel="Time (BKJD, days)", ylabel="Relative flux")
            if result.bls is not None:
                sp = result.bls.spectrum
                ax = axes[1]
                ax.plot(sp["period"], sp["power"], lw=0.9, color="#0284C7")
                ax.scatter([c["period_days"] for c in result.bls.candidates], [c["power"] for c in result.bls.candidates], color="#F5A524", zorder=3)
                ax.set(xscale="log", title="BLS periodogram (peak height ranks periods; not a significance test)", xlabel="Trial period (days)", ylabel="BLS power")
            if view is not None and view.fold is not None:
                f = view.fold
                ph = ((np.asarray(f["phase"]) + 0.5) % 1.0) - 0.5
                axes[2].scatter(ph, 1 + np.asarray(f["folded_flux"]), s=2, color="#7C93C9", alpha=0.4)
                c = ((np.asarray(f["phase_bin_centers"]) + 0.5) % 1.0) - 0.5
                o = np.argsort(c)
                axes[2].plot(c[o], 1 + np.asarray(f["mean_flux"])[o], color="#0284C7", lw=1.6)
                axes[2].set(title=f"Phase fold at P = {view.period_days:.4f} d", xlabel="Orbital phase (0 = BLS midpoint)", ylabel="Relative flux")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return buf.getvalue()
