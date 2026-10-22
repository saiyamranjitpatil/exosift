"""Data-derived wording for the dashboard (outcome, key insights, interpretation).

Pure functions: no Streamlit imports, every sentence is built from computed values so
nothing on screen can drift from the analysis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from exosift_pipeline import (
    MIN_INFERENCE_LENGTH,
    AnalysisResult,
    PeriodView,
    derive_cautions,
    harmonic_relation,
)

HELP = {
    "cnn_score": (
        "Raw output of the 1D CNN (0-1) for the highest-scoring contiguous segment. It is a screening "
        "score, not a probability of a planet. The threshold only sets the label."
    ),
    "bls_period": (
        "Trial period with the highest Box Least Squares power. It ranks candidate periods; it is not "
        "a confirmed orbital period and has no significance test attached."
    ),
    "coverage": (
        "Observations the CNN scored versus valid observations in the quarter. The CNN needs an "
        "unbroken run of at least 1,024 points, so shorter runs are not scored."
    ),
    "quality": (
        "Rows returned by NASA/MAST versus rows kept after removing non-finite or duplicate samples, "
        "plus the number of real gaps (more than 3x the median cadence)."
    ),
}


def fmt_int(n: int) -> str:
    return f"{int(n):,}"


def pct(frac: float) -> str:
    """Percent with enough precision that a 0.5% bin never reads as 0%."""
    v = 100.0 * frac
    return f"{v:.1f}%" if v < 10 else f"{v:.0f}%"


def plural(n: int, one: str, many: Optional[str] = None) -> str:
    return one if n == 1 else (many or one + "s")


def outcome(result: AnalysisResult, threshold: float) -> dict:
    """Careful outcome state. Never stronger than 'preliminary'."""
    seg = result.best_segment
    if seg is None:
        return {
            "key": "insufficient",
            "title": "Insufficient data",
            "summary": (
                f"No contiguous run of at least {fmt_int(MIN_INFERENCE_LENGTH)} observations could be scored "
                "in the quarters that were retrieved. Try another quarter or target."
            ),
            "tone": "amber",
            "status": "Insufficient data",
        }
    if result.bls is None or not result.bls.candidates or result.top_view is None:
        return {
            "key": "inconclusive",
            "title": "Inconclusive",
            "summary": (
                "The CNN scored the data, but the periodicity analysis did not produce a usable result, "
                "so the signal cannot be assessed beyond the raw score."
            ),
            "tone": "amber",
            "status": "Preliminary",
        }
    if seg.planet_score >= threshold:
        return {
            "key": "candidate",
            "title": "Preliminary candidate signal",
            "summary": (
                f"The CNN score ({seg.planet_score:.4f}) is above the screening threshold ({threshold:.4f}). "
                "Periodicity and transit-shape evidence still require verification; this is not a confirmed detection."
            ),
            "tone": "cyan",
            "status": "Preliminary screening",
        }
    return {
        "key": "no_candidate",
        "title": "No candidate above screening threshold",
        "summary": (
            f"The highest CNN score ({seg.planet_score:.4f}) is below the screening threshold ({threshold:.4f}). "
            "This does not rule out a planet; the model was trained on very few examples."
        ),
        "tone": "slate",
        "status": "Preliminary screening",
    }


def lightcurve_insight(result: AnalysisResult) -> str:
    q, seg = result.best_quarter, result.best_segment
    scored = [s for s in q.segments if s.planet_score is not None]
    short_pts = sum(s.n_points for s in q.segments if not s.scorable)
    n_gaps = len(q.gaps)
    text = (
        f"{q.label} has {fmt_int(q.n_valid)} valid observations over {q.span_days:.1f} days. "
        f"{n_gaps} {plural(n_gaps, 'gap')} split it into {len(q.segments)} contiguous "
        f"{plural(len(q.segments), 'run')}. The CNN scored {len(scored)} {plural(len(scored), 'run')} "
        f"and the highest-scoring one ({fmt_int(seg.n_points)} points) is highlighted."
    )
    if short_pts:
        text += f" {fmt_int(short_pts)} points lie in runs shorter than {fmt_int(MIN_INFERENCE_LENGTH)} and cannot be scored."
    return text


def periodogram_insight(result: AnalysisResult, view: Optional[PeriodView]) -> str:
    bls = result.bls
    top = bls.candidates[0]
    parts = []
    if len(bls.candidates) > 1:
        ratio = top["power"] / bls.candidates[1]["power"]
        parts.append(
            f"The strongest peak is at {top['period_days']:.4f} d (power {top['power']:.1f}), "
            f"{ratio:.1f}x the next-highest peak."
        )
        related = [c for c in bls.candidates[1:] if harmonic_relation(c["period_days"], top["period_days"])]
        if related:
            parts.append(
                f"{len(related)} of the other {len(bls.candidates) - 1} top peaks are simple multiples or fractions of it."
            )
    else:
        parts.append(f"The strongest peak is at {top['period_days']:.4f} d (power {top['power']:.1f}).")
    if view is not None and view.source != "bls_best":
        rank = int((bls.spectrum["power"] > view.power).sum()) + 1
        parts.append(
            f"You are viewing P = {view.period_days:.4f} d (power {view.power:.1f}, rank {rank} of "
            f"{fmt_int(len(bls.spectrum['power']))} trial periods)."
        )
    parts.append("Peak height ranks trial periods; it is not a significance test.")
    return " ".join(parts)


def fold_insight(view: PeriodView) -> str:
    d = view.fold_summary
    if not d:
        return "The folded profile could not be summarised for this period."
    text = (
        f"Folded at {view.period_days:.4f} d, the deepest bin is {100 * d['depth_rel']:.3f}% below the median"
    )
    ratio = d.get("depth_over_scatter")
    if ratio is not None and np.isfinite(ratio):
        text += f", {ratio:.1f}x the scatter of the out-of-dip bins"
    text += (
        f". That dip spans about {pct(d['dip_width_phase'])} of the phase; "
        f"the BLS box spans {pct(d['bls_box_phase'])}."
    )
    if ratio is not None and np.isfinite(ratio):
        if ratio < 3:
            text += " A dip this close to the bin-to-bin scatter is not visually conclusive."
        else:
            text += " The dip stands clearly above the bin scatter, but its shape and period still need independent checks."
    return text


def evidence_insight(result: AnalysisResult, view: PeriodView, null: Optional[dict]) -> str:
    rt = view.rt
    if rt is None:
        return "The window analysis could not be computed for this period."
    snrs = [e["snr"] for e in rt["events"]]
    text = (
        f"{rt['n_transits']} period-length windows were evaluated"
        + (f" (median per-window signal-to-noise {np.median(snrs):.1f})." if snrs else ".")
    )
    score = rt["repeatability_score"]
    if null and null.get("n") and score is not None:
        text += (
            f" The same score on {null['n']} shuffled copies of this flux ranges {null['min']:.2f}-{null['max']:.2f}; "
        )
        if score <= null["max"]:
            text += f"this star's {score:.3f} falls inside that range, so the score does not separate it from noise."
        else:
            text += f"this star's {score:.3f} is above that range, but the metric is still experimental."
    return text


def interpretation(result: AnalysisResult, threshold: float, view: Optional[PeriodView]) -> dict:
    q, seg = result.best_quarter, result.best_segment
    view = view or result.top_view
    cautions = derive_cautions(result, result.top_view)
    codes = {c["code"] for c in cautions}

    supports = [
        f"CNN score {seg.planet_score:.4f} is {'above' if seg.planet_score >= threshold else 'below'} the screening "
        f"threshold {threshold:.4f} (a label on a raw model output)."
    ]
    if result.bls and result.bls.candidates:
        supports.append(
            f"BLS ranks {result.bls.candidates[0]['period_days']:.4f} d as its strongest of "
            f"{fmt_int(len(result.bls.spectrum['period']))} trial periods."
        )
    supports.append(
        f"Results come from {fmt_int(q.n_valid)} real observations ({q.label}, {q.flux_source}); "
        "no synthetic data and no resampling across gaps."
    )
    fs = result.top_view.fold_summary if result.top_view else None
    if fs and abs(fs["phase_at_min"]) < 0.05:
        supports.append("The folded profile's deepest bin sits at the BLS transit midpoint.")

    uncertain = [c["text"] for c in cautions if c["level"] == "caution"]
    uncertain.append("Window count, repeatability, calibrated probability and composite score are experimental.")
    uncertain.append("This is not a confirmed exoplanet detection.")

    if codes & {"harmonic_ladder", "long_duration"}:
        nxt = "Compare the fold at P and 2P (Explore phase fold) and check whether the dip is box-shaped or a smooth variation."
    elif seg.planet_score < threshold:
        nxt = "Try another quarter of this star or a different target."
    elif "single_quarter" in codes:
        nxt = "Re-run with Quarter set to Auto to see whether the same period appears in other quarters."
    else:
        nxt = "Inspect the dip in the phase fold, then verify with independent data."
    return {"supports": supports, "uncertain": uncertain, "next_step": nxt, "cautions": cautions}


def data_quality_text(result: AnalysisResult) -> tuple[str, str]:
    q = result.best_quarter
    n_gaps = len(q.gaps)
    return (
        f"{fmt_int(q.n_valid)} of {fmt_int(q.n_rows)}",
        f"valid rows, {n_gaps} {plural(n_gaps, 'gap')}",
    )
