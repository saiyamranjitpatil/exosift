from pathlib import Path

import pandas as pd
import streamlit as st

from ui import components, style

style.inject()
components.hero("Methodology", [("How ExoSift produces each number", False)], status="Research prototype")
st.write("")

MODELS = Path(__file__).resolve().parents[2] / "models"

tab_over, tab_data, tab_cnn, tab_signal, tab_exp, tab_status = st.tabs(
    ["Overview", "Data and counts", "CNN screening", "Signal analysis", "Experimental outputs", "What is verified"]
)

with tab_over:
    st.markdown(
        "ExoSift screens a Kepler star in two independent ways and shows both. A 1D convolutional neural network scores "
        "the light curve, and a Box Least Squares (BLS) search looks for a repeating dip. Neither is a detection."
    )
    st.dataframe(pd.DataFrame([
        {"Step": "1. Retrieve", "What happens": "Kepler long-cadence light curves are downloaded from NASA/MAST with Lightkurve (PDCSAP flux, else SAP).", "Status": "Verified in code"},
        {"Step": "2. Clean and split", "What happens": "Non-finite and duplicate samples are dropped. The quarter is split wherever the gap exceeds 3x the median cadence. Runs of at least 1,024 points can be scored.", "Status": "Verified in code"},
        {"Step": "3. CNN screening", "What happens": "The variable-length CNN scores each qualifying run. The highest score is shown.", "Status": "Limited validation (7 planets)"},
        {"Step": "4. BLS periodogram", "What happens": "BLS power is computed for 2,000 trial periods from 0.5 to 50 days on the whole quarter.", "Status": "Computation verified; peaks are not significance"},
        {"Step": "5. Phase fold", "What happens": "The quarter is folded at a chosen period and averaged in 200 phase bins.", "Status": "Verified in code"},
        {"Step": "6. Window evidence", "What happens": "Period-length windows are counted and scored for consistency.", "Status": "Experimental"},
        {"Step": "7. Calibration", "What happens": "The raw score is mapped by a Platt scaler.", "Status": "Experimental"},
        {"Step": "8. Composite score", "What happens": "A weighted mix of the above.", "Status": "Experimental, under review"},
    ]), hide_index=True, alt="ExoSift pipeline steps and their verification status")

with tab_data:
    st.markdown(
        "**Why the chart can show 4,070 observations while the CNN uses 1,925.** The two numbers count different things "
        "at different stages:"
    )
    st.markdown(
        "1. **Rows returned.** Lightkurve returns every cadence in the quarter, including a few with missing flux.\n"
        "2. **Valid observations.** Samples with non-finite time or flux, and duplicate timestamps, are dropped. "
        "This is the quarter total shown in the chart and the footer.\n"
        "3. **Contiguous runs.** Kepler pauses to download data, which leaves real gaps. A gap longer than 3x the median "
        "cadence ends a run. The pipeline never bridges a gap.\n"
        "4. **CNN input.** The CNN needs one unbroken run of at least 1,024 points, so shorter runs cannot be scored. "
        "Every qualifying run is scored and the **highest-scoring run** is shown, which biases the headline score upward.\n\n"
        "BLS, the phase fold and the window analysis use **all** valid observations of the quarter."
    )
    cache = st.session_state.get("_analysis_cache")
    result = next(reversed(cache.values())) if cache else None
    if result is not None and result.best_quarter is not None:
        q, seg = result.best_quarter, result.best_segment
        st.info(
            f"**Your current analysis, KIC {result.kic} {q.label}:** {q.n_rows:,} rows returned, {q.n_removed} removed, "
            f"{q.n_valid:,} valid; {len(q.gaps)} gap(s) make {len(q.segments)} runs "
            f"({', '.join(f'{s.n_points:,}' for s in q.segments)} points). The CNN used {seg.n_points:,} "
            f"({seg.n_points / q.n_valid:.0%} of the quarter).",
            icon=":material/fact_check:",
        )
    st.markdown(
        "**Preprocessing for the CNN** (the same function used for training): fill non-finite values, estimate a baseline with "
        "a 101-point rolling median, **replace points more than 7 robust sigma from the baseline by interpolation**, "
        "remove the baseline, rescale by the median absolute deviation and clip to plus/minus 5. "
        "The outlier step and the clip were tuned on the Kaggle training data."
    )

with tab_cnn:
    st.markdown(
        "**Model.** Three Conv1D blocks (32, 64 and 128 filters; kernel sizes 15, 11 and 7) with batch normalisation, ReLU, "
        "max pooling and dropout, then global average pooling, a 64-unit dense layer and a sigmoid output. "
        "The output is a score between 0 and 1, **not a probability**."
    )
    st.markdown(
        "**Training data.** The Kaggle Kepler exoplanet dataset: 5,087 training stars (**37 planets**, 5,050 non-planets) and a "
        "held-out test set of 570 stars (**5 planets**). The test set was not used to choose the threshold."
    )
    st.markdown(
        "**Variable length.** The saved model was trained on 3,197-point sequences. A copy with an unspecified input length "
        "was built by recreating the same layers and transferring the exact weights; global average pooling makes this "
        "possible. Nothing was retrained."
    )
    csv = MODELS / "variable_length_validation.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        show = df[["length", "n_positives", "roc_auc", "pr_auc", "tuned_precision", "tuned_recall", "tuned_f1"]].rename(columns={
            "length": "Input length", "n_positives": "Planets in validation", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
            "tuned_precision": "Precision", "tuned_recall": "Recall", "tuned_f1": "F1"})
        st.markdown("**Validation by input length** (validation split only, threshold 0.0939):")
        st.dataframe(show, hide_index=True, alt="CNN validation metrics at three input lengths", column_config={
            c: st.column_config.NumberColumn(format="%.3f") for c in ("ROC-AUC", "PR-AUC", "Precision", "Recall", "F1")})
        st.warning(
            f"Every row rests on only {int(df['n_positives'].iloc[0])} planets, so one more or fewer changes recall by about 14 points. "
            "Shorter inputs are leading slices of the 3,197-point sequences, which may cut a transit out, so the drop "
            "at shorter lengths may partly reflect the test itself. Lengths above 3,197 have not been validated.",
            icon=":material/warning:",
        )
    st.markdown(
        "**Threshold.** 0.0939 maximised F1 on the validation split. It only decides the label in the interface; "
        "the slider relabels the score and does not re-run anything."
    )

with tab_signal:
    st.markdown(
        "**Box Least Squares.** The flux is detrended and rescaled (the same function as for the CNN), then astropy's "
        "BoxLeastSquares is evaluated at 2,000 logarithmically spaced periods from 0.5 to 50 days with 50 box durations. "
        "**BLS power** is astropy's default log-likelihood with no flux errors supplied, so it grows with the number of "
        "points. It ranks trial periods. It is not a signal detection efficiency and no significance test is applied."
    )
    st.markdown(
        "**Things to know when reading a periodogram**\n"
        "- A real periodic signal also produces peaks at multiples and fractions of its period (harmonics). Several top peaks "
        "lining up usually means one signal, not several.\n"
        "- Smooth stellar variability can produce strong BLS peaks. A transit is a narrow, flat-bottomed dip; check the fold.\n"
        "- The preprocessing replaces points beyond 7 sigma, so **very deep transits are removed before BLS**. For the star "
        "analysed in the dashboard the threshold is shown in the periodogram dialog.\n"
        "- The box-duration grid is a fixed 0.12 to 4.8 hours for every period. The code comments describe fractions of the "
        "period, but the values are passed as absolute days. On the worked example this does not move the best period."
    )
    st.markdown(
        "**Phase fold.** Observations are placed by their phase at the chosen period and averaged in 200 bins. Depth is the "
        "median of the binned profile minus its minimum, relative to the quarter's median flux. The fold uses flux that is "
        "**not detrended**, whereas BLS uses detrended flux."
    )
    st.markdown(
        "**Window evidence (experimental).** Around each expected epoch, a window of plus/minus 0.25 period is examined. "
        "A window counts as an event when it has at least 3 points, at least 2 of them in the central third. "
        "Depth is the local median minus the lowest central point, which is positive even for pure noise. The score is the mean of "
        "three 1/(1+x) terms for depth, duration and timing scatter. **A window is counted whatever the flux does**, and the score "
        "does not separate the worked example from shuffled copies of its own flux. It is therefore not used in the outcome."
    )

with tab_exp:
    st.markdown(
        "**Calibrated value.** A Platt (logistic) scaler fitted to the CNN's validation outputs. It was fitted on 1,018 "
        "validation samples of which **7 are planets**, using the fixed 3,197-point model, and is applied here to the "
        "variable-length model. Its fit was evaluated on the same data. Because the validation base rate is under 1%, the "
        "mapped value stays low whatever the star. **It is not the probability that a star hosts a planet.**"
    )
    st.markdown(
        "**Composite score.** A weighted sum of the CNN score (0.25), a BLS term `min(power/10, 1)` (0.20), a depth term "
        "`min(depth/0.05, 1)` (0.15) and the window score (0.15), renormalised because no random-forest score exists for a live "
        "target. The BLS term is pinned at 1.0 for any power above 10 (including pure noise), the depth term penalises small "
        "planets, and the window score is uninformative. The weights were not derived from data. Treat it as a placeholder "
        "until the methodology is reviewed."
    )

with tab_status:
    st.dataframe(pd.DataFrame([
        {"Output": "Observation counts and gaps", "Status": "Verified in code", "Basis": "Traced from Lightkurve rows to the CNN input."},
        {"Output": "CNN score", "Status": "Reproducible, limited evidence", "Basis": "Same value as the original script; validated on 7 planets."},
        {"Output": "BLS best period", "Status": "Reproducible, not significance", "Basis": "Stable under finer period grids; ranks periods only."},
        {"Output": "Folded depth", "Status": "Verified in code", "Basis": "Relative flux units; not detrended."},
        {"Output": "Window count and score", "Status": "Experimental", "Basis": "Scores noise like data."},
        {"Output": "Calibrated value", "Status": "Experimental", "Basis": "7 validation planets; length mismatch."},
        {"Output": "Composite score", "Status": "Experimental", "Basis": "Arbitrary caps and weights."},
    ]), hide_index=True, alt="Verification status of each displayed output")
    st.caption(
        "The full audit, with the experiments behind these statements and the unresolved limitations, is in "
        "docs/scientific_audit.md in the project folder."
    )
