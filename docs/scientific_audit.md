# ExoSift scientific audit

This document traces four displayed numbers back to source, states what is and is not
verified, and records the experiments behind each claim. It was produced while building
the V2.1 dashboard (`app/`) and backs the in-app Methodology page, which is the version a
user actually reads; this file has the full derivations and raw experiment output.

Worked example throughout: **KIC 757076, Quarter 2** (the dashboard's own example target).

## 1. Why the light curve shows 4,070 observations while the CNN uses 1,925

**Trace, stage by stage** (`src/exosift_pipeline.py:prepare_quarter`, reusing
`predict_real_star.clean_time_flux` / `find_contiguous_runs` unchanged):

| Stage | Count | Where |
|---|---:|---|
| Rows returned by Lightkurve | 4,075 | `search_kepler` + `.download()` |
| Non-finite flux (dropped) | 5 | `clean_time_flux` |
| **Valid observations** | **4,070** | `clean_time_flux` output — this is what the chart and KPI call "the quarter" |
| Real gaps (> 3x median cadence) | 4 | `find_contiguous_runs` gap rule |
| Contiguous runs produced | 5 | sizes 570, 1925, 312, 1160, 103 |
| Runs >= 1,024 points (CNN-scorable) | 2 | 1,925 and 1,160 |
| **CNN input (highest-scoring run)** | **1,925** | `score_segment` on the 1,925-point run |

So "4,070" and "1,925" count different things at different stages: 4,070 is every valid
observation in the quarter; 1,925 is the single longest unbroken run, which is also the
only one the dashboard highlights because it scored higher than the other scorable run
(0.567409 vs 0.029426). The pipeline never bridges a gap or interpolates across one — each
run is scored independently or not scored at all (the three runs under 1,024 points, 985
points total, are never passed to the CNN).

**Consequence not previously surfaced:** because the *highest* of two scores is shown, the
headline is biased upward relative to picking either run at random. The dashboard now
states this explicitly (`derive_cautions`, code `selection_effect`) and the light-curve
card distinguishes three states (highest-scoring segment / scored but lower / not
scored) instead of a single highlighted span.

**Preprocessing parity:** `score_segment` calls the same `preprocess_light_curve` used in
training (rolling-median detrend, 7-sigma outlier interpolation, MAD normalisation, clip
+-5), confirmed by reading `src/predict_real_star.py:preprocess_window` and
`src/preprocessing.py` directly — no separate "inference preprocessing" path exists.

**Variable-length input validity:** addressed in section 5.

## 2. How "117 events" and the 0.781 repeatability score are calculated

**Trace** (`src/repeated_transit.py:analyze_repeated_transits`, unchanged by this work):

1. Fold the light curve at the candidate period to find the deepest phase bin (the
   "transit center").
2. Lay out expected epochs `transit_time + k * period` across the observed time range.
3. For each epoch, take a window of +/- 0.25 period. **Count it as an event if it has at
   least 3 points, with at least 2 in the central third** — regardless of what the flux
   does there.
4. Per-window depth = median of the outer two-thirds minus the minimum of the central
   third. This is non-negative by construction (it subtracts the window's own minimum),
   so it is positive even for pure noise.
5. Score = mean of three `1/(1+x)` terms for depth scatter, duration scatter, and timing
   scatter/period across all counted windows.

**Is 117 plausible as a count of windows (not transits)?** Quarter span 88.7 d / period
0.7262 d = 122.1; 117 of those 122 candidate epochs had >= 3 points nearby (the rest fall
in gaps). That is windows-that-fit-in-the-data, not a transit count — confirmed by reading
the code, not an assumption.

**Does the score separate the real star from noise?** I generated 20 shuffled-flux
controls — the real quarter's flux values randomly permuted in time (destroying any real
signal) at the *same* period and epoch — and re-ran `analyze_repeated_transits` on each
(`src/exosift_pipeline.py:null_control`, exposed live in the "How is this calculated?"
dialog). Result for KIC 757076 Q02 at P=0.7262 d:

```
shuffled-flux repeatability scores (n=20): min 0.760, median 0.771, max 0.782
real star:                                                        0.781
```

**The real star's score falls inside the shuffled-noise range.** The metric is reported in
the dashboard as experimental, excluded from the outcome banner, and the control is shown
live next to the metric so this isn't just an assertion.

**Why the score is noise-like:** the timing term is `1/(1+timing_scatter/period)`; because
a window's detected minimum can only move within +/- 0.25 period of its center, that
scatter is bounded by construction and the term sits near 0.9-1.0 whether or not a transit
exists — it is not measuring transit consistency so much as window geometry.

**Cutoff provenance:** `score_threshold=0.7` in `is_repeatable()` is a literal default in
the function signature, not a value derived from a ROC curve or calibration study.

## 3. The formula behind the composite score of 0.613

**Formula** (`src/explainability.py:rank_candidate`, called from
`src/exosift_pipeline.py:compute_composite`):

```
normalised_cnn     = cnn_probability                       (raw sigmoid, used as-is)
normalised_bls     = min(bls_power / 10, 1)
normalised_depth   = min(relative_transit_depth / 0.05, 1)
normalised_repeat  = repeatability_score                    (used as-is)

composite = sum(weight[k] * normalised[k] for k in {cnn, bls, depth, repeatability})
```

Original weights `{cnn: .25, rf: .25, bls: .20, depth: .15, repeatability: .15}` assume a
Random Forest score exists. For a live target there is no RF score (the V1 RF model
predicts on hand-engineered Kaggle features, not raw light curves), so the app drops the
`rf` term and renormalises the rest to sum to 1: `{cnn: 1/3, bls: 4/15, depth: 1/5,
repeatability: 1/5}`.

**Reproduction.** With KIC 757076 Q02's actual numbers (CNN 0.567409, BLS power 215.3,
folded depth 0.00028155, repeatability 0.7815):

```
normalised_cnn    = 0.567409
normalised_bls    = min(215.3/10, 1)   = 1.0        <- saturated
normalised_depth  = min(0.00028155/0.05, 1) = 0.00563
normalised_repeat = 0.7815

composite = (1/3)(0.567409) + (4/15)(1.0) + (1/5)(0.00563) + (1/5)(0.7815)
          = 0.18914 + 0.26667 + 0.00113 + 0.15630
          = 0.61323
```

This matches the dashboard's displayed 0.613 (reproduced exactly; see
`tests/test_pipeline_helpers.py:TestComposite.test_matches_known_worked_example`).

**Why this is not a defensible score, even after the RF fix:**
- The BLS term saturates at power >= 10. The shuffled-flux controls in section 2 had BLS
  power around 13 on unrelated noise realizations (not shown above, but observed during
  development on permuted flux) — the term is pinned at 1.0 for noise just as readily as
  for a real peak, so it contributes no discriminating information at typical Kepler power
  levels.
- The depth cap of 0.05 (5%) is arbitrary and penalises small, realistic planets (an
  Earth-size transit of a Sun-like star is about 0.01%, two orders of magnitude below the
  cap).
- The repeatability term is the noise-like metric from section 2.
- The CNN term is a raw, uncalibrated sigmoid output, not a probability.
- No component weight was fit to data; they are hand-picked defaults.

**Conclusion:** the composite is reproducible and its arithmetic is verified, but its
methodology is not validated. The dashboard labels it "Experimental: methodology under
review", keeps it out of the primary KPIs and outcome banner, and shows the full
component breakdown in Advanced diagnostics rather than hiding the reasoning.

## 4. Calibration methodology behind the 0.050156 value

**Trace** (`src/calibration.py`):

- `fit_and_save_calibrators()` reads `models/cnn_validation_predictions.csv`: **1,018
  validation rows, 7 of them planets** (0.69% base rate) — these are the fixed 3,197-point
  CNN's outputs on the held-out validation split (never the test split).
- Platt scaling fits a 1-D logistic regression, raw score -> P(planet), on exactly those
  1,018 rows, and is *evaluated on the same 1,018 rows* (`evaluate_calibration`, called
  with the training data — the code's own docstring calls this "an optimistic estimate").
- `CNNCalibrator(method="platt").calibrate(x)` applies `sigmoid(3.801*x - 5.098)` (the
  fitted coefficients, read directly from the saved joblib file).

**Applied to a different model than it was fit on.** The dashboard's CNN input is the
*variable-length* model (`cnn_1d_variable.keras`) scoring a 1,925-point segment; the
calibrator was fit on the *fixed* 3,197-point model's validation outputs. The two models
share identical weights (section 5), but a 1,925-point input is a different statistical
regime than the 3,197-point validation data the mapping was fit on.

**Why the output stays low regardless of the star:**

```
raw 0.0939 -> platt 0.0087      raw 0.5000 -> platt 0.0393
raw 0.2000 -> platt 0.0129      raw 0.5674 -> platt 0.0502   (KIC 757076 Q02)
raw 0.9000 -> platt 0.1575      raw 0.9900 -> platt 0.2083
```

Even a raw score of 0.99 maps to about 0.21. With a 0.69% base rate and only 7 positives,
the logistic fit cannot produce a mapped value much above ~0.2-0.3 regardless of how
extreme the raw score is — the mapped number mostly reflects the training base rate, not
evidence specific to the star being analysed. The isotonic calibrator (also fit and saved)
is degenerate in the other direction — it maps 0.0939 to 0.9998 — and is not used by the
app, by original design.

**Conclusion:** the Platt mapping is reproducible and its arithmetic is correct, but it is
**not** the probability that a given star hosts a planet. The dashboard calls it
"calibrated value", keeps it inside Advanced diagnostics, and states this limitation
directly next to the number.

## 5. Variable-length CNN validity

- **Weight transfer:** `src/validate_variable_cnn.py:verify_weight_transfer()` compares
  every weight tensor between `cnn_1d.keras` (fixed, 3,197-point) and
  `cnn_1d_variable.keras` (variable-length) with `np.allclose`. Re-run during this audit:
  shapes match and values match exactly (`rtol=1e-5`) — the variable-length model is a
  faithful architectural rebuild with `set_weights()`, not a retrained or approximated
  model.
- **Validation evidence is thin.** `models/variable_length_validation.csv` (produced by the
  same script, on the stratified validation split — the held-out test set is never read)
  shows, at the three lengths tested:

  | Input length | Validation planets | ROC-AUC | PR-AUC | Recall @ 0.0939 |
  |---:|---:|---:|---:|---:|
  | 1,024 | 7 | 0.908 | 0.444 | 0.571 |
  | 1,925 | 7 | 0.917 | 0.838 | 0.571 |
  | 3,197 (training length) | 7 | 0.967 | 0.861 | 0.857 |

  Every row rests on 7 positives — one more or fewer shifts recall by ~14 points, so these
  are indicative, not precise. The 1,024/1,925-point sequences are **leading truncations**
  of the same 3,197-point validation sequences (`slice_to_length`), so a transit occurring
  later in a sequence can be truncated away at the shorter lengths — the apparent drop in
  recall at shorter lengths is partly a property of this truncation, not purely a model
  limitation. No length above 3,197 has been validated at all.
- **Threshold 0.0939** was chosen by maximizing F1 on this same validation split
  (`src/train_cnn.py`), i.e. also informed by only 7 positives.

**Conclusion:** the weight transfer is verified and exact. Real-world performance at
1,925 points (the length actually used for the dashboard's worked example) is plausible
but resting on very little evidence, and the dashboard says so wherever the score appears.

## 6. The `detect_periods` change made for this dashboard

**What changed:** `src/periodicity.py:detect_periods` gained one new, opt-in keyword,
`return_spectrum=False`. When left at its default, the function's behavior and return
value are byte-identical to before. When `True`, it additionally returns the full BLS
period/power/depth/duration/transit_time arrays (previously discarded after picking the
top-k candidates), so the dashboard's periodogram can plot the whole curve and let a user
click any point, not just the top 5 peaks.

**Why this is safe:** the new code path only reads already-computed `result.*` arrays
from the existing `BoxLeastSquares.power()` call — it adds no new computation and touches
no existing line before the `return candidates` statement.

**Verification performed before relying on it:**
1. Loaded the pre-edit `periodicity.py` from git `HEAD` side-by-side with the edited
   version in the same process.
2. Compared `detect_periods(...)` output (default call, `detrend=False`, and a
   custom-grid call) on three datasets: a synthetic injected transit, real KIC 757076 Q02
   data, and that same real data with flux shuffled — 9 combinations total. All 9 produced
   **identical** candidate lists (`a == b`) between the original and edited code, and the
   `return_spectrum=True` candidates also matched the original exactly.
3. Checked that the spectrum's maximum-power entry always matches `candidates[0]` exactly
   (one case initially mismatched: a synthetic edge case with a 50 d period-grid boundary
   and 41 BLS grid points tied at the exact same floating-point power value, where
   `argmax` and the candidate-dedup loop are both free to pick any of the tied points —
   confirmed as a floating-point tie, not a bug, and not something that occurs in the real
   KIC 757076 data).
4. This exact process is captured as an offline regression suite in
   `tests/test_periodicity.py` (run with `python -m unittest discover tests`).

**Separately investigated and intentionally left unchanged:** two suspicious-looking
properties of the BLS implementation were found and are **not** fixed by this change,
because fixing them would alter the science without being asked to:

- The duration grid (`DURATION_FRAC_MIN`/`MAX` in `periodicity.py`) is named as if it were
  a *fraction of the period* but is passed to `BoxLeastSquares.power()` as **absolute
  days**, identical at every trial period (0.12 h to 4.8 h regardless of whether the trial
  period is 0.5 d or 50 d). Tested by re-running BLS with a genuinely period-scaled
  duration grid on real Q02 data and on synthetic injected transits (period 1-40 d,
  duration 1.5-12 h): the headline period for KIC 757076 Q02 is unchanged (0.7262 d either
  way), but longer-period injected transits (P >= 10 d) are missed by both the current and
  the "fixed" duration grid in this noise regime — the duration-grid issue is real but is
  not what drives the particular 0.726 d result shown in the dashboard.
- The 7-sigma outlier-interpolation step in `preprocess_light_curve` removes any dip
  deeper than roughly 0.16% for this star **before** BLS ever sees the data (confirmed by
  injecting box transits of increasing depth into real Q02 timestamps: a 0.30%-deep,
  2.5-hour injected transit was completely erased by preprocessing, while a 0.10%-deep one
  survived). The dashboard surfaces this as a caution (`deep_transit_blind`) with the
  exact cutoff computed for the star being viewed.

## 7. Caution flags shown in the dashboard and their derivation

All of the following are computed from the live analysis, not hard-coded
(`src/exosift_pipeline.py:derive_cautions`):

- **`short_sequence`** — CNN input length < 3,197 (the training length).
- **`selection_effect`** — more than one segment was scored; the maximum is shown.
- **`harmonic_ladder`** — >= 3 of the other top BLS peaks are simple multiples/fractions
  (within 2%) of the best period. For KIC 757076 Q02, 4 of 4 other top peaks qualify
  (1.5x, 2x, 2.5x, 4.5x).
- **`long_duration`** — the BLS box spans > 15% of the period (KIC 757076 Q02: 20.4%, which
  is also exactly the upper edge of the duration search grid).
- **`deep_transit_blind`** — informational; states the depth below which preprocessing
  would erase a transit, computed per-star from its own point-to-point scatter.
- **`partial_coverage`** — informational; CNN input is < 60% of the quarter's valid
  observations.
- **`single_quarter`** — informational; only one quarter was analysed.

## 8. Unresolved limitations (not fixed by this work, stated here for the record)

1. Training data remains 37 planets / 5,050 non-planets; test remains 5 planets / 565
   non-planets. No amount of UI work changes this.
2. The BLS duration grid's unit/period-scaling inconsistency (section 6) is unresolved.
3. The repeatability score and the composite score, as implemented, do not separate the
   worked example from shuffled noise or informative evidence respectively; they are
   flagged, not fixed, per this task's instruction not to invent a new scientific
   methodology.
4. Calibration is fit on a different model/length than it is applied to, and on 7
   positives; a proper recalibration would need validation-set predictions from the
   variable-length model itself at 1,925 points, which do not currently exist.
5. BLS significance: no bootstrap, permutation test, or false-alarm-probability estimate
   is computed anywhere in the codebase. "Power" ranks trial periods only.
6. Only Kepler long-cadence data is supported; short-cadence and K2/TESS are out of scope.
