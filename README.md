# ExoSift

AI-assisted exoplanet transit-candidate detection using NASA Kepler stellar light curves.

When a planet crosses in front of its host star, it can cause a small dip in measured brightness. ExoSift uses machine learning to identify patterns associated with labelled planet examples and aims to analyse real Kepler observations for possible transit candidates.

> ⚠️ **Scientific caution:** ExoSift produces machine-learning candidate scores, not confirmed exoplanet detections. The training labels come from the Kaggle Kepler exoplanet dataset, which is extremely imbalanced (37 planet / 5,050 non-planet training examples; 5 planet / 565 non-planet test examples). All results should be interpreted with this limitation in mind.

## Project status

**V1 (current):** End-to-end supervised classification pipeline — data inspection, preprocessing, feature engineering, Random Forest baseline, 1D CNN, validation + held-out test evaluation, model comparison, and real-Kepler-target inference via NASA/MAST and Lightkurve.

**V2 (implemented):** Transit periodicity detection (BLS), phase folding, repeated-transit evidence, candidate ranking & explainability, confidence calibration, and an interactive Streamlit UI (`app/streamlit_app.py`).

## Repository contents

| Path | Contents |
|------|----------|
| `src/` | All pipeline scripts |
| `data/raw/` | Kaggle CSV downloads (excluded from git — see `.gitignore`) |
| `data/processed/` | Regenerable preprocessed arrays (excluded from git) |
| `models/` | Trained model weights and evaluation artifacts |
| `app/` | Streamlit UI for interactive Kepler target analysis |
| `notebooks/` | Placeholder for exploratory notebooks (V2) |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/saiyamranjitpatil/exosift.git
cd exosift
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# or: .venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Kaggle dataset

The raw Kaggle Kepler data is **not** included in this repository (the training CSV is ~262 MB and exceeds GitHub file limits).

Download `exoTrain.csv` and `exoTest.csv` from the Kaggle Kepler exoplanet dataset and place them in `data/raw/`:

```
data/raw/exoTrain.csv
data/raw/exoTest.csv
```


## Running the pipeline (V1)

All scripts are in `src/`. Run from the project root:

```bash
# 1. Inspect the raw dataset
python src/inspect_data.py

# 2. Preprocess light curves (normalization, detrending, outlier handling)
python src/preprocessing.py

# 3. Engineer features and build sequences
python src/features.py

# 4. Train Random Forest baseline
python src/train_rf.py

# 5. Train 1D CNN (3,197-point sequences)
python src/train_cnn.py

# 6. Compare Random Forest vs CNN
python src/compare_models.py

# 7. Quick-look light-curve plots
python src/plot_lightcurves.py

# 8. Real-Kepler-target inference (example: KIC 757076)
python src/predict_real_star.py --kic 757076
```

> Note: scripts import from `src/` using flat imports (`from preprocessing import ...`). Run from the project root so `src/` is on `sys.path`, or set `PYTHONPATH=src`.

## Running the app (V2)

```bash
streamlit run app/streamlit_app.py
```

Enter a KIC ID (e.g. `757076`) and click **Analyze Target**. The first run downloads light curves from NASA/MAST and loads TensorFlow, so it can take a minute; repeat analyses of the same target are cached.

## Model architecture — 1D CNN

The CNN is a fixed-weight, variable-length-capable architecture:

1. `Conv1D(32, kernel_size=15)` → BatchNorm → ReLU → MaxPooling1D(4) → Dropout(0.15)
2. `Conv1D(64, kernel_size=11)` → BatchNorm → ReLU → MaxPooling1D(4) → Dropout(0.20)
3. `Conv1D(128, kernel_size=7)` → BatchNorm → ReLU → MaxPooling1D(4) → Dropout(0.25)
4. `GlobalAveragePooling1D` (enables variable-length input)
5. `Dense(64, ReLU)` → Dropout(0.40)
6. `Dense(1, sigmoid)` — planet probability

The original model (`cnn_1d.keras`) was trained on 3,197-point sequences. A variable-length version (`cnn_1d_variable.keras`) was created by recreating the same stack with `Input(shape=(None, 1))` and transferring the exact learned weights by `set_weights`. No retraining or interpolation is performed.

## Evaluation results (V1, Kaggle test set)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Threshold |
|-------|---------|--------|-----------|--------|----|-----------|
| Random Forest (default 0.50) | 0.8931 | 0.2545 | 0.0000 | 0.0000 | 0.0000 | 0.50 |
| Random Forest (validation-selected) | 0.8931 | 0.2545 | 0.1000 | 0.8000 | 0.1778 | 0.1148 |
| 1D CNN (default 0.50) | 0.9982 | 0.7100 | 0.8000 | 0.8000 | 0.8000 | 0.50 |
| 1D CNN (validation-selected) | 0.9982 | 0.7100 | 0.8333 | 1.0000 | 0.9091 | 0.0939 |

> **Important:** The test set contains only **five positive planet examples**. A single example swings recall by 20% and F1 by ~0.18. These metrics are evidence about this specific Kaggle split, not proof of real-world exoplanet discovery capability. Thresholds were selected **only** on validation data — never on the held-out test set.

## Real-star inference example

```
python src/predict_real_star.py --kic 757076
```

- Searches NASA/MAST for Kepler long-cadence observations of KIC 757076.
- Selects a continuous Q02 segment of 1,925 real observations (no interpolation across gaps).
- Preprocesses with the same pipeline as training data.
- Runs the variable-length CNN at the native 1,925-point length.
- Result: planet score 0.567409 at threshold 0.0939 → classified "PLANET CANDIDATE".

> The score is a raw sigmoid output, **not a calibrated probability**, and is **not** an astronomical confirmation. See the `scientific_warning` field in the result JSON.

## Key limitations (read before using results)

1. **Extreme class imbalance:** 37 planet training examples. Metrics are noisy and not conclusive.
2. **Test-set size:** Only 5 positive test examples. Do not treat test metrics as proof of generalization.
3. **Uncalibrated scores:** CNN sigmoid outputs are not calibrated probabilities.
4. **Variable-length CNN unvalidated:** Performance at lengths other than 3,197 (e.g., the 1,925-point real-star segment) has not yet been validated on labelled data. This is Priority A work.
5. **Preprocessing tuned on Kaggle data:** Detrending/normalization parameters (trend window 101, MAD threshold 7.0, clip ±5.0) were tuned on the Kaggle CSV layout and may not be optimal for real Kepler quarters.
6. **V2 periodicity is heuristic:** BLS "power" is a likelihood-based proxy, not a calibrated SDE, and short-period peaks near the 0.5-day search limit can reflect stellar variability or systematics rather than transits. Treat period/repeatability output as supporting evidence only.

## Contributing

This is a learning-oriented, scientifically careful project. Changes are made one logical step at a time, with verification after each. Please open an issue to discuss before proposing significant changes.

## License

[To be added]
