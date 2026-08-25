# Study 4 · NeuralProphet inclination prediction

Status: complete. The [study report](report/neuralprophet_inclination_prediction_report.pdf)
contains the findings; the executed notebook and CSV exports contain the supporting results.

This study asks whether the sensors mounted on the Gubbio structure can estimate the current
hourly inclination change and forecast later changes. The primary response is the gap-safe
one-hour change in raw inclination. It is formed only between adjacent accepted observations from
the same instrument era, so it never bridges missing data or the 2025 instrument change. The
compensated-cleaned change is retained only as a matched sensitivity.

The main result is qualified. Air temperature predicts change much better than zero, but adds less
than one percent skill beyond seasonal time structure in same-time estimation. Forecasts beat a
causal daily-naive baseline through 48 hours, predominantly because of the recent inclination
history. Radiation and wall temperature add short-horizon skill on their much smaller matched
window. Missing-hour changes can be estimated, but their accumulated level reconstruction fails
the gap-closure check and must not replace observed inclination.

## Input and model designs

The only input is `../../data/interim/archive/gubbio_archive_20min.csv`, Study 1's verdict-aware
archive product. The study does not read external proxies or the raw `.adc` archive.

| Design | Predictors | Purpose |
|---|---|---|
| Core | on-structure air temperature and relative humidity | Long-record environmental model |
| Rich | core plus accepted solar radiation and filtered wall temperature | Current-era physical increment |
| Negative control | battery voltage | Acquisition and daily-cycle control |
| Forecast reference | 24 past inclination changes, without physical predictors | Isolate the value of recent response history |

Same-time models use contemporaneous predictor values and no target lags. Forecast models use 24
hours of inclination history and 12 hours of predictor history; they never consume a future
observed predictor. Core horizons are 1, 3, 6, 12, 24, 48, 72 and 168 hours. Rich horizons stop at
24 hours. Five chronological evaluation blocks follow one frozen training origin.

## Contents

| File | Role |
|---|---|
| `neuralprophet_inclination_prediction_study.py` | Editable `py:percent` notebook source |
| `neuralprophet_inclination_prediction_study.ipynb` | Generated and executed notebook |
| `report/neuralprophet_inclination_prediction_report.tex` | Report source |
| `report/neuralprophet_inclination_prediction_report.pdf` | Nine-page result report |
| `tests/test_prediction.py` | Unit and NeuralProphet boundary tests for `shmlib.prediction` |
| `outputs/` | Regenerated figures, predictions, metrics, skill, metadata and gap diagnostics |

The numbered outputs are:

| Output | Content |
|---|---|
| `NP_01_availability.csv` | Accepted hourly coverage by response and predictor |
| `NP_02_fold_inventory.csv` | Chronological train and evaluation blocks |
| `NP_03_predictions.csv` | Every out-of-sample model and baseline prediction |
| `NP_04_metrics_by_fold.csv` | Error and interval scores by evaluation block |
| `NP_05_metrics_pooled.csv` | Pooled scores |
| `NP_06_matched_skill.csv` | Paired 24-hour block-bootstrap skill comparisons |
| `NP_07_run_metadata.csv` | Exact executed lags, horizons, epochs, seed and refit policy |
| `NP_08_missing_hour_estimates.csv` | Experimental prior-only missing-hour changes |
| `NP_09_gap_closure.csv` | Accumulated reconstruction checks with explicit status |
| `NP_10_best_models.csv` | Mechanical minimum-MAE summary by task and tier |
| `NP_F01`--`NP_F05` | Coverage, error, horizon, sensitivity and gap figures in PNG and SVG |

## Reproducing

The committed evaluation results used ten epochs. Run from this folder:

```bash
conda activate neuralprophet_env
jupytext --to ipynb neuralprophet_inclination_prediction_study.py
NP_EPOCHS=10 jupyter nbconvert --to notebook --execute --inplace \
  neuralprophet_inclination_prediction_study.ipynb --ExecutePreprocessor.timeout=3600
```

Set `NP_SMOKE=1 NP_EPOCHS=1` for a fast path check. Set `NP_REFIT_EACH_FOLD=1` to replace the
frozen-origin run with expanding refits. These settings are written to `NP_07_run_metadata.csv`.

Run the study tests from `studies/`:

```bash
python 04_neuralprophet_inclination_prediction/tests/test_prediction.py
```

Rebuild the report after the notebook:

```bash
cd report
pdflatex neuralprophet_inclination_prediction_report.tex
pdflatex neuralprophet_inclination_prediction_report.tex
```

All reusable target construction, gap segmentation, chronological folding, NeuralProphet wrapping,
scoring, bootstrap comparison, baseline and gap-closure logic lives in `../shmlib/prediction.py`.
