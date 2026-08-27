# Study 4 · Decomposition, expectation and anomaly judgement

Status: **complete.** Built to the design in
`docs/superpowers/specs/2026-08-25-study04-decomposition-and-anomaly-design.md`. Every number in the
report is read from a table body in `outputs/`, so no claim in the prose can drift away from the
artefact behind it.

This study asks three questions of one inclinometer at station 02, over the window that follows the
271-day outage ending on 20 June 2023, and answers each with a qualification it earned:

1. **What is the record made of?** A trend carrying 48.9 % of the variance and spanning 213 mdeg, a
   residual carrying 30.4 %, air temperature 12.2 % and the annual cycle 8.4 %. The learned thermal
   gain is **−2.56 mdeg/°C**, which agrees to within 8 % with Study 03's independent −2.79 measured
   by a method sharing no machinery with this one. The gain survives re-estimation on disjoint thirds
   of the record (−2.47, −2.33, −2.39); the trend does not (+38.6, +4.8, +45.5 mdeg/yr), so the drift
   is quoted once, in-sample, at −2.77 mdeg/yr and never extrapolated.
2. **Is this reading the expected one?** A refitted expectation beats a frozen one decisively — mean
   absolute error 18.2 mdeg against 30.6, bias −10.1 against −28.8 — because a frozen fit goes stale
   rather than noisy. Its conformal 90 % interval nevertheless covers **67.7 %**, which is reported
   rather than corrected. The detector built on that residual is quoted at a measured 122-day
   false-alarm budget and, at that budget, sees only a 16 mdeg growth in the daily swing: a timing
   shift up to two hours and a drift up to 20 mdeg/yr stay invisible.
3. **How far ahead is prediction worth anything?** Answered in section 8 of the report, against
   zero-change, persistence and seasonal-naive baselines with paired block-bootstrap intervals.

Wall temperature and solar radiation are excluded: they cover 17.6 % and 18.8 % of the window,
because the instruments recording them were installed on 21 February 2025 and do not exist for the
first three fifths of it. Carrying them would force every result to be reported twice on
incomparable windows. An extension study may add them.

## Input

`../../data/interim/archive/gubbio_archive_20min.csv` — Study 1's verdict-aware archive product,
read at its native 20-minute cadence. The study reads no external proxies and no raw `.adc` file.

## Outputs

| Artefact | What it holds |
|---|---|
| `NP_01_window_coverage` | Accepted values and coverage per channel over the window |
| `NP_02_gap_inventory` | One row per gap, classified by duration |
| `NP_03_segment_survival` | What each autoregressive window costs in usable runs and windows |
| `NP_04_cadence_evidence` | Why the forecast target is the hourly change |
| `NP_05_component_shares` | Variance share and range of each fitted component |
| `NP_06_learned_gains` | The learned thermal gain against Study 03's measurements |
| `NP_07_nowcast_metrics` | The expectation scored three ways, with interval diagnostics |
| `NP_08_residual_diagnostics` | Ljung–Box, standard deviation and MAD per fit |
| `NP_09_alarm_episodes` | Alarming episodes over the monitored record |
| `NP_10_detectability` | Detectability by mechanism, magnitude and persistence |
| `NP_11_forecast_metrics` | Forecast error by model and horizon |
| `NP_12_skill_vs_baseline` | Paired block-bootstrap skill against three baselines |
| `NP_13_ablation` | The skill increment each predictor bought over the rung below it |
| `NP_14_gap_closure` | Whether accumulated predictions close the gaps they span |
| `NP_15_run_metadata` | Every parameter and library version behind this run |
| `NP_16_component_stability` | The interpretable components re-estimated on disjoint thirds |

Figures `NP_F01` through `NP_F11`, each written as both `.png` and `.svg`. The figure that
would have shown a reconstructed gap is not produced, for the reason section 9 of the report
gives: no gap in this record is scorable, so there is nothing to draw.

## Reproducing

```bash
conda activate neuralprophet_env
jupytext --to ipynb neuralprophet_inclination_prediction_study.py
jupyter nbconvert --to notebook --execute --inplace \
  neuralprophet_inclination_prediction_study.ipynb --ExecutePreprocessor.timeout=14400
```

Two cautions, both learned the hard way during this study. `nbconvert` **exits 0 even when a cell
raises**, so verify a run by reading the printed lines back out of the `.ipynb` rather than by
trusting the exit code. And if `auto_watcher.py` is running it will re-sync the `.py` onto the
`.ipynb` after `nbconvert` writes it, silently leaving newly added cells unexecuted; execute to a
scratch path with `--output-dir` and copy the result back, or stop the watcher first.

Run the tests from `studies/`:

```bash
python 04_neuralprophet_inclination_prediction/tests/test_prediction.py
python 04_neuralprophet_inclination_prediction/tests/test_gaps.py
python 04_neuralprophet_inclination_prediction/tests/test_decomposition.py
python 04_neuralprophet_inclination_prediction/tests/test_folder_honesty.py
python shmlib/tests/test_shmlib.py
python shmlib/tests/test_monitoring.py
```

Rebuild the report after the notebook, from `report/`:

```bash
pdflatex neuralprophet_inclination_prediction_report.tex
pdflatex neuralprophet_inclination_prediction_report.tex
```

All reusable logic lives in `../shmlib/`. This folder holds the notebook, its parameters, its
outputs, its tests and its report — and no library code.
