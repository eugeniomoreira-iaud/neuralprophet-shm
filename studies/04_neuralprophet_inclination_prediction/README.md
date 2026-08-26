# Study 4 · Decomposition, expectation and anomaly judgement

Status: **in progress.** The folder is being rebuilt to the design in
`docs/superpowers/specs/2026-08-25-study04-decomposition-and-anomaly-design.md`. Only artefacts
present in `outputs/` are claimed anywhere; the previous run's notebook and report body were
destroyed before this repository had version control, and its results are not recoverable and are
not cited.

This study asks three questions of one inclinometer at station 02, over the window that follows the
271-day outage ending on 20 June 2023:

1. **What is the record made of?** A NeuralProphet decomposition of the compensated, cleaned
   inclination level into trend, daily cycle, response to the measured environment, and remainder.
2. **Is this reading the expected one?** Given air temperature and relative humidity measured at the
   same instant, an expected inclination with a calibrated interval, and control charts that judge
   the departure.
3. **How far ahead is prediction worth anything?** Forecast skill on the gap-safe hourly change
   against zero-change, persistence and seasonal-naive baselines.

Wall temperature and solar radiation are excluded: they cover 17 % of the window, and carrying them
would force every result to be reported twice on incomparable windows. An extension study may add
them.

## Input

`../../data/interim/archive/gubbio_archive_20min.csv` — Study 1's verdict-aware archive product,
read at its native 20-minute cadence. The study reads no external proxies and no raw `.adc` file.

## Reproducing

```bash
conda activate neuralprophet_env
jupytext --to ipynb neuralprophet_inclination_prediction_study.py
jupyter nbconvert --to notebook --execute --inplace \
  neuralprophet_inclination_prediction_study.ipynb --ExecutePreprocessor.timeout=7200
```

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
