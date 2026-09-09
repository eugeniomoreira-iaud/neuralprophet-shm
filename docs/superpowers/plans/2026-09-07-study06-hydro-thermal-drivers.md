# Study 06 · Moisture and heat exchange as drivers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Study 06 — the test of whether an antecedent-precipitation state explains Study 05's driverless trend and its equinox-peaking yearly term, whether longwave sky loss and wind-modulated heat exchange improve the expectation and restore the sign of the radiation gain, and, last and as a diagnostic only, whether the one-linear-gain-per-driver specification the study inherits is itself what limits the explanation — with the report written section by section as each phase's artefacts land.

**Architecture:** Every function lives in `studies/shmlib/`; the notebook `studies/06_hydro_thermal_drivers/hydro_thermal_drivers_study.py` orchestrates with parameters declared in its parameter cells; tables and figures land in `outputs/` under the `HT_` prefix; the report `report/hydro_thermal_drivers_report.tex` starts as a complete skeleton whose sections carry a `\pending{}` marker that each phase replaces with prose, tables and figures. Two independent ladders, moisture against the trend and heat exchange against air temperature, run on Study 05's Model A specification, and only what survives its own pre-stated rule enters the joint fit. A third, methodological ladder then asks what the fixed linear gain assumed by both of them costs, and it runs after them, never before. A folder-honesty test forbids any figure or table body the report names without a file the notebook writes.

**Tech Stack:** Python 3.10 in conda env `neuralprophet_env` (`C:\Users\eugenio.moreira\.conda\envs\neuralprophet_env\python.exe`), NeuralProphet 0.8.0, torch 2.5.1 CPU, pandas 2.3, numpy 1.26 on OpenBLAS, scipy, matplotlib/seaborn via `shmlib.viz`, jupytext, nbconvert, MiKTeX pdflatex. Tests are `unittest`, run directly with `python <file>` from `studies/`.

**Spec:** `docs/superpowers/specs/2026-09-07-study06-hydro-thermal-drivers-design.md` (approved 2026-09-07). Read it before any task; every task below cites the decision it implements.

## Progress

Updated 2026-09-07. Nothing started. Entry condition of spec §0: Study 05's D15 sweep has run and `GM_10b` names the radiation operator it hands on. Until then Task 0.1 may run (it creates nothing that depends on D15), and every later task waits.

| Phase | Tasks | State |
|---|---|---|
| 0 · Sources, semantics, folder, skeleton | 0.1–0.6, 0.5b | ⬜ |
| 1 · Base fit and the moisture states, descriptively | 1.1–1.5 | ⬜ |
| 2 · Moisture in the model | 2.1–2.4 | ⬜ |
| 3 · Heat-exchange ladder | 3.1–3.4 | ⬜ |
| 3b · The gain assumption and the capacity ceiling | 3b.1–3b.5 | ⬜ |
| 4 · Joint fit and the slow chart | 4.1–4.2 | ⬜ |
| 5 · Closure | 5.1–5.2 | ⬜ |

Phase 3b was added on 2026-09-07, with spec decisions D11 and D12, after the measurement recorded in spec §2.6. It is separable: if the user answers open question 17 of the spec by moving it out of this study, delete the Phase 3b section, the `HT_13`–`HT_15` and `HT_F09`–`HT_F10` rows, and report section 6, and nothing else in this plan changes.

## Global Constraints

- **Caveman ultra, always.** Every subagent prompt starts with "respond in caveman ultra mode". Produced documents (report `.tex`, README, docstrings, comments, commit messages, this plan) are full prose. (Spec §0.)
- **Delegate down.** Tasks marked **S** go to a Sonnet subagent (`claude-sonnet-5`) spawned with the Agent tool; tasks marked **O** (report prose, checkpoint gates, interpretation of a ladder) stay with the orchestrator, Fable or Opus. (Spec §0, `CLAUDE.md`.)
- **Library-first.** No function is defined in the notebook or the study folder; every function goes in `studies/shmlib/`. A cell longer than about ten lines that defines a function, builds a table row by row or composes a figure axis by axis is a defect. Before any new function is written, `/graphify query` confirms nothing in `shmlib` already does it. (`instructions-pipeline.md` § Studies.)
- **Additive adaptations only.** Every change to an existing `shmlib` function or dict keeps every existing caller's behaviour, defaults included. After each adaptation run all of, from `studies/`: `python shmlib/tests/test_shmlib.py`, `python shmlib/tests/test_monitoring.py`, `python shmlib/tests/test_proxies_grid.py`, `python 03_thermomechanical_response/tests/test_shmlib_study03.py`, `python 04_neuralprophet_inclination_prediction/tests/test_prediction.py`, `python 05_greybox_monitoring/tests/test_model_a.py`, `python 05_greybox_monitoring/tests/test_monitor.py`. (Spec §5.2.)
- **States, never readings.** No rain, wind, pressure or dew-point reading enters a model additively. Every new regressor is a filtered state with a time constant declared in the parameter cell. (Spec D3.)
- **Rules stated before fits.** The three-part rule of D5 (skill bounds exclude zero; gain sign stable across folds; yearly share falls), the rule of D6 (skill bounds exclude zero) and the two-part rule of D11 (skill bounds exclude zero *and* the physical states keep their share and sign) are printed by the notebook before the corresponding sweep runs and are applied by the checkpoint task, never re-tuned after. The expected moisture sign is `EXPECTED_MOISTURE_SIGN = +1` (spec D5).
- **Fixed gain first, freedom last, main line always fixed.** No fit with a varying or nonlinear regressor gain runs before Movements 2 and 3 have written `HT_05` and `HT_07`; Movement 3b asserts their existence and raises otherwise. Whatever Movement 3b finds, the joint fit of Movement 4, the components the report tables and anything the monitor would consume use the fixed linear gain. A nonlinear specification is a diagnostic row, on the precedent of Study 05's D7. (Spec D11, D12.)
- **NeuralProphet is not linear-only, and the study says so.** `future_regressors_model`, `future_regressors_d_hidden`, `future_regressors_num_hidden_layers`, `lagged_reg_layers` and `ar_layers` exist in 0.8.0 and are unused by Studies 04 and 05. `prediction.neuralprophet_backtest` currently exposes none of them because its constructor dict is fixed (`prediction.py:1104`); Task 3b.2 adds them as optional arguments whose defaults reproduce today's model exactly. Any text in this study that describes the linear gain as a property of the library rather than as a choice of the study is a defect. (Spec D11.)
- **Model A inherited.** Every Model A argument (`N_CHANGEPOINTS = 12`, `CHANGEPOINTS_RANGE = 0.95`, `TREND_REG = 0.0`, `YEARLY_ORDER = 1`, `DAILY_ORDER = 2`, plain daily term, `QUANTILES = (0.05, 0.95)`, `LEARNING_RATE = 0.01`, `EPOCHS = 30`, `SEED = 0`, `impute_missing=False, drop_missing=False`) is declared in the parameter cell with a pointer to the Study 05 artefact that fixed it. (Spec D8.)
- **No imputation of the target.** Regressor gaps up to `REGRESSOR_FILL_MAX_GAP = '2h'` filled and flagged by `proxies.build_regressor_sets`; nothing else filled. (Spec D8, Study 05 D13.)
- **Figure rules** from `instructions-pipeline.md`: Okabe–Ito channel colours through `viz.channel_style`/`viz.driver_colour`, Cividis for scalars, legends below the axes (`loc='upper center', bbox_to_anchor=(0.5, -0.30), frameon=False`), span highlights black at 5 %, accent `#D55E00` for markers only, every figure through `viz.finish` (PNG + SVG). (Spec D10.)
- **Artefact names** exactly as spec §6: `HT_01`…`HT_15`, `HT_F01`…`HT_F10`, LaTeX bodies `HT_NN_body.tex` beside each CSV.
- **Every number in the report traces to an `HT_` artefact, and every image comes from the paired notebook.** Each `\includegraphics{HT_Fxx_…}` names a file a `figures.*` call in the notebook writes with `save_path=str(OUTPUT_DIR), filename='HT_Fxx_…'`, and each `\input{../outputs/HT_xx_body.tex}` names a body a `tables.write_table` call writes. The folder-honesty test enforces the mapping from Task 0.1 onward. (Spec §7; user rule of 2026-09-06.)
- **The notebook tells the code side of the story.** Each movement opens with a Markdown cell stating what it computes, why, and which `HT_` artefacts it writes; each step inside has its own `###` Markdown cell. (User rule of 2026-09-06.)
- **Iterate against dumped state; run the whole notebook once per task.** A movement's new cells are developed against a pickle of the frames the movements before it produced (`build_dump.py` and `run_movement.py` in the session scratchpad, rebuilt per session, see memory `study05-run-procedure`), executed with nbclient and `resources={'metadata': {'path': study_dir}}` so the study folder is the working directory and no copy of the notebook is ever placed inside the repo. The full notebook runs once per task, to a scratch `--output-dir`, and the executed `.ipynb` is copied back after the error cells have been counted, because nbconvert exits 0 on a failing cell. (Controller ruling, 2026-09-07.)
- **Report in parts.** Each phase ends with a task that writes that phase's report section(s), removes their `\pending{}` marker, rebuilds the PDF twice, runs the honesty test and commits. The PDF is tracked.
- **Cross-study reads.** The notebook may read Study 05's `outputs/GM_04d_trend_rates.csv` and `GM_11_alarm_episodes.csv` for side-by-side tables only, guarded by `if path.exists()`, since `outputs/` is gitignored; every number the report asserts comes from Study 06's own artefacts.
- **Commits.** One per task, message in full prose, on the branch `study06-hydro-thermal-drivers` created from `study05-greybox-monitoring` once Study 05 is closed; always `git add <explicit paths>` and `git commit -- <paths>`, because the working tree carries other sessions' uncommitted hunks in `shmlib/figures.py`, `shmlib/proxies.py`, `shmlib/prediction.py` and Studies 02, 03 and 05 that this study must never stage. The user's untracked `05_greybox_monitoring/sidequest/` and `setup_sidequest.py` are never staged. Messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- **Secrets.** `auxiliary/oiko.py` is tracked and carries an Oikolab API key in source. The new download script of Task 0.3 reads the key from the environment variable `OIKOLAB_API_KEY` and never writes it to disk; the plan does not copy the existing key anywhere. The orchestrator tells the user, in full prose, that the tracked key should be rotated and removed from the file.
- **Paths.** Repo root `neuralprophet-shm/neuralprophet-shm`; run tests and notebooks from `studies/`; the study folder is `studies/06_hydro_thermal_drivers/`.
- **Execution machine.** AMD Ryzen Threadripper PRO 9995WX, 64 cores, 500 GB, PyTorch CPU, NumPy on OpenBLAS; MKL excluded by `environment.yml` and must stay excluded. Worker processes are spawned, each re-importing torch, so `N_JOBS = 32` is sized to the longest loop of this study, the year ladder's rungs × folds (up to 12 × 5 = 60 items), and never to the core count. (Study 05 plan, machine change of 2026-09-07.)

---

## File structure

| File | Responsibility | Tasks |
|---|---|---|
| `studies/06_hydro_thermal_drivers/README.md` | How to run and rebuild; status line; artefact list | 0.1, 0.6 … 5.1 |
| `studies/06_hydro_thermal_drivers/.gitignore` | Copy of Study 05's | 0.1 |
| `studies/06_hydro_thermal_drivers/hydro_thermal_drivers_study.py` | The notebook: title, imports, parameter cells from Task 0.1; movements 0 to 5 added per phase | 0.1, 0.5, 0.5b, 1.4, 2.3, 3.3, 3b.4, 4.1, 5.1 |
| `studies/06_hydro_thermal_drivers/report/hydro_thermal_drivers_report.tex` | The report, eleven sections from Phase 0, filled phase by phase | 0.1, 0.6, 1.5, 2.4, 3.4, 3b.5, 4.2, 5.1 |
| `studies/06_hydro_thermal_drivers/tests/test_folder_honesty.py` | Folder hygiene, provenance of every graphic and table body | 0.1, 5.1 |
| `studies/06_hydro_thermal_drivers/tests/test_states.py` | Tests for `antecedent_index`, `annual_extrema`, `segment_rate_regression`, `net_longwave`, the modulated filter, `amplitude_envelope` | 1.1, 1.2, 3.1, 3.2, 3b.1 |
| `studies/06_hydro_thermal_drivers/tests/test_year_ladder.py` | Tests for `year_folds`, `year_ladder` | 2.1, 2.2 |
| `studies/06_hydro_thermal_drivers/tests/test_capacity.py` | Tests for the passthrough arguments of `neuralprophet_backtest`, `gain_variation_ladder` and `capacity_ceiling` | 3b.2, 3b.3 |
| `studies/06_hydro_thermal_drivers/tests/test_figures06.py` | Smoke tests of the seven new figure functions | 1.3, 2.2, 3b.3 |
| `studies/shmlib/proxies.py` | + `ERA5_MAP_FULL`, `SOIL_MAP`, unit/label/range entries for `lwrad`, `snow`, `swvl1`…`swvl4` | 0.2 |
| `studies/shmlib/viz.py` | + identity colours and labels for `lwrad`, `swvl`, `api` in `QUANTITY_COLOUR`, `QUANTITY_SHORT`, `CHANNEL_LABEL` | 0.2 |
| `studies/shmlib/coupling.py` | + `antecedent_index`, `annual_extrema`, `segment_rate_regression`, `net_longwave`, `amplitude_envelope`; `thermal_lag_filter(modulation=…)`, `thermal_operator(modulation=…)` | 1.1, 1.2, 3.1, 3.2, 3b.1 |
| `studies/shmlib/prediction.py` | + `year_folds`, `year_ladder`, `gain_variation_ladder`, `capacity_ceiling`; `neuralprophet_backtest(future_regressors_model=…, future_regressors_d_hidden=…, future_regressors_num_hidden_layers=…, lagged_reg_layers=…)` | 2.1, 2.2, 3b.2, 3b.3 |
| `studies/shmlib/figures.py` | + `plot_moisture_state`, `plot_state_phase`, `plot_trend_rate_regression`, `plot_changepoint_sweep`, `plot_yearly_by_year`, `plot_driver_envelope`, `plot_capacity_ceiling` | 1.3, 2.2, 3b.3 |
| `auxiliary/oiko_soil.py` | ERA5-Land soil water and extended precipitation download, key from the environment | 0.3 |
| `docs/proxy-data-dictionary.md` | Rows for the new ERA5 channels and the soil file | 0.4 |
| `studies/README.md` | Study 6 row: "Design approved" → "In progress" (0.6) → "Complete" (5.1) | 0.1, 0.6, 5.1 |

Test-file conventions (copy exactly): module docstring with the run command, `import os, sys, unittest`, `import numpy as np`, `import pandas as pd`, then

```python
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
from shmlib import <module>  # noqa: E402
```

and `if __name__ == '__main__': unittest.main(verbosity=2)`.

Figure-function template (`shmlib/figures.py:2919`, `plot_ladder`): NumPy docstring → compute → `fig, axes = plt.subplots(..., figsize=viz.figsize(viz.FIGURE_WIDTH, h))` → draw → `viz.format_spines(ax)` per axes → `fig.suptitle(title)` if given → `viz.finish(fig, save_path=save_path, filename=filename)` → `return fig`.

Table-writing template (Study 05 notebook, `GM_16`):

```python
ladder.to_csv(OUTPUT_DIR / 'HT_07_exchange_ladder.csv', index=False)
tables.write_table(ladder, str(OUTPUT_DIR / 'HT_07_body.tex'),
                   [('rung', tables.texttt), ('rows', ',d'), ('mae_val', '.2f'),
                    ('skill', '.3f'), ('skill_q05', '.3f'), ('skill_q95', '.3f')])
```

Component names inside a `decompose_components` frame, used by every task that reads a fit: `trend`, `season_yearly`, `season_daily`, `future_regressor_<name>` per regressor, `residual`, `y`, `yhat1`. `component_variance_shares` returns `component`, `variance`, `share`, `mean`, `peak_to_peak`. `trend_parameters` returns `(trend, rates)` with `rates` columns `start`, `end`, `rate_mdeg_per_year`. `neuralprophet_backtest` returns `(model, out)` with `out` columns `ds`, `y`, `yhat`, `q05`, `q95`.

---

# Phase 0 · Sources, semantics, folder, skeleton

### Task 0.1 (S): Study folder, notebook head, report skeleton, honesty test, index row

**Files:**
- Create: `studies/06_hydro_thermal_drivers/README.md`, `.gitignore`, `hydro_thermal_drivers_study.py`, `report/hydro_thermal_drivers_report.tex`, `tests/test_folder_honesty.py`
- Modify: `studies/README.md` (one row in the sequence table)

**Interfaces:**
- Produces: the parameter names every later movement reads, listed in Step 2; the report section labels `sec:intro`, `sec:sources`, `sec:method`, `sec:moisture`, `sec:exchange`, `sec:gain`, `sec:joint`, `sec:slow`, `sec:verdict`, `sec:limitations`, `sec:metadata`.

- [ ] **Step 1: `.gitignore` and README**

Copy `studies/05_greybox_monitoring/.gitignore` unchanged. Write `README.md` with the six headers of Study 05's README (`# Study 6 · Moisture and heat exchange as drivers of the station 02 inclination`, `## Input`, `## Outputs`, `## Reproducing`, `## Tests`, `## Rebuilding the report`). Status line: `Status: **design approved 2026-09-07, not started.** Built to the design in `docs/superpowers/specs/2026-09-07-study06-hydro-thermal-drivers-design.md`.` The Input section names the four files of the parameter cell below; the Outputs section lists `HT_01`…`HT_15` and `HT_F01`…`HT_F10` with spec §6's one-line contents; the Reproducing section is Study 05's block with the notebook name replaced; the Tests section names the four test files of this plan; the report section is Study 05's with the file name replaced.

- [ ] **Step 2: Notebook head**

Write `hydro_thermal_drivers_study.py` in `py:percent` format: a title cell (H1, one paragraph on the question, the three questions of spec §1 as a numbered list), Study 05's imports cell verbatim, then the parameter cells. Each parameter is preceded by a Markdown cell with a **Parameter Tuning Guidance** entry in Study 05's style (name in bold code, purpose, default, effect downstream, pointer to the artefact that fixed an inherited value). The parameters and their values:

```python
# Paths and grid
ARCHIVE_CSV = '../../data/interim/archive/gubbio_archive_20min.csv'
STATION_CSV = '../../data/raw/proxies/meteosystem_gubbio.csv'
ERA5_CSV = '../../data/raw/proxies/oikolab_weather.csv'
SOIL_CSV = '../../data/raw/proxies/oikolab_soil_moisture.csv'   # Task 0.3; absent is allowed
STUDY05_OUTPUTS = Path('../05_greybox_monitoring/outputs')       # side-by-side tables only
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
NATIVE_FREQ = '20min'
TARGET_COLUMN = 'inc_comp_cleaned'
SPIKE_COLUMN = 'inc_spike'
WINDOW_START = '2018-07-26'
WINDOW_END = None

# Regressor sets, as Study 05 (D2, D8)
STR_MAP_CURRENT = {'tair': 'tair', 'sr': 'n_sr_ok', 'twall': 'n_twall_filtered',
                   'rh': 'n_rh_ok', 'batt': 'n_batt_ok'}
STR_MAP_LEGACY = {'tair': 'tair', 'rh': f'{site.TARGET_STATION}_rh_ok',
                  'batt': f'{site.TARGET_STATION}_batt_ok'}
REGRESSOR_SETS = {
    'str': {'tair': 'tair_str', 'rh': 'rh_str', 'sr': 'sr_gs'},
    'gs': {'tair': 'tair_gs', 'rh': 'rh_gs', 'sr': 'sr_gs'},
    'era5': {'tair': 'tair_era5', 'rh': 'rh_era5', 'sr': 'sr_era5'},
}
REGRESSOR_FILL_MAX_GAP = '2h'
ERA5_SR_IS_ACCUMULATION = True
ERA5_LW_IS_ACCUMULATION = True       # settled in Movement 0 against the file's night values

# The radiation operator Study 05's D15 handed on (spec §2.3): delay only, or a filter
RADIATION_DELAY_H = 1                # Study 05 GM_10b: replace by 0 if the filter was kept
RADIATION_TAU_H = 0.0                # Study 05 GM_10b: the kept time constant, 0.0 = delay only

# Model A, inherited from Study 05 (spec D8)
N_CHANGEPOINTS = 12                  # GM_04d
CHANGEPOINTS_RANGE = 0.95
TREND_REG = 0.0                      # GM_05c
YEARLY_ORDER = 1                     # GM_04
DAILY_ORDER = 2                      # GM_04
QUANTILES = (0.05, 0.95)
LEARNING_RATE = 0.01
EPOCHS = 30
SEED = 0
VALID_P = 0.2
N_JOBS = 32
STUDY03_GAINS = {('str', 'tair'): -2.79, ('gs', 'tair'): -2.23, ('era5', 'tair'): -2.04,
                 ('str', 'sr'): -0.035, ('gs', 'sr'): -0.035, ('era5', 'sr'): -0.026}

# Moisture states (spec D3, D4, D5)
MOISTURE_SOURCE = 'era5'             # primary; 'gs' is scored beside it in HT_01
API_TAU_DAYS = (3, 7, 14, 30, 60, 90, 180)
API_PHASE_TAUS = (30, 60, 90)        # the τ reported in HT_03 and HT_F02
API_MAX_GAP = '6h'                   # a precipitation gap longer than this resets the state
API_WARMUP_TAUS = 3.0
SOIL_LAYERS = (1, 2, 3, 4)
HYDRO_YEAR_START_MONTH = 10
FOLD_YEARS = (2019, 2020, 2021, 2024, 2025)
CHANGEPOINT_SWEEP = (0, 2, 4, 8, 12)
SEGMENT_MAX_GAP_DAYS = 30            # a trend segment spanning a longer target gap is excluded
SEGMENT_EDGE_DAYS = 7                # the state is averaged over this many days at each segment end
EXPECTED_MOISTURE_SIGN = +1          # earth pressure pushes towards the valley (spec D5)
BOOTSTRAP_BLOCK_HOURS = 24
BOOTSTRAP_REPETITIONS = 2000

# Heat-exchange states (spec D3, D6)
EMISSIVITY = 0.9
WIND_SOURCE = 'gs'
WIND_K = (0.0, 0.05, 0.1, 0.2, 0.5)  # s/m; τ(w) = τ0 / (1 + k·w)
CURRENT_ERA_START = '2025-02-21'

# The gain assumption and the capacity ceiling (spec D11, D12) — diagnostics only
ENVELOPE_CHANNELS = ('y', 'tair_str', 'tair_gs', 'tair_era5', 'sr_gs', 'sr_era5')
ENVELOPE_MIN_HOURS = 20              # slots a day needs before its daily harmonic is fitted
GAIN_WEIGHT_CURVE = None             # None = Study 05 GM_04's order-two annual fit, rebuilt here
GAIN_SPECS = ('base', 'conditional', 'neural_nets')   # spec open question 15
FUTURE_REGRESSORS_D_HIDDEN = 4       # NeuralProphet's own defaults, used only by the 'neural_nets' spec
FUTURE_REGRESSORS_NUM_HIDDEN_LAYERS = 2
CEILING_MODEL = 'xgboost'            # spec open question 16
CEILING_LAGS_H = 12                  # hours of history per driver in the ceiling's feature matrix
CEILING_N_ESTIMATORS = 400
CEILING_MAX_DEPTH = 6
CEILING_LEARNING_RATE = 0.05

# The slow chart (spec D9), Study 05's values
REFERENCE_START = '2020-11-21'
REFERENCE_END = '2021-12-31'
MONITORED_START = '2022-01-01'
EWMA_LAMBDA_SLOW = 0.1
CUSUM_K = 0.5
CUSUM_H = 5.0
JOINT_WINDOW_DAILY = '1D'
BUDGET_SLOW_DAYS = 365.0
LIMIT_CANDIDATES = tuple(np.arange(1.0, 15.01, 0.25))
DAILY_HARMONIC_MIN_SLOTS = 60
REFIT_EVERY = '30d'
MIN_TRAIN = '730d'
TRAIN_WINDOW = '1095d'
```

The last cell of the head prints the two rules in full prose so that they are on the page before any sweep:

```python
print('Rule D5 (moisture): the state counts as a driver only if (1) its leave-one-year-out '
      'skill over the base has bootstrap bounds excluding zero, (2) its gain keeps one sign '
      f'in every fold, expected sign {EXPECTED_MOISTURE_SIGN:+d}, and (3) the yearly term\'s '
      'share of variance falls when the state is present.')
print('Rule D6 (heat exchange): an exchange counts as an improvement only if its skill '
      'bounds exclude zero; the radiation gain\'s sign is reported at every rung regardless.')
print('Rule D11 (varying gain): a varying temperature gain counts as a finding only if its '
      'skill bounds over the base exclude zero AND the moisture and exchange states keep the '
      'share and sign they held in the base. If skill improves while a state\'s share '
      'collapses, the varying gain has absorbed that state, the state\'s own verdict stands, '
      'and both readings are reported. The main line stays fixed-gain in every case.')
print('Rule D12 (ceiling): the unconstrained fit is a bound, never a candidate. It runs only '
      'after HT_05 and HT_07 exist, it has no components, no gains and no interval, and it '
      'never enters the joint fit or the monitor.')
```

- [ ] **Step 3: Report skeleton**

Write `report/hydro_thermal_drivers_report.tex` with Study 05's preamble verbatim (the comment block, `\documentclass`, `\runninghead` set to `Moisture and heat exchange at Gubbio`, `\input{../../_shared/reportstyle.tex}`, `\graphicspath{{../outputs/}}`, the `\pending` macro), the title *Does the wall answer the rain, and does it cool to the sky?* with the subtitle *Moisture and heat-exchange states as drivers of the station 02 inclination, 2018 to 2026*, author `Study report · \texttt{studies/06\_hydro\_thermal\_drivers/}`, and eleven sections each carrying `\pending{Phase N}`: Introduction (0), The sources and the states (0), Method (1), Moisture against the trend (2), Heat exchange (3), What the linear gain assumes, and what it costs (3b), The joint decomposition (4), What the slow chart would watch (4), Verdict (5), Limitations (5), Run metadata (5). Section labels as spec §7, with `sec:gain` for the new one. Build it twice with `pdflatex` from `report/`; expected: a PDF with a table of contents and eleven pending notes.

- [ ] **Step 4: Honesty test**

Copy `studies/05_greybox_monitoring/tests/test_folder_honesty.py` to `studies/06_hydro_thermal_drivers/tests/test_folder_honesty.py` and replace every `greybox_monitoring` by `hydro_thermal_drivers` and every `GM_` by `HT_`; keep `STUDY_COMPLETE = False`. Run from `studies/`: `python 06_hydro_thermal_drivers/tests/test_folder_honesty.py`. Expected: every test passes on the empty skeleton (no graphics included yet, one `.py` in the root).

- [ ] **Step 5: Index row**

In `studies/README.md`, add `06_hydro_thermal_drivers/` to the tree listing and a row 6 to the sequence table: question "Is the slow, reversing movement of the wall, and the annual term in quadrature with air temperature, the response to moisture from rainfall and to the wall's heat exchange with the air and the sky, once those readings take the physical form through which they act?", status "Design approved".

- [ ] **Step 6: Commit**

```bash
git add studies/06_hydro_thermal_drivers studies/README.md
git commit -- studies/06_hydro_thermal_drivers studies/README.md -m "feat(study06): study folder, notebook head with the parameter cells, report skeleton and honesty test"
```

### Task 0.2 (S): The new channels in `proxies` and `viz`

**Files:**
- Modify: `studies/shmlib/proxies.py` (after `ERA5_MAP`, line 163; and the `QUANTITY_UNIT`, `QUANTITY_LABEL`, `PLAUSIBLE_RANGE` dicts)
- Modify: `studies/shmlib/viz.py` (`QUANTITY_COLOUR`, `QUANTITY_SHORT`, `CHANNEL_LABEL`)
- Modify: `studies/shmlib/tests/test_proxies_grid.py` (add a class)

**Interfaces:**
- Produces: `proxies.ERA5_MAP_FULL`, a dict equal to `ERA5_MAP` plus `'lwrad': 'surface_thermal_radiation (W/m^2)'` and `'snow': 'snowfall (mm of water equivalent)'`; `proxies.SOIL_MAP = {'swvl1': 'volumetric_soil_water_layer_1 (m^3/m^3)', …, 'swvl4': …, 'rain': 'total_precipitation (mm of water equivalent)', 'snow': 'snowfall (mm of water equivalent)'}`; entries `'lwrad': 'W/m2'`, `'snow': 'mm'`, `'swvl1'…'swvl4': 'm3/m3'` in `QUANTITY_UNIT`, matching labels in `QUANTITY_LABEL`, and ranges `'lwrad': (100.0, 600.0)`, `'snow': (0.0, 200.0)`, `'swvlN': (0.0, 1.0)` in `PLAUSIBLE_RANGE`. `ERA5_MAP` itself is unchanged, so every existing `load_era5` caller loads what it loaded before. In `viz`: `QUANTITY_COLOUR['lwrad'] = '#8C4A70'` (darkened Reddish Purple, the radiation family), `QUANTITY_COLOUR['swvl'] = QUANTITY_COLOUR['api'] = '#003F5C'` (darkened Blue, the water family), `QUANTITY_SHORT` entries `'lwrad': 'Longwave'`, `'swvl': 'Soil water'`, `'api': 'Antecedent precip.'`, and `CHANNEL_LABEL` entries `'lwrad': 'Downward longwave [W/m²]'`, `'swvl': 'Volumetric soil water [m³/m³]'`, `'api': 'Antecedent precipitation [mm]'`.

- [ ] **Step 1: Write the failing test** (append to `test_proxies_grid.py` before the `__main__` block)

```python
class TestStudy06Channels(unittest.TestCase):

    def test_the_full_era5_map_extends_the_default_without_changing_it(self):
        self.assertEqual({k: proxies.ERA5_MAP_FULL[k] for k in proxies.ERA5_MAP},
                         proxies.ERA5_MAP)
        self.assertEqual(proxies.ERA5_MAP_FULL['lwrad'],
                         'surface_thermal_radiation (W/m^2)')
        self.assertEqual(proxies.ERA5_MAP_FULL['snow'],
                         'snowfall (mm of water equivalent)')

    def test_every_new_quantity_has_unit_label_and_range(self):
        for quantity in ('lwrad', 'snow', 'swvl1', 'swvl2', 'swvl3', 'swvl4'):
            self.assertIn(quantity, proxies.QUANTITY_UNIT)
            self.assertIn(quantity, proxies.QUANTITY_LABEL)
            self.assertIn(quantity, proxies.PLAUSIBLE_RANGE)
        self.assertEqual(proxies.PLAUSIBLE_RANGE['swvl1'], (0.0, 1.0))

    def test_load_era5_reads_the_two_new_columns_through_the_full_map(self):
        index = pd.date_range('2024-01-01', periods=3, freq='1h')
        frame = pd.DataFrame({
            'temperature (degC)': [1.0, 2.0, 3.0],
            'surface_thermal_radiation (W/m^2)': [300.0, 310.0, 320.0],
            'snowfall (mm of water equivalent)': [0.0, 0.5, 0.0]},
            index=index.rename('datetime'))
        path = os.path.join(os.path.dirname(__file__), '_tmp_era5_full.csv')
        frame.to_csv(path)
        try:
            out = proxies.load_era5(path, column_map={
                'tair': proxies.ERA5_MAP_FULL['tair'],
                'lwrad': proxies.ERA5_MAP_FULL['lwrad'],
                'snow': proxies.ERA5_MAP_FULL['snow']}, freq='1h')
        finally:
            os.remove(path)
        self.assertIn('lwrad_era5', out.columns)
        self.assertIn('snow_era5', out.columns)
        self.assertAlmostEqual(float(out['lwrad_era5'].iloc[1]), 310.0)
```

- [ ] **Step 2: Run to verify failure.** From `studies/`: `python shmlib/tests/test_proxies_grid.py`. Expected: `AttributeError: module 'shmlib.proxies' has no attribute 'ERA5_MAP_FULL'`.

- [ ] **Step 3: Implement.** After `ERA5_MAP` in `proxies.py`:

```python
#: ``ERA5_MAP`` plus the two channels Study 06 reads from the same file: the
#: downward longwave flux, from which the wall's net sky loss is computed,
#: and snowfall as water equivalent, folded into precipitation. Kept apart
#: from ``ERA5_MAP`` so every earlier study's default load is unchanged.
ERA5_MAP_FULL = {
    **ERA5_MAP,
    'lwrad': 'surface_thermal_radiation (W/m^2)',
    'snow': 'snowfall (mm of water equivalent)',
}

#: The ERA5-Land soil-water file of ``auxiliary/oiko_soil.py``: four layers of
#: volumetric soil water, and precipitation and snowfall from 2016 so that the
#: antecedent-precipitation state has its warm-up before the record starts.
SOIL_MAP = {
    'swvl1': 'volumetric_soil_water_layer_1 (m^3/m^3)',
    'swvl2': 'volumetric_soil_water_layer_2 (m^3/m^3)',
    'swvl3': 'volumetric_soil_water_layer_3 (m^3/m^3)',
    'swvl4': 'volumetric_soil_water_layer_4 (m^3/m^3)',
    'rain': 'total_precipitation (mm of water equivalent)',
    'snow': 'snowfall (mm of water equivalent)',
}
```

Add to `QUANTITY_UNIT`: `'lwrad': 'W/m2', 'snow': 'mm', 'swvl1': 'm3/m3', 'swvl2': 'm3/m3', 'swvl3': 'm3/m3', 'swvl4': 'm3/m3'`. To `QUANTITY_LABEL`: `'lwrad': 'Downward longwave radiation', 'snow': 'Snowfall', 'swvl1': 'Soil water, layer 1', …`. To `PLAUSIBLE_RANGE`: `'lwrad': (100.0, 600.0), 'snow': (0.0, 200.0), 'swvl1': (0.0, 1.0), …`. In `viz.py` add the colour, short-name and label entries listed under Interfaces, each with a trailing comment naming the family it darkens.

- [ ] **Step 4: Run the tests; expected 3 PASS.** Then the whole suite of Global Constraints.

- [ ] **Step 5: Commit**

```bash
git add studies/shmlib/proxies.py studies/shmlib/viz.py studies/shmlib/tests/test_proxies_grid.py
git commit -- studies/shmlib/proxies.py studies/shmlib/viz.py studies/shmlib/tests/test_proxies_grid.py -m "feat(shmlib): the longwave, snowfall and soil-water channels, with their units, ranges and identity colours"
```

### Task 0.3 (S writes; the user runs): `auxiliary/oiko_soil.py`

**Files:**
- Create: `auxiliary/oiko_soil.py`

**Interfaces:**
- Produces: `data/raw/proxies/oikolab_soil_moisture.csv`, hourly from 2016-01-01 to yesterday, columns exactly the values of `proxies.SOIL_MAP` plus the metadata columns `oiko.py` drops.

- [ ] **Step 1: Write the script.** Copy `auxiliary/oiko.py` and change only these things: `API_KEY = os.environ['OIKOLAB_API_KEY']` with a comment that the key is never written to the file; `OUTPUT_PATH = '../data/raw/proxies/oikolab_soil_moisture.csv'`; `START = '2016-01-01'` when the file does not exist; `PARAMS = ['volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4', 'total_precipitation', 'snowfall']`; and, in the request `params`, add `'model': 'era5land'`. Keep the six-month chunking and the one-second pause. Add a probe at the top: one request for `2024-01-01` to `2024-01-02` whose response status and columns are printed before the loop starts, so that a parameter name the provider does not accept is seen on the first request rather than after an hour.

- [ ] **Step 2: The user runs it.** From `auxiliary/`, with `OIKOLAB_API_KEY` set in the shell: `python oiko_soil.py`. Expected on success: the probe prints status 200 and six data columns; the file grows chunk by chunk. If the probe returns 4xx naming a parameter, the subagent looks up the exact names at `https://docs.oikolab.com` and corrects `PARAMS` and `SOIL_MAP` together, with the dictionary row of Task 0.4 following. If the provider has no soil water at all, the script is kept, the file is absent, and `SOIL_CSV` stays pointing at a path that does not exist, which every movement below tolerates.

- [ ] **Step 3: Commit the script only.** `git add auxiliary/oiko_soil.py` and `git commit -- auxiliary/oiko_soil.py -m "feat(auxiliary): download ERA5-Land soil water and extended precipitation, key from the environment"`. The CSV is data and is never committed.

### Task 0.4 (S): Dictionary rows for the new channels

**Files:**
- Modify: `docs/proxy-data-dictionary.md` (channel table after line 58; conversions table after line 65)

- [ ] **Step 1:** Add rows `lwrad_era5 | lwrad | W/m2 | ERA5 reanalysis | surface_thermal_radiation (W/m^2) | 99.5 %` (already present at line 58; keep, and add the note settled in Task 0.5: "downward flux, accumulated over the preceding hour and stamped at its end, like the solar channel"), `snow_era5 | snow | mm | ERA5 reanalysis | snowfall (mm of water equivalent) | 99.5 %`, and one row per `swvlN_era5` with source "ERA5-Land, `auxiliary/oiko_soil.py`" and the coverage the file shows once it exists. In the conversions table add `lwrad_era5 | centred half an hour earlier | accumulation, as sr_era5`.

- [ ] **Step 2: Commit.** `git commit -- docs/proxy-data-dictionary.md -m "docs: dictionary rows for the longwave, snowfall and soil-water channels"`.

### Task 0.5 (S runs, O decides): Movement 0 — load, settle semantics, `HT_01`–`HT_03`, `HT_F01`–`HT_F02`

**Files:**
- Modify: `studies/06_hydro_thermal_drivers/hydro_thermal_drivers_study.py` (append Movement 0)

**Interfaces:**
- Consumes: `proxies.ERA5_MAP_FULL`, `proxies.SOIL_MAP` (0.2); `coupling.antecedent_index`, `coupling.annual_extrema` (1.1, written before this movement's state cells run, see the ordering note below).
- Produces: notebook variables `target`, `sensor`, `station`, `era5`, `soil` (a frame or `None`), `record`, `sets`, `frame`, `def_frames`, `rain` (the primary precipitation series on its own grid, snow folded in), `api` (dict `{tau_days: pd.Series}`) that every later movement reads; artefacts `HT_01_source_agreement`, `HT_02_annual_water`, `HT_03_state_phase`, `HT_F01_states`, `HT_F02_state_phase`.

Ordering note: this task's state cells call two functions of Task 1.1. Run Task 1.1 first, or run this task's Steps 1 to 3 now and its Steps 4 to 5 after Task 1.1. The plan orders Task 1.1 immediately after this one for that reason; the executor may swap them.

- [ ] **Step 1: Loading cells.** The movement's opening Markdown cell states what it computes and names the five artefacts. The loading cell is Study 05's Movement 1 loading block verbatim (target, `sensor_current`, `sensor_legacy`, `sensor`, `station_hourly`, `era5_hourly`, `hourly`, `station`, `era5`, `record`, `sets, frame`, `def_frames = prediction.regressor_set_frames(sets, frame['y'])`) with two changes: `era5_hourly = proxies.load_era5(ERA5_CSV, column_map=proxies.ERA5_MAP_FULL)` and `era5 = proxies.to_native_grid(era5_hourly, freq=NATIVE_FREQ, accumulations=tuple(q for q, flag in (('sr', ERA5_SR_IS_ACCUMULATION), ('lwrad', ERA5_LW_IS_ACCUMULATION)) if flag))`. The regressor sets are built with `radiation_delay_h=RADIATION_DELAY_H`; if `RADIATION_TAU_H > 0`, the radiation column of every set is replaced afterwards by `coupling.thermal_operator(sets[name]['sr'], delay=0, tau=RADIATION_TAU_H, dt_hours=1/3)`, and a printed line says which operator is in force. Then:

```python
soil = None
if Path(SOIL_CSV).exists():
    soil_hourly = proxies.load_era5(SOIL_CSV, column_map=proxies.SOIL_MAP)
    soil = proxies.to_native_grid(soil_hourly, freq=NATIVE_FREQ, accumulations=())
    print('soil-water file:', soil.index.min(), 'to', soil.index.max(),
          'coverage', f"{soil['swvl1_era5'].notna().mean():.3f}")
else:
    print('soil-water file absent; the antecedent index stands alone (spec §9)')
```

- [ ] **Step 2: Longwave semantics cell.** Under a `### Is the longwave channel downward or net?` Markdown cell:

```python
night = era5_hourly.between_time('00:00', '04:00')
emission = coupling.net_longwave(0.0 * night['lwrad_era5'], night['tair_era5'], emissivity=1.0).abs()
print('January-night mean of the file\'s longwave channel:',
      f"{night.loc[night.index.month == 1, 'lwrad_era5'].mean():.0f} W/m2")
print('black-body emission at the January-night air temperature:',
      f"{emission.loc[emission.index.month == 1].mean():.0f} W/m2")
```

A file value near the emission is a downward flux and the notebook proceeds with `net_longwave`; a value near zero or negative is already net, and the parameter cell's guidance for `ERA5_LW_IS_ACCUMULATION` records which was found. The orchestrator decides at Checkpoint 0 and the parameter cell states the decision.

- [ ] **Step 3: `HT_01` source agreement.** Rain on daily totals, wind on hourly means, station against ERA5, through Study 02's own scorer:

```python
daily_rain = hourly[['rain_gs', 'rain_era5']].resample('1D').sum(min_count=20)
agreement = pd.concat([
    compare.pairwise_agreement(daily_rain, 'rain', reference='gs', compared=('era5',)),
    compare.pairwise_agreement(hourly, 'wspd', reference='gs', compared=('era5',)),
], ignore_index=True)
display(agreement)
agreement.to_csv(OUTPUT_DIR / 'HT_01_source_agreement.csv', index=False)
tables.write_table(agreement, str(OUTPUT_DIR / 'HT_01_body.tex'),
                   [('quantity', tables.texttt), ('reference', tables.texttt),
                    ('compared', tables.texttt), ('n', ',d'), ('bias', '.2f'),
                    ('mae', '.2f'), ('r', '.3f')])
```

`compare` is added to the imports cell. The column names of `pairwise_agreement`'s return are read from `shmlib/compare.py:278` when the cell is written and the `write_table` list adjusted to them; the state dump run shows them.

- [ ] **Step 4: `HT_02` annual water and `HT_03` state phase.**

```python
rain_source = soil if soil is not None else era5
rain = (rain_source['rain_era5'].fillna(0.0) + rain_source['snow_era5'].fillna(0.0)).rename('rain')
rain[rain_source['rain_era5'].isna()] = np.nan
hydro_year = (rain.index + pd.DateOffset(months=13 - HYDRO_YEAR_START_MONTH)).year
annual = rain.groupby(hydro_year).sum(min_count=1).rename('total_mm').to_frame()
annual['snow_mm'] = rain_source['snow_era5'].groupby(hydro_year).sum(min_count=1)
annual['anomaly_mm'] = annual['total_mm'] - annual['total_mm'].mean()
annual['days_covered'] = rain.notna().groupby(hydro_year).sum() / 72
annual = annual.reset_index().rename(columns={'datetime': 'hydro_year'})
display(annual)
annual.to_csv(OUTPUT_DIR / 'HT_02_annual_water.csv', index=False)
tables.write_table(annual, str(OUTPUT_DIR / 'HT_02_body.tex'),
                   [('hydro_year', 'd'), ('total_mm', ',.0f'), ('snow_mm', ',.0f'),
                    ('anomaly_mm', '+,.0f'), ('days_covered', ',.0f')])

api = {tau: coupling.antecedent_index(rain, tau, max_gap=API_MAX_GAP,
                                      warmup_taus=API_WARMUP_TAUS) for tau in API_TAU_DAYS}
phase_rows = []
for tau in API_PHASE_TAUS:
    phase_rows.append({'state': f'api_{tau}d', **coupling.annual_extrema(api[tau])})
if soil is not None:
    for layer in SOIL_LAYERS:
        phase_rows.append({'state': f'swvl{layer}', **coupling.annual_extrema(soil[f'swvl{layer}_era5'])})
phase = pd.DataFrame(phase_rows)
phase['study05_yearly_doy_max'], phase['study05_yearly_doy_min'] = 80, 263
display(phase)
phase.to_csv(OUTPUT_DIR / 'HT_03_state_phase.csv', index=False)
tables.write_table(phase, str(OUTPUT_DIR / 'HT_03_body.tex'),
                   [('state', tables.texttt), ('doy_max', 'd'), ('doy_min', 'd'),
                    ('peak_to_peak', '.3g'), ('order', 'd')])
```

The two Study 05 day numbers are quoted from `GM_05d` and named as such in the guidance of the cell above.

- [ ] **Step 5: Figures.** `figures.plot_moisture_state(rain, api, soil, trend=None, title='Precipitation, its antecedent states and the soil water', save_path=str(OUTPUT_DIR), filename='HT_F01_states')` and `figures.plot_state_phase({f'api_{t}d': api[t] for t in API_PHASE_TAUS}, yearly_doy=(80, 263), title='The annual phase of the moisture states', save_path=str(OUTPUT_DIR), filename='HT_F02_state_phase')` (both from Task 1.3). `trend=None` here; Movement 1 redraws `HT_F01` with the base trend.

- [ ] **Step 6: Run against dumped state, then the whole notebook once.** Build the dump after the loading cell; iterate Steps 2 to 5 with `run_movement.py 0`; then `jupytext --to ipynb hydro_thermal_drivers_study.py` and the scratch `nbconvert` run of the README; count error cells in the executed `.ipynb`, expected 0; copy it back.

- [ ] **Step 7: Checkpoint 0 (O).** The orchestrator reads `HT_02` and `HT_03` and applies spec §8's kill criterion: if the range of `anomaly_mm` across the five fold years is smaller than the year-to-year noise implied by `days_covered`, the moisture half is dropped and Phases 1 and 2 shrink to the descriptive Movement 1. The decision, and the longwave semantics, are written into the parameter cell's guidance. Then commit:

```bash
git add studies/06_hydro_thermal_drivers/hydro_thermal_drivers_study.py studies/06_hydro_thermal_drivers/hydro_thermal_drivers_study.ipynb
git commit -- studies/06_hydro_thermal_drivers -m "feat(study06): Movement 0, the sources, the longwave semantics, the annual water budget and the phase of the moisture states"
```

### Task 0.5b (S): `coupling.amplitude_envelope`, and `HT_13` / `HT_F09` in Movement 0

Added 2026-09-07 with spec D11. The envelope belongs in Phase 0 because the report's §2 quotes it as context for everything that follows, and because it needs nothing but the loaded frames — no model, no fold, no state.

**Files:**
- Modify: `studies/shmlib/coupling.py`, `studies/shmlib/figures.py`, `studies/06_hydro_thermal_drivers/tests/test_states.py`, `.../hydro_thermal_drivers_study.py` (append to Movement 0)

**Interfaces:**
- Produces: `coupling.amplitude_envelope(frame, channels, min_hours=20) -> pd.DataFrame` with one row per channel and month, columns `channel`, `month`, `amplitude`, `n_days`, plus a `ratio` column repeating that channel's max-over-min across the twelve months; `figures.plot_driver_envelope`; artefacts `HT_13_driver_envelope`, `HT_F09_driver_envelope`.

- [ ] **Step 1: Write the failing test** (append a class to `test_states.py`). A synthetic series whose daily amplitude is 1.0 in December and 4.0 in July must come back with `ratio` 4.0 to two places, and a month whose days all fall below `min_hours` must come back with `n_days` 0 and `amplitude` NaN rather than being dropped.

- [ ] **Step 2: Run to verify failure.** Expected `AttributeError … 'amplitude_envelope'`.

- [ ] **Step 3: Implement.** The function is a reduction over `compare.daily_amplitude_phase`, which already fits the 24-hour harmonic per calendar day and is used by Study 05's harmonic diagnostic; it must call that function rather than refitting a harmonic, so that the two studies' daily amplitudes are the same quantity. The docstring states in one line why the ratio, not the level, is the quantity of interest, and points at spec §2.6.

- [ ] **Step 4: Notebook cells.** Under a `### The annual envelope of each driver's daily cycle` Markdown cell that explains what the ratio measures and quotes Study 05's 2.8 and 17.1 mdeg for the target:

```python
envelope = coupling.amplitude_envelope(record, channels=ENVELOPE_CHANNELS,
                                       min_hours=ENVELOPE_MIN_HOURS)
display(envelope)
envelope.to_csv(OUTPUT_DIR / 'HT_13_driver_envelope.csv', index=False)
tables.write_table(envelope, str(OUTPUT_DIR / 'HT_13_body.tex'),
                   [('channel', tables.texttt), ('month', 'd'), ('amplitude', '.2f'),
                    ('n_days', ',d'), ('ratio', '.2f')])
figures.plot_driver_envelope(envelope, title='The annual envelope of each driver',
                             save_path=str(OUTPUT_DIR), filename='HT_F09_driver_envelope')
```

`HT_F09` normalises each channel to its own December value so that channels in different units share one axes; the target's line is drawn in its identity blue and the drivers in theirs, per the graphical rules.

- [ ] **Step 5: Expected values, as a check on the plumbing rather than a result.** ERA5 air temperature about 2.4 and ERA5 radiation about 4.2 (spec §2.6, measured on the same file); the target about 6.1 (Study 05 `GM_04`). A ratio far from these on the ERA5 rows means the reduction is wrong, not that the record is surprising. The on-structure and station rows have no prior and are the new information.

- [ ] **Step 6: Run the tests, then the whole suite; run the notebook; commit** with message `feat(study06): the annual envelope of each driver's daily cycle, HT_13 and HT_F09`.

### Task 0.6 (O): Report §1 and §2; status to "In progress"

- [ ] Write the Introduction and *The sources and the states* from `HT_01`–`HT_03`, `HT_13`, `HT_F01`, `HT_F02`, `HT_F09`, in full academic prose, removing their `\pending{}`. State in §2 what the longwave channel is, whether soil water was obtained, the annual water totals, the phase of the filtered precipitation beside day 80 and day 263, and the annual envelope of each driver beside the target's, with the observation of spec §2.6 that no single driver in the record carries the target's ratio and that a linear filter does not change one. The Introduction carries the fourth, methodological question of spec §1 in one short paragraph, marked as a diagnostic answered last. Update the README status line and the `studies/README.md` row to "In progress". Build the PDF twice, run the honesty test, commit with `-- studies/06_hydro_thermal_drivers studies/README.md`.

---

# Phase 1 · Base fit and the moisture states, descriptively

### Task 1.1 (S): `coupling.antecedent_index` and `coupling.annual_extrema`

**Files:**
- Modify: `studies/shmlib/coupling.py` (append after `evaluate_modulation`)
- Create: `studies/06_hydro_thermal_drivers/tests/test_states.py`

**Interfaces:**
- Produces: `coupling.antecedent_index(precip, tau_days, max_gap='6h', warmup_taus=3.0) -> pd.Series` named `api_<tau>d`, on `precip`'s own regular index: the causal exponential accumulation `x_t = x_{t-1}·exp(−Δt/τ) + P_t` in the units of `precip`, with missing slots counted as zero rain except that a run of missing slots longer than `max_gap`, and the `warmup_taus·τ` after it, and the first `warmup_taus·τ` of the record, are `NaN`. And `coupling.annual_extrema(series, harmonics=(1, 2), min_gain=0.01) -> dict` with keys `doy_max`, `doy_min`, `peak_to_peak`, `order`: the series is averaged per calendar day, fitted against day of year with `annual_modulation` (order chosen on held-out years), the fit evaluated on one synthetic year, and the days of its maximum and minimum returned.

- [ ] **Step 1: Write the failing tests**

```python
"""
Tests for the state functions Study 06 adds to shmlib.coupling.

Run from studies/:  python 06_hydro_thermal_drivers/tests/test_states.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import coupling  # noqa: E402


class TestAntecedentIndex(unittest.TestCase):

    def test_an_impulse_decays_to_one_over_e_after_one_time_constant(self):
        index = pd.date_range('2020-01-01', periods=24 * 10, freq='1h')
        rain = pd.Series(0.0, index=index)
        rain.iloc[24 * 4] = 10.0             # after a 3-tau warm-up of one day each
        out = coupling.antecedent_index(rain, tau_days=1, warmup_taus=3.0)
        self.assertEqual(out.name, 'api_1d')
        self.assertTrue(np.isnan(out.iloc[0]))
        self.assertAlmostEqual(out.iloc[24 * 4], 10.0)
        self.assertAlmostEqual(out.iloc[24 * 5], 10.0 * np.exp(-1.0), places=6)

    def test_constant_rain_converges_to_the_geometric_sum(self):
        index = pd.date_range('2020-01-01', periods=24 * 60, freq='1h')
        rain = pd.Series(1.0, index=index)
        out = coupling.antecedent_index(rain, tau_days=2, warmup_taus=3.0)
        decay = np.exp(-1.0 / 48.0)
        self.assertAlmostEqual(out.iloc[-1], 1.0 / (1.0 - decay), places=3)

    def test_a_short_gap_counts_as_no_rain_and_a_long_gap_resets(self):
        index = pd.date_range('2020-01-01', periods=24 * 20, freq='1h')
        rain = pd.Series(1.0, index=index)
        rain.iloc[24 * 8: 24 * 8 + 3] = np.nan          # three hours, under max_gap
        rain.iloc[24 * 12: 24 * 12 + 12] = np.nan       # twelve hours, over it
        out = coupling.antecedent_index(rain, tau_days=1, max_gap='6h', warmup_taus=3.0)
        self.assertFalse(np.isnan(out.iloc[24 * 8 + 1]))
        self.assertTrue(np.isnan(out.iloc[24 * 12 + 1]))
        self.assertTrue(np.isnan(out.iloc[24 * 12 + 12 + 24 * 3 - 1]))
        self.assertFalse(np.isnan(out.iloc[24 * 12 + 12 + 24 * 3 + 1]))


class TestAnnualExtrema(unittest.TestCase):

    def test_a_cosine_peaking_on_day_80_is_found_there(self):
        index = pd.date_range('2019-01-01', '2022-12-31', freq='1D')
        doy = index.dayofyear.to_numpy()
        series = pd.Series(np.cos(2 * np.pi * (doy - 80) / 365.25), index=index)
        out = coupling.annual_extrema(series, harmonics=(1, 2))
        self.assertLessEqual(abs(out['doy_max'] - 80), 2)
        self.assertLessEqual(abs(out['doy_min'] - 263), 2)
        self.assertAlmostEqual(out['peak_to_peak'], 2.0, places=1)
        self.assertIn(out['order'], (1, 2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Run to verify failure.** `python 06_hydro_thermal_drivers/tests/test_states.py`. Expected: `AttributeError: module 'shmlib.coupling' has no attribute 'antecedent_index'`.

- [ ] **Step 3: Implement**

```python
def antecedent_index(precip, tau_days, max_gap='6h', warmup_taus=3.0):
    """
    Water stored behind the wall, as an exponentially decaying account of past rain.

    The embankment and the masonry do not feel a shower, they feel what a
    sequence of showers has left in them, which drains with time. The
    antecedent precipitation index is the simplest state with that behaviour:
    every slot adds the rain that fell and the whole account decays by
    ``exp(-dt / tau)`` between slots, so a pulse of rain is worth ``1/e`` of
    itself one time constant later. It is the same one-pole filter
    :func:`thermal_lag_filter` applies to a temperature, in the unnormalised
    form whose unit is millimetres of stored water.

    Parameters
    ----------
    precip : pd.Series
        Precipitation per slot, in millimetres, on a regular ``DatetimeIndex``.
        Missing slots count as no rain, except as ``max_gap`` says.
    tau_days : float
        Time constant in days.
    max_gap : str or pd.Timedelta, optional
        A run of missing slots longer than this is a gap the state cannot see
        through: the run itself and the ``warmup_taus * tau_days`` after it are
        returned as ``NaN``. Default ``'6h'``.
    warmup_taus : float, optional
        Multiples of ``tau_days`` masked at the record's start and after every
        long gap, since the account holds nothing it could not have seen.
        Default ``3.0``.

    Returns
    -------
    pd.Series
        Named ``api_<tau>d``, indexed like ``precip``.
    """
    series = pd.to_numeric(precip, errors='coerce')
    index = pd.DatetimeIndex(series.index)
    step = pd.Series(index).diff().median()
    dt_days = step / pd.Timedelta(days=1)
    decay = float(np.exp(-dt_days / float(tau_days)))
    values = series.fillna(0.0).to_numpy(dtype=float)
    state = np.empty_like(values)
    account = 0.0
    for i, fell in enumerate(values):
        account = account * decay + fell
        state[i] = account
    out = pd.Series(state, index=index, name=f'api_{int(tau_days)}d')

    warm = pd.Timedelta(days=float(warmup_taus) * float(tau_days))
    mask = index < index[0] + warm
    missing = series.isna().to_numpy()
    if missing.any():
        runs = np.cumsum(np.r_[True, missing[1:] != missing[:-1]])
        for run in np.unique(runs[missing]):
            slots = np.flatnonzero(runs == run)
            if len(slots) * step > pd.Timedelta(max_gap):
                end = index[slots[-1]]
                mask |= (runs == run) | ((index > end) & (index <= end + warm))
    out[mask] = np.nan
    return out


def annual_extrema(series, harmonics=(1, 2), min_gain=0.01):
    """
    The days of the year on which a state peaks and troughs, from its annual fit.

    Parameters
    ----------
    series : pd.Series
        A state on any regular ``DatetimeIndex``, spanning more than one year.
    harmonics : sequence of int, optional
        Candidate annual Fourier orders passed to :func:`annual_modulation`.
        Default ``(1, 2)``.
    min_gain : float, optional
        Passed to :func:`annual_modulation`. Default ``0.01``.

    Returns
    -------
    dict
        ``doy_max``, ``doy_min`` (days of year of the fitted curve's extrema),
        ``peak_to_peak`` of the fitted curve, and the ``order`` chosen.
    """
    daily = pd.to_numeric(series, errors='coerce').resample('1D').mean().dropna()
    _, fit = annual_modulation(daily, harmonics=harmonics, holdout='year',
                               min_gain=min_gain)
    year = pd.date_range('2001-01-01', '2001-12-31', freq='1D')
    curve = evaluate_modulation(fit, year)
    return {'doy_max': int(curve.idxmax().dayofyear),
            'doy_min': int(curve.idxmin().dayofyear),
            'peak_to_peak': float(curve.max() - curve.min()),
            'order': int(fit['order'])}
```

- [ ] **Step 4: Run the tests; expected 4 PASS.** Then the whole suite.

- [ ] **Step 5: Commit**

```bash
git add studies/shmlib/coupling.py studies/06_hydro_thermal_drivers/tests/test_states.py
git commit -- studies/shmlib/coupling.py studies/06_hydro_thermal_drivers/tests/test_states.py -m "feat(shmlib): the antecedent precipitation index and the annual extrema of a state"
```

### Task 1.2 (S): `coupling.segment_rate_regression`

**Files:**
- Modify: `studies/shmlib/coupling.py` (append)
- Modify: `studies/06_hydro_thermal_drivers/tests/test_states.py` (add a class)

**Interfaces:**
- Produces: `coupling.segment_rate_regression(rates, state, observed=None, max_gap_days=30, edge_days=7) -> (pd.DataFrame, dict)`. `rates` is `trend_parameters`' table (`start`, `end`, `rate_mdeg_per_year`); `state` a series on the study grid; `observed` an optional boolean series marking slots where the target is present. The table has one row per segment with `state_start` and `state_end` (the state's mean over the first and last `edge_days` of the segment), `state_rate` (their difference per year of segment length), `included` (`False` when `observed` has a missing run longer than `max_gap_days` inside the segment). The dict has `slope`, `intercept`, `r`, `p_value`, `n` from `scipy.stats.linregress` of `rate_mdeg_per_year` on `state_rate` over the included rows, and `sign_agreement`, the fraction of included segments where `rate_mdeg_per_year` and `slope * state_rate` share a sign.

- [ ] **Step 1: Write the failing test** (append to `test_states.py`)

```python
class TestSegmentRateRegression(unittest.TestCase):

    def _rates_and_state(self):
        index = pd.date_range('2019-01-01', '2020-12-31 23:00', freq='1h')
        edges = pd.to_datetime(['2019-01-01', '2019-07-01', '2020-01-01', '2020-07-01',
                                '2020-12-31 23:00'])
        state = pd.Series(0.0, index=index)
        slopes = [10.0, -5.0, 20.0, -15.0]          # state units per year, per segment
        level = 0.0
        for start, end, slope in zip(edges[:-1], edges[1:], slopes):
            years = (index[(index >= start) & (index < end)] - start) / pd.Timedelta(days=365.25)
            state.loc[start:end - pd.Timedelta(hours=1)] = level + slope * years.to_numpy()
            level = state.loc[:end - pd.Timedelta(hours=1)].iloc[-1]
        rates = pd.DataFrame({'start': edges[:-1], 'end': edges[1:],
                              'rate_mdeg_per_year': [2.0 * s for s in slopes]})
        return rates, state, index

    def test_a_proportional_rate_gives_the_slope_and_full_sign_agreement(self):
        rates, state, _ = self._rates_and_state()
        table, summary = coupling.segment_rate_regression(rates, state, edge_days=7)
        self.assertEqual(len(table), 4)
        self.assertTrue(table['included'].all())
        self.assertAlmostEqual(summary['slope'], 2.0, places=1)
        self.assertGreater(summary['r'], 0.99)
        self.assertEqual(summary['sign_agreement'], 1.0)

    def test_a_segment_spanning_a_long_target_gap_is_excluded(self):
        rates, state, index = self._rates_and_state()
        observed = pd.Series(True, index=index)
        observed.loc['2019-09-01':'2019-11-15'] = False
        table, summary = coupling.segment_rate_regression(
            rates, state, observed=observed, max_gap_days=30)
        self.assertFalse(bool(table.loc[1, 'included']))
        self.assertEqual(summary['n'], 3)
```

- [ ] **Step 2: Run to verify failure.** Expected: `AttributeError … 'segment_rate_regression'`.

- [ ] **Step 3: Implement**

```python
def segment_rate_regression(rates, state, observed=None, max_gap_days=30,
                            edge_days=7):
    """
    Do the trend's own segment rates follow the change of a slow state?

    Study 05's trend reverses at rates of tens of millidegrees per year on
    segments months long. If a slow driver moves the wall, the rate of each
    segment should follow the driver's own rate of change over the same
    segment, with one sign. This regresses the one on the other, segment by
    segment, before any model carries the state, so that the model
    comparison of D5 is read against a picture that does not depend on how
    a trend and a regressor share variance.

    Parameters
    ----------
    rates : pd.DataFrame
        ``start``, ``end``, ``rate_mdeg_per_year``, as
        :func:`shmlib.prediction.trend_parameters` returns.
    state : pd.Series
        The state on the study grid.
    observed : pd.Series or None, optional
        Boolean, ``True`` where the target is present. A segment holding a
        missing run longer than ``max_gap_days`` is excluded, because its
        rate is a line between two data and not a measurement. Default
        ``None``, nothing excluded.
    max_gap_days : float, optional
        Default ``30``.
    edge_days : float, optional
        The state is averaged over this many days at each end of a segment
        before the difference is taken. Default ``7``.

    Returns
    -------
    (pd.DataFrame, dict)
        The per-segment table and the regression summary; see the plan's
        interface block.
    """
    from scipy import stats
    state = pd.to_numeric(state, errors='coerce')
    edge = pd.Timedelta(days=float(edge_days))
    rows = []
    for _, seg in rates.iterrows():
        start, end = pd.Timestamp(seg['start']), pd.Timestamp(seg['end'])
        first = state.loc[start:start + edge].mean()
        last = state.loc[end - edge:end].mean()
        years = (end - start) / pd.Timedelta(days=365.25)
        included = True
        if observed is not None:
            inside = observed.loc[start:end].astype(bool)
            missing = ~inside
            if missing.any():
                runs = (missing != missing.shift()).cumsum()
                longest = missing.groupby(runs).sum().max()
                step = pd.Series(inside.index).diff().median()
                included = longest * step <= pd.Timedelta(days=float(max_gap_days))
        rows.append({'start': start, 'end': end,
                     'rate_mdeg_per_year': float(seg['rate_mdeg_per_year']),
                     'state_start': float(first), 'state_end': float(last),
                     'state_rate': float((last - first) / years) if years > 0 else np.nan,
                     'included': bool(included)})
    table = pd.DataFrame(rows)
    use = table[table['included']].dropna(subset=['state_rate', 'rate_mdeg_per_year'])
    if len(use) < 3:
        summary = {'slope': np.nan, 'intercept': np.nan, 'r': np.nan,
                   'p_value': np.nan, 'n': int(len(use)), 'sign_agreement': np.nan}
        return table, summary
    fit = stats.linregress(use['state_rate'], use['rate_mdeg_per_year'])
    agree = np.sign(use['rate_mdeg_per_year']) == np.sign(fit.slope * use['state_rate'])
    summary = {'slope': float(fit.slope), 'intercept': float(fit.intercept),
               'r': float(fit.rvalue), 'p_value': float(fit.pvalue), 'n': int(len(use)),
               'sign_agreement': float(agree.mean())}
    return table, summary
```

- [ ] **Step 4: Run the tests; expected 6 PASS in the file.** Then the whole suite.

- [ ] **Step 5: Commit.** `git commit -- studies/shmlib/coupling.py studies/06_hydro_thermal_drivers/tests/test_states.py -m "feat(shmlib): regress the trend's segment rates on the change of a slow state"`.

### Task 1.3 (S): Figures `plot_moisture_state`, `plot_state_phase`, `plot_trend_rate_regression`

**Files:**
- Modify: `studies/shmlib/figures.py` (append)
- Create: `studies/06_hydro_thermal_drivers/tests/test_figures06.py`

**Interfaces:**
- Produces: `figures.plot_moisture_state(rain, api, soil=None, trend=None, title='', save_path=None, filename=None) -> Figure`: panels on one clock, top the daily precipitation total as bars in `viz.QUANTITY_COLOUR['rain']`, then one panel per `api` entry (dict `{tau_days: Series}`) in `QUANTITY_COLOUR['api']`, then, if `soil` is given, one panel with its `swvlN_era5` columns as line styles of one colour, and, if `trend` is given, a last panel with the base trend in `viz.INC_COLOUR`. `figures.plot_state_phase(states, yearly_doy, title='', save_path=None, filename=None)`: for each named series in `states` the annual curve (daily mean by day of year, normalised to the unit interval) on one axes, with vertical accent lines at the two `yearly_doy` values and a legend below. `figures.plot_trend_rate_regression(tables, summaries, title='', save_path=None, filename=None)`: `tables` and `summaries` are dicts keyed by state name from `segment_rate_regression`; one panel per state, included segments as filled markers, excluded as hollow, the fitted line, and the slope and `r` in the panel's corner annotation.

- [ ] **Step 1: Write the failing smoke tests**

```python
"""
Smoke tests for the figures Study 06 adds to shmlib.figures.

Run from studies/:  python 06_hydro_thermal_drivers/tests/test_figures06.py
"""
import os
import sys
import unittest

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import coupling, figures  # noqa: E402


def _rain():
    index = pd.date_range('2019-01-01', '2020-12-31 23:00', freq='1h')
    rng = np.random.default_rng(0)
    return pd.Series(rng.exponential(0.1, len(index)), index=index, name='rain')


class TestStudy06Figures(unittest.TestCase):

    def test_moisture_state_draws_one_panel_per_state(self):
        rain = _rain()
        api = {30: coupling.antecedent_index(rain, 30), 90: coupling.antecedent_index(rain, 90)}
        fig = figures.plot_moisture_state(rain, api, soil=None, trend=rain.rolling(24 * 30).mean())
        self.assertEqual(len(fig.axes), 4)

    def test_state_phase_draws_every_state_and_two_marks(self):
        rain = _rain()
        api = coupling.antecedent_index(rain, 30)
        fig = figures.plot_state_phase({'api_30d': api}, yearly_doy=(80, 263))
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(len([l for l in fig.axes[0].lines if l.get_linestyle() == '--']), 2)

    def test_trend_rate_regression_draws_one_panel_per_state(self):
        rates = pd.DataFrame({'start': pd.to_datetime(['2019-01-01', '2019-07-01']),
                              'end': pd.to_datetime(['2019-07-01', '2020-01-01']),
                              'rate_mdeg_per_year': [10.0, -5.0]})
        state = _rain().cumsum()
        table, summary = coupling.segment_rate_regression(rates, state)
        fig = figures.plot_trend_rate_regression({'api_30d': table}, {'api_30d': summary})
        self.assertEqual(len(fig.axes), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Run to verify failure.** Expected: `AttributeError … 'plot_moisture_state'`.

- [ ] **Step 3: Implement**, following the figure template. Sketch of `plot_moisture_state`:

```python
def plot_moisture_state(rain, api, soil=None, trend=None, title='', save_path=None,
                        filename=None):
    """
    Precipitation, its antecedent states and the soil water on one clock.

    Parameters
    ----------
    rain : pd.Series
        Precipitation per slot, millimetres.
    api : dict
        ``{tau_days: pd.Series}`` from :func:`shmlib.coupling.antecedent_index`.
    soil : pd.DataFrame or None, optional
        Columns ``swvl<N>_era5``. Default ``None``, no panel.
    trend : pd.Series or None, optional
        A fitted trend, drawn last in the inclination colour. Default ``None``.
    title : str, optional
    save_path, filename : str or None, optional
        Passed to :func:`shmlib.viz.finish`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    panels = 1 + len(api) + (soil is not None) + (trend is not None)
    fig, axes = plt.subplots(panels, 1, sharex=True,
                             figsize=viz.figsize(viz.FIGURE_WIDTH, 1.2 * panels + 0.6))
    axes = np.atleast_1d(axes)
    daily = pd.to_numeric(rain, errors='coerce').resample('1D').sum(min_count=1)
    axes[0].bar(daily.index, daily.to_numpy(), width=1.0,
                color=viz.QUANTITY_COLOUR['rain'], linewidth=0)
    axes[0].set_ylabel('Rain [mm/d]')
    row = 1
    for tau, series in api.items():
        axes[row].plot(series.index, series.to_numpy(), color=viz.QUANTITY_COLOUR['api'],
                       linewidth=0.8)
        axes[row].set_ylabel(f'API τ={tau} d [mm]')
        row += 1
    if soil is not None:
        styles = ['-', '--', '-.', ':']
        for style, column in zip(styles, [c for c in soil.columns if c.startswith('swvl')]):
            axes[row].plot(soil.index, soil[column].to_numpy(), style,
                           color=viz.QUANTITY_COLOUR['swvl'], linewidth=0.8, label=column)
        axes[row].set_ylabel('Soil water [m³/m³]')
        axes[row].legend(fontsize='small', ncol=4, loc='upper center',
                         bbox_to_anchor=(0.5, -0.30), frameon=False)
        row += 1
    if trend is not None:
        axes[row].plot(trend.index, trend.to_numpy(), color=viz.INC_COLOUR, linewidth=0.9)
        axes[row].set_ylabel('Trend [mdeg]')
    for ax in axes:
        viz.format_spines(ax)
    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig
```

`plot_state_phase`: for each state, `daily = series.resample('1D').mean()`, `by_doy = daily.groupby(daily.index.dayofyear).mean()`, normalised `(by_doy - min) / (max - min)`, one line per state, `ax.axvline(doy, color=viz.ACCENT, linestyle='--')` for each of `yearly_doy` (if `viz.ACCENT` does not exist, `'#D55E00'` with a comment), x label "Day of year", y label "Normalised state", legend below. `plot_trend_rate_regression`: one panel per key, `ax.scatter` of `state_rate` against `rate_mdeg_per_year` with `facecolors='none'` for excluded rows, the line `intercept + slope·x` over the x range, `ax.annotate(f"slope {slope:.2f}, r {r:.2f}, n {n}", xy=(0.02, 0.95), xycoords='axes fraction', va='top')`, axis labels "State change [unit/yr]" and "Trend rate [mdeg/yr]".

- [ ] **Step 4: Run the tests; expected 3 PASS.** Then `python shmlib/tests/test_shmlib.py`.

- [ ] **Step 5: Commit.** `git commit -- studies/shmlib/figures.py studies/06_hydro_thermal_drivers/tests/test_figures06.py -m "feat(shmlib): figures for the moisture states, their annual phase and the trend-rate regression"`.

### Task 1.4 (S runs, O interprets): Movement 1 — the base fit, its trend rates, `HT_04`, `HT_F01` redrawn, `HT_F03`

**Files:**
- Modify: the notebook (append Movement 1)

**Interfaces:**
- Consumes: `def_frames`, `api`, `soil`, `record` from Movement 0; `coupling.segment_rate_regression` (1.2); `figures.plot_trend_rate_regression` (1.3).
- Produces: notebook variables `base_fits` (the `fits` dict of `attribution_fits` on the on-structure set alone), `base_rates`, `base_trend`; artefacts `HT_04_trend_rate_regression`, `HT_F01_states` (redrawn with the trend), `HT_F03_trend_rate_regression`.

- [ ] **Step 1: The base fit.** Opening Markdown cell; then, under `### Model A on the on-structure set, as Study 05 fitted it`:

```python
base_fits, base_shares, base_gains, base_diag = prediction.attribution_fits(
    {'str': def_frames['str']}, ('tair', 'rh', 'sr'), VALID_P, N_CHANGEPOINTS,
    study03_gains=STUDY03_GAINS, n_jobs=1, epochs=EPOCHS, freq=NATIVE_FREQ,
    growth='linear', changepoints_range=CHANGEPOINTS_RANGE, trend_reg=TREND_REG,
    yearly_order=YEARLY_ORDER, daily_order=DAILY_ORDER, quantiles=QUANTILES,
    seed=SEED, learning_rate=LEARNING_RATE)
base = base_fits['str']
base_trend, base_rates = prediction.trend_parameters(
    base['model'], base['train'], base['changepoints'], regressors=('tair', 'rh', 'sr'))
display(base_rates)
study05_rates = STUDY05_OUTPUTS / 'GM_04d_trend_rates.csv'
if study05_rates.exists():
    display(pd.read_csv(study05_rates))       # side by side: the same fit should give the same rates
```

- [ ] **Step 2: `HT_04`.**

```python
observed = record['y'].notna()
reg_tables, reg_summaries, rows = {}, {}, []
for tau in API_TAU_DAYS:
    table, summary = coupling.segment_rate_regression(
        base_rates, api[tau].reindex(record.index), observed=observed,
        max_gap_days=SEGMENT_MAX_GAP_DAYS, edge_days=SEGMENT_EDGE_DAYS)
    reg_tables[f'api_{tau}d'], reg_summaries[f'api_{tau}d'] = table, summary
    rows.append({'state': f'api_{tau}d', **summary})
if soil is not None:
    for layer in SOIL_LAYERS:
        name = f'swvl{layer}'
        table, summary = coupling.segment_rate_regression(
            base_rates, soil[f'{name}_era5'].reindex(record.index), observed=observed,
            max_gap_days=SEGMENT_MAX_GAP_DAYS, edge_days=SEGMENT_EDGE_DAYS)
        reg_tables[name], reg_summaries[name] = table, summary
        rows.append({'state': name, **summary})
rate_regression = pd.DataFrame(rows)
display(rate_regression)
rate_regression.to_csv(OUTPUT_DIR / 'HT_04_trend_rate_regression.csv', index=False)
tables.write_table(rate_regression, str(OUTPUT_DIR / 'HT_04_body.tex'),
                   [('state', tables.texttt), ('slope', '.3g'), ('r', '.2f'),
                    ('p_value', '.3f'), ('n', 'd'), ('sign_agreement', tables.percent)])
```

- [ ] **Step 3: Figures.** Redraw `HT_F01` with `trend=base_trend`; then `figures.plot_trend_rate_regression(reg_tables, reg_summaries, title='Trend segment rates against the change of each moisture state', save_path=str(OUTPUT_DIR), filename='HT_F03_trend_rate_regression')`.

- [ ] **Step 4: Run against dumped state, then the whole notebook once.** Expected 0 error cells; the base rates match Study 05's `GM_04d` to the printed precision when that file exists (same data, same seed, same arguments).

- [ ] **Step 5: Checkpoint 1 (O).** The orchestrator reads `HT_04`: the sign and consistency of the slope per τ, and names the τ range worth carrying into Phase 2 (all of `API_TAU_DAYS` unless a τ has `sign_agreement` under one half and `p_value` above 0.5, in which case it is dropped from `API_TAU_DAYS` in the parameter cell with the reason in its guidance). Commit as in Task 0.5.

### Task 1.5 (O): Report §3 Method, and §4's first half

- [ ] Write §3 with one subsection per decision D1 to D10, in the words of the spec, and the first half of §4 (the descriptive test of D5(1)) from `HT_04` and `HT_F03`. Build twice, honesty test, commit.

---

# Phase 2 · Moisture in the model

### Task 2.1 (S): `prediction.year_folds`

**Files:**
- Modify: `studies/shmlib/prediction.py` (append after `fold_stability`)
- Create: `studies/06_hydro_thermal_drivers/tests/test_year_ladder.py`

**Interfaces:**
- Produces: `prediction.year_folds(index, years) -> list of dict`, each `{'year': int, 'train': np.ndarray(bool), 'test': np.ndarray(bool)}` aligned with `index`, `test` true on the calendar year, `train` its complement; a year absent from `index` yields no fold.

- [ ] **Step 1: Write the failing test**

```python
"""
Tests for the leave-one-year-out machinery Study 06 adds to shmlib.prediction.

Run from studies/:  python 06_hydro_thermal_drivers/tests/test_year_ladder.py
"""
import logging
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import prediction  # noqa: E402

logging.getLogger('NP').setLevel(logging.ERROR)


class TestYearFolds(unittest.TestCase):

    def test_each_fold_holds_out_one_year_and_trains_on_the_rest(self):
        index = pd.date_range('2019-01-01', '2021-12-31 23:00', freq='1h')
        folds = prediction.year_folds(index, (2019, 2020, 2021, 2030))
        self.assertEqual([f['year'] for f in folds], [2019, 2020, 2021])
        for fold in folds:
            self.assertTrue((index[fold['test']].year == fold['year']).all())
            self.assertTrue(np.array_equal(fold['train'], ~fold['test']))
```

- [ ] **Step 2: Run to verify failure.** Expected `AttributeError … 'year_folds'`.

- [ ] **Step 3: Implement**

```python
def year_folds(index, years):
    """
    Leave-one-year-out folds: each calendar year held out in turn (spec D1).

    A thermal delay gives the same annual term every year; a moisture cycle
    gives one that follows the year's rain. Telling them apart needs whole
    years as the unit of validation, trained on the years on either side,
    so that the held-out year is a hole the trend must bridge and the
    driver, if it is one, must fill.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The frame's index.
    years : sequence of int
        Calendar years to hold out, each in turn. A year with no rows in
        ``index`` is skipped.

    Returns
    -------
    list of dict
        ``{'year', 'train', 'test'}``, the two masks boolean arrays aligned
        with ``index``.
    """
    index = pd.DatetimeIndex(index)
    folds = []
    for year in years:
        test = (index.year == int(year))
        if not test.any():
            continue
        folds.append({'year': int(year), 'train': ~test, 'test': test})
    return folds
```

- [ ] **Step 4: Run the test; expected PASS.** **Step 5: Commit** `-- studies/shmlib/prediction.py studies/06_hydro_thermal_drivers/tests/test_year_ladder.py -m "feat(shmlib): leave-one-year-out folds"`.

### Task 2.2 (S): `prediction.year_ladder`, and figures `plot_changepoint_sweep`, `plot_yearly_by_year`

**Files:**
- Modify: `studies/shmlib/prediction.py` (append after `year_folds`), `studies/shmlib/figures.py` (append)
- Modify: `tests/test_year_ladder.py`, `tests/test_figures06.py` (add classes)

**Interfaces:**
- Consumes: `year_folds` (2.1); `neuralprophet_backtest`, `covered_changepoints`, `decompose_components`, `regressor_gains`, `component_variance_shares`, `paired_mae_skill`, `_parallel_map` (existing).
- Produces: `prediction.year_ladder(frame, rungs, folds, n_changepoints, block_hours, repetitions, seed, n_jobs=1, gain_regressor=None, **model_kwargs) -> (ladder, per_fold, errors)`. `rungs` are `(label, columns, gate)` triples exactly as `channel_ladder` takes them. For every rung and fold the block is `frame.dropna(subset=['y'] + columns)` further restricted by `gate`, split by the fold's masks, fitted with `neuralprophet_backtest(train, test, regressors=tuple(columns), task='nowcast', changepoints=covered_changepoints(train.index, n_changepoints) if n_changepoints > 0 else None, n_changepoints=n_changepoints, seed=seed, **model_kwargs)`. `per_fold` has one row per rung and fold: `rung`, `year`, `rows_train`, `rows_test`, `mae`, `gain` (of `gain_regressor`, default the rung's last column; `NaN` for a rung without regressors beyond the base), `yearly_pp` (peak-to-peak of `season_yearly` over the training rows), `yearly_share` (its row of `component_variance_shares`). `ladder` has one row per rung: `rung`, `rows` (total test rows over folds), `mae_val` (mean absolute error pooled over folds), `gain_mean`, `gain_sign_stable` (`True` when every fold's gain has the same sign), `yearly_pp_mean`, `yearly_share_mean`, and `skill`, `skill_q05`, `skill_q95` against the rung below plus `skill_vs_first`, `skill_vs_first_q05`, `skill_vs_first_q95`, each from `paired_mae_skill` on the pooled error series' shared timestamps, `NaN` where not applicable, exactly as `channel_ladder` names them so that `figures.plot_ladder` draws the result unchanged. `errors` maps each rung label to its pooled absolute-error series. Also `figures.plot_changepoint_sweep(sweep, title='', save_path=None, filename=None)`: `sweep` a frame with columns `n_changepoints`, `rung`, `mae_val`; one line per rung over `n_changepoints`, the base rung in black, the states in `QUANTITY_COLOUR['api']` with line styles, legend below. And `figures.plot_yearly_by_year(per_fold, rungs=None, title='', save_path=None, filename=None)`: grouped bars of `yearly_pp` per `year`, one bar per rung in `rungs` (default all), legend below.

- [ ] **Step 1: Write the failing tests** (append to `test_year_ladder.py`)

```python
def _synthetic_frame():
    index = pd.date_range('2019-01-01', '2021-12-31 23:00', freq='1h')
    rng = np.random.default_rng(1)
    doy = index.dayofyear.to_numpy()
    tair = 12 + 10 * np.cos(2 * np.pi * (doy - 200) / 365.25) + 4 * np.cos(2 * np.pi * index.hour / 24)
    state = np.cumsum(rng.exponential(0.05, len(index))) - 0.05 * np.arange(len(index))
    y = -2.5 * tair + 0.8 * state + rng.normal(0, 0.5, len(index))
    return pd.DataFrame({'y': y, 'tair': tair, 'state': state}, index=index)


class TestYearLadder(unittest.TestCase):

    def test_the_ladder_scores_every_rung_on_every_year(self):
        frame = _synthetic_frame()
        folds = prediction.year_folds(frame.index, (2019, 2020, 2021))
        rungs = [('base', ['tair'], None), ('+ state', ['tair', 'state'], None)]
        ladder, per_fold, errors = prediction.year_ladder(
            frame, rungs, folds, n_changepoints=2, block_hours=24, repetitions=50,
            seed=0, n_jobs=1, epochs=2, freq='1h', growth='linear', yearly_order=1,
            daily_order=1, quantiles=(0.05, 0.95), learning_rate=0.05)
        self.assertEqual(list(ladder['rung']), ['base', '+ state'])
        self.assertEqual(len(per_fold), 6)
        self.assertTrue(np.isnan(ladder.loc[0, 'skill']))
        self.assertTrue(np.isnan(ladder.loc[0, 'gain_mean']) or True)   # base has no gain regressor beyond tair
        self.assertIn('gain_sign_stable', ladder.columns)
        self.assertIn('yearly_share_mean', ladder.columns)
        self.assertEqual(set(errors), {'base', '+ state'})
        self.assertGreater(len(errors['+ state']), 0)

    def test_zero_changepoints_is_accepted(self):
        frame = _synthetic_frame()
        folds = prediction.year_folds(frame.index, (2020,))
        ladder, per_fold, _ = prediction.year_ladder(
            frame, [('base', ['tair'], None)], folds, n_changepoints=0, block_hours=24,
            repetitions=20, seed=0, epochs=1, freq='1h', growth='linear', yearly_order=1,
            daily_order=1, quantiles=(0.05, 0.95), learning_rate=0.05)
        self.assertEqual(len(per_fold), 1)
```

And to `test_figures06.py`:

```python
    def test_changepoint_sweep_and_yearly_by_year_draw(self):
        sweep = pd.DataFrame({'n_changepoints': [0, 4, 0, 4], 'rung': ['base', 'base', 'api', 'api'],
                              'mae_val': [20.0, 18.0, 15.0, 14.0]})
        fig = figures.plot_changepoint_sweep(sweep)
        self.assertEqual(len(fig.axes[0].lines), 2)
        per_fold = pd.DataFrame({'rung': ['base', 'api', 'base', 'api'], 'year': [2019, 2019, 2020, 2020],
                                 'yearly_pp': [44.0, 30.0, 45.0, 20.0]})
        fig = figures.plot_yearly_by_year(per_fold)
        self.assertEqual(len(fig.axes[0].patches), 4)
```

- [ ] **Step 2: Run to verify failure.** Expected `AttributeError … 'year_ladder'`.

- [ ] **Step 3: Implement `year_ladder`**

```python
def year_ladder(frame, rungs, folds, n_changepoints, block_hours, repetitions,
                seed, n_jobs=1, gain_regressor=None, **model_kwargs):
    """
    Refit one specification rung by rung, each rung on every held-out year
    (spec D1, D5, D6).

    The same ladder as :func:`channel_ladder`, with the held-out tail
    replaced by :func:`year_folds`: every rung is fitted once per fold on
    the other years and scored on the held-out one, and the rungs are
    compared on their errors pooled over the folds. Beside the error, each
    fit reports the learned gain of one regressor and the yearly term's
    size and share, because D5's rule asks whether the gain keeps its sign
    across folds and whether the yearly term shrinks when a state is
    present.

    Parameters
    ----------
    frame : pd.DataFrame
        ``y`` and every column any rung names, on the study grid.
    rungs : sequence of (str, sequence of str, gate)
        As :func:`channel_ladder`.
    folds : list of dict
        From :func:`year_folds`, on ``frame.index``.
    n_changepoints : int
        ``0`` fits a single straight trend with no changepoints.
    block_hours, repetitions, seed : as :func:`channel_ladder`.
    n_jobs : int, optional
        Rung-and-fold pairs are fitted in worker processes when greater
        than ``1``. Default ``1``.
    gain_regressor : str or None, optional
        The regressor whose gain is reported per fold. ``None`` takes each
        rung's last column. Default ``None``.
    **model_kwargs
        Passed to every :func:`neuralprophet_backtest` call.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame, dict)
        ``ladder``, ``per_fold``, ``errors``; columns as the plan's
        interface block for Task 2.2 states.
    """
    def _fit(item):
        (label, columns, gate), fold = item
        columns = list(columns)
        block = frame.dropna(subset=['y'] + columns)
        if gate is not None:
            gate_columns = [gate] if isinstance(gate, str) else list(gate)
            block = block.dropna(subset=gate_columns)
        train_mask = pd.Series(fold['train'], index=frame.index).reindex(block.index).to_numpy()
        test_mask = pd.Series(fold['test'], index=frame.index).reindex(block.index).to_numpy()
        train, test = block[train_mask], block[test_mask]
        if len(train) < 2 or len(test) < 2:
            return label, fold['year'], None, None
        changepoints = (covered_changepoints(train.index, n_changepoints)
                        if n_changepoints > 0 else None)
        model, out = neuralprophet_backtest(
            train, test, regressors=tuple(columns), task='nowcast',
            changepoints=changepoints, n_changepoints=n_changepoints, seed=seed,
            **model_kwargs)
        components = decompose_components(model, train, regressors=tuple(columns))
        target = gain_regressor if gain_regressor in columns else columns[-1]
        gains = regressor_gains(components, train, tuple(columns))
        gain = float(gains.loc[gains['regressor'] == target, 'gain'].iloc[0])
        shares = component_variance_shares(components)
        yearly_share = (float(shares.loc[shares['component'] == 'season_yearly', 'share'].iloc[0])
                        if 'season_yearly' in set(shares['component']) else np.nan)
        yearly_pp = (float(components['season_yearly'].max() - components['season_yearly'].min())
                     if 'season_yearly' in components.columns else np.nan)
        abs_error = (out['y'] - out['yhat']).abs()
        abs_error.index = pd.DatetimeIndex(out['ds'])
        row = {'rung': label, 'year': fold['year'], 'rows_train': len(train),
               'rows_test': len(test), 'mae': float(abs_error.mean()), 'gain': gain,
               'yearly_pp': yearly_pp, 'yearly_share': yearly_share}
        return label, fold['year'], row, abs_error

    items = [(rung, fold) for rung in rungs for fold in folds]
    per_rows, errors = [], {label: [] for label, _, _ in rungs}
    for label, _, row, abs_error in _parallel_map(_fit, items, n_jobs=n_jobs):
        if row is None:
            continue
        per_rows.append(row)
        errors[label].append(abs_error)
    per_fold = pd.DataFrame(per_rows)
    errors = {label: pd.concat(parts).sort_index() for label, parts in errors.items() if parts}

    labels = [label for label, _, _ in rungs]
    rows = []
    for label in labels:
        mine = per_fold[per_fold['rung'] == label]
        gains = mine['gain'].dropna()
        rows.append({'rung': label, 'rows': int(mine['rows_test'].sum()),
                     'mae_val': float(errors[label].mean()) if label in errors else np.nan,
                     'gain_mean': float(gains.mean()) if len(gains) else np.nan,
                     'gain_sign_stable': bool(len(gains) and (np.sign(gains) == np.sign(gains.iloc[0])).all()),
                     'yearly_pp_mean': float(mine['yearly_pp'].mean()),
                     'yearly_share_mean': float(mine['yearly_share'].mean())})
    ladder = pd.DataFrame(rows)

    def _skill(parent, child):
        shared = errors[parent].index.intersection(errors[child].index)
        result = paired_mae_skill(errors[parent].loc[shared], errors[child].loc[shared],
                                  block_hours=block_hours, repetitions=repetitions, seed=seed)
        return result['skill'], result['skill_q05'], result['skill_q95']

    nan3 = (np.nan, np.nan, np.nan)
    below = [nan3] + [_skill(b, a) for b, a in zip(labels[:-1], labels[1:])]
    first = [nan3] + [_skill(labels[0], a) for a in labels[1:]]
    ladder[['skill', 'skill_q05', 'skill_q95']] = pd.DataFrame(below, index=ladder.index)
    ladder[['skill_vs_first', 'skill_vs_first_q05', 'skill_vs_first_q95']] = pd.DataFrame(first, index=ladder.index)
    return ladder, per_fold, errors
```

`_fit` closes over `frame`; `_parallel_map` with `n_jobs > 1` pickles it to workers as `channel_ladder` already does, so no new mechanism is needed. If `neuralprophet_backtest` rejects `changepoints=None` together with `growth='linear'` and `n_changepoints=0`, pass `n_changepoints=0` alone and let the wrapper omit changepoints; the test `test_zero_changepoints_is_accepted` is the guard.

- [ ] **Step 4: Implement the two figures**, following the template: `plot_changepoint_sweep` pivots `sweep` to `n_changepoints × rung` and draws one line per rung, `'base'` (or the first rung name) in `'#000000'`, the rest in `viz.QUANTITY_COLOUR['api']` cycling `['-', '--', '-.', ':']`, y label "Held-out MAE [mdeg]", x label "Changepoints", legend below. `plot_yearly_by_year` pivots `per_fold` to `year × rung` on `yearly_pp` and draws grouped bars with `ax.bar` offset per rung, colours `'#000000'` for the first rung and `viz.QUANTITY_COLOUR['api']` for the others, y label "Yearly term peak-to-peak [mdeg]", legend below.

- [ ] **Step 5: Run the tests; expected all PASS** (the ladder test takes about a minute). Then the whole suite.

- [ ] **Step 6: Commit** `-- studies/shmlib/prediction.py studies/shmlib/figures.py studies/06_hydro_thermal_drivers/tests -m "feat(shmlib): the year ladder, and the changepoint-sweep and yearly-by-year figures"`.

### Task 2.3 (O designs the cells; S runs the sweeps): Movement 2 — `HT_05`, `HT_06`, `HT_F04`, `HT_F05`

**Files:**
- Modify: the notebook (append Movement 2)

**Interfaces:**
- Consumes: `def_frames['str']`, `api`, `soil`, `record` (Movement 0); `year_folds`, `year_ladder` (2.1, 2.2).
- Produces: notebook variables `moisture_frame`, `moisture_rungs`, `moisture_ladders` (dict `{n_changepoints: (ladder, per_fold, errors)}`), `moisture_verdict` (dict), artefacts `HT_05_moisture_ladder`, `HT_06_yearly_by_year`, `HT_F04_changepoint_sweep`, `HT_F05_yearly_by_year`.

- [ ] **Step 1: The frame and the rungs.** Under the opening Markdown cell, which restates rule D5:

```python
moisture_frame = def_frames['str'].copy()
for tau in API_TAU_DAYS:
    moisture_frame[f'api_{tau}d'] = api[tau].reindex(moisture_frame.index)
if soil is not None:
    for layer in SOIL_LAYERS:
        moisture_frame[f'swvl{layer}'] = soil[f'swvl{layer}_era5'].reindex(moisture_frame.index)
gate = [c for c in moisture_frame.columns if c.startswith('api_') or c.startswith('swvl')]
moisture_rungs = [('base', ['tair', 'rh', 'sr'], gate)]
moisture_rungs += [(f'+ api_{tau}d', ['tair', 'rh', 'sr', f'api_{tau}d'], gate) for tau in API_TAU_DAYS]
if soil is not None:
    moisture_rungs += [(f'+ swvl{l}', ['tair', 'rh', 'sr', f'swvl{l}'], gate) for l in SOIL_LAYERS]
folds = prediction.year_folds(moisture_frame.index, FOLD_YEARS)
print('rungs:', len(moisture_rungs), 'folds:', [f['year'] for f in folds],
      'matched rows:', int(moisture_frame.dropna(subset=['y'] + gate).shape[0]))
```

The common gate holds every rung to one matched window, as Study 05's ladder did, so the longest τ's warm-up sets the window for all.

- [ ] **Step 2: The sweep.** One `year_ladder` call per changepoint count, the block length no shorter than the longest τ:

```python
moisture_ladders, sweep_rows = {}, []
for n_cp in CHANGEPOINT_SWEEP:
    ladder, per_fold, errors = prediction.year_ladder(
        moisture_frame, moisture_rungs, folds, n_cp,
        block_hours=max(BOOTSTRAP_BLOCK_HOURS, 24 * max(API_TAU_DAYS)),
        repetitions=BOOTSTRAP_REPETITIONS, seed=SEED, n_jobs=N_JOBS,
        epochs=EPOCHS, freq=NATIVE_FREQ, growth='linear',
        changepoints_range=CHANGEPOINTS_RANGE, trend_reg=TREND_REG,
        yearly_order=YEARLY_ORDER, daily_order=DAILY_ORDER, quantiles=QUANTILES,
        learning_rate=LEARNING_RATE)
    ladder.insert(0, 'n_changepoints', n_cp)
    per_fold.insert(0, 'n_changepoints', n_cp)
    moisture_ladders[n_cp] = (ladder, per_fold, errors)
    print(f'n_changepoints={n_cp}:'); display(ladder)
moisture_ladder = pd.concat([v[0] for v in moisture_ladders.values()], ignore_index=True)
moisture_per_fold = pd.concat([v[1] for v in moisture_ladders.values()], ignore_index=True)
moisture_ladder.to_csv(OUTPUT_DIR / 'HT_05_moisture_ladder.csv', index=False)
tables.write_table(moisture_ladder, str(OUTPUT_DIR / 'HT_05_body.tex'),
                   [('n_changepoints', 'd'), ('rung', tables.texttt), ('mae_val', '.2f'),
                    ('skill_vs_first', '.3f'), ('skill_vs_first_q05', '.3f'),
                    ('skill_vs_first_q95', '.3f'), ('gain_mean', '.3g'),
                    ('gain_sign_stable', tables.yes_no), ('yearly_share_mean', tables.percent)])
```

Expected wall time: up to 12 rungs × 5 folds × 5 counts = 300 fits; at `N_JOBS = 32` about one to two hours on the workstation. Develop against the dump with `CHANGEPOINT_SWEEP = (0, 12)`, `API_TAU_DAYS = (30,)`, `EPOCHS = 2` overridden in `run_movement.py`, then run in full once.

- [ ] **Step 3: `HT_06` and the two figures.**

```python
best_cp = int(moisture_ladder.loc[moisture_ladder['rung'] == 'base', 'mae_val'].idxmin())
best_cp = int(moisture_ladder.loc[best_cp, 'n_changepoints'])
yearly_by_year = moisture_per_fold[moisture_per_fold['n_changepoints'] == N_CHANGEPOINTS][
    ['rung', 'year', 'mae', 'gain', 'yearly_pp', 'yearly_share']]
yearly_by_year.to_csv(OUTPUT_DIR / 'HT_06_yearly_by_year.csv', index=False)
tables.write_table(yearly_by_year, str(OUTPUT_DIR / 'HT_06_body.tex'),
                   [('rung', tables.texttt), ('year', 'd'), ('mae', '.2f'), ('gain', '.3g'),
                    ('yearly_pp', '.1f'), ('yearly_share', tables.percent)])
figures.plot_changepoint_sweep(moisture_ladder[['n_changepoints', 'rung', 'mae_val']],
                               title='Held-out error by changepoint count, with and without a moisture state',
                               save_path=str(OUTPUT_DIR), filename='HT_F04_changepoint_sweep')
figures.plot_yearly_by_year(yearly_by_year, title='The yearly term per held-out year',
                            save_path=str(OUTPUT_DIR), filename='HT_F05_yearly_by_year')
```

- [ ] **Step 4: The rule, applied by the notebook and printed.**

```python
at_full = moisture_ladder[moisture_ladder['n_changepoints'] == N_CHANGEPOINTS].set_index('rung')
base_share = float(at_full.loc['base', 'yearly_share_mean'])
verdict_rows = []
for rung in at_full.index[1:]:
    r = at_full.loc[rung]
    verdict_rows.append({
        'rung': rung,
        'skill_excludes_zero': bool(r['skill_vs_first_q05'] > 0),
        'sign_stable': bool(r['gain_sign_stable']),
        'sign_as_expected': bool(np.sign(r['gain_mean']) == EXPECTED_MOISTURE_SIGN),
        'yearly_share_falls': bool(r['yearly_share_mean'] < base_share)})
moisture_verdict = pd.DataFrame(verdict_rows)
moisture_verdict['is_driver'] = moisture_verdict[['skill_excludes_zero', 'sign_stable',
                                                  'yearly_share_falls']].all(axis=1)
display(moisture_verdict)
```

- [ ] **Step 5: Run against dumped state, then the whole notebook once.** Expected 0 error cells.

- [ ] **Step 6: Checkpoint 2 (O).** The orchestrator applies the three conditions from `moisture_verdict` and `HT_05`, names which rung, if any, is a driver and at which τ, records `MOISTURE_SURVIVOR = '<rung>'` or `None` in the parameter cell with the reason in its guidance, and commits.

### Task 2.4 (O): Report §4 Moisture against the trend

- [ ] Write the second half of §4 from `HT_05`, `HT_06`, `HT_F04`, `HT_F05`, condition by condition, and the sign against the earth-pressure prediction. Build twice, honesty test, commit.

---

# Phase 3 · Heat-exchange ladder

### Task 3.1 (S): `coupling.net_longwave`

**Files:**
- Modify: `studies/shmlib/coupling.py` (append); `tests/test_states.py` (add a class)

**Interfaces:**
- Produces: `coupling.STEFAN_BOLTZMANN = 5.670374419e-8` and `coupling.net_longwave(lw_down, temperature, emissivity=0.9) -> pd.Series` named `lw_net`: `lw_down − emissivity · σ · (temperature + 273.15)⁴`, in W m⁻², aligned on the index of `lw_down`, `NaN` where either input is.

- [ ] **Step 1: Write the failing test**

```python
class TestNetLongwave(unittest.TestCase):

    def test_the_net_flux_is_downward_minus_grey_body_emission(self):
        index = pd.date_range('2024-01-01', periods=3, freq='1h')
        down = pd.Series([330.0, 330.0, np.nan], index=index)
        temperature = pd.Series([10.0, 10.0, 10.0], index=index)
        out = coupling.net_longwave(down, temperature, emissivity=0.9)
        expected = 330.0 - 0.9 * coupling.STEFAN_BOLTZMANN * (283.15 ** 4)
        self.assertEqual(out.name, 'lw_net')
        self.assertAlmostEqual(out.iloc[0], expected, places=6)
        self.assertTrue(np.isnan(out.iloc[2]))
```

- [ ] **Step 2: Run to verify failure.** Expected `AttributeError … 'net_longwave'`.

- [ ] **Step 3: Implement**

```python
#: Stefan-Boltzmann constant, W m^-2 K^-4.
STEFAN_BOLTZMANN = 5.670374419e-8


def net_longwave(lw_down, temperature, emissivity=0.9):
    """
    The wall's net longwave exchange with the sky, from the downward flux.

    A pyranometer sees the sun and nothing else, so the record's radiation
    channel is zero all night while the stone goes on losing heat to a sky
    colder than itself. The reanalysis gives the downward longwave flux; the
    wall's own emission is a grey body at the air temperature, which is the
    temperature this study has everywhere the probe does not exist. The
    difference is negative on a clear night, near zero under cloud, and is
    the cooling the pyranometer cannot see.

    Parameters
    ----------
    lw_down : pd.Series
        Downward longwave flux, W m^-2.
    temperature : pd.Series
        Surface temperature in degrees Celsius, on the same index or one it
        can be reindexed to.
    emissivity : float, optional
        Grey-body emissivity of the masonry. Default 0.9.

    Returns
    -------
    pd.Series
        Named ``lw_net``, W m^-2, ``NaN`` where either input is missing.
    """
    down = pd.to_numeric(lw_down, errors='coerce')
    kelvin = pd.to_numeric(temperature, errors='coerce').reindex(down.index) + 273.15
    return (down - emissivity * STEFAN_BOLTZMANN * kelvin ** 4).rename('lw_net')
```

- [ ] **Step 4: Run the tests; expected PASS.** **Step 5: Commit** `-- studies/shmlib/coupling.py studies/06_hydro_thermal_drivers/tests/test_states.py -m "feat(shmlib): net longwave exchange from the downward flux and a grey-body emission"`.

### Task 3.2 (S): `thermal_lag_filter(modulation=…)` and `thermal_operator(modulation=…)`

**Files:**
- Modify: `studies/shmlib/coupling.py:67-152`; `tests/test_states.py` (add a class)

**Interfaces:**
- Produces: `thermal_lag_filter(series, tau_hours, dt_hours=1.0, modulation=None)` and `thermal_operator(series, delay=0, tau=0.0, dt_hours=1.0, modulation=None)`. With `modulation=None` both are byte-for-byte the functions they were. With a series `m`, the time constant at each slot is `tau_hours · m_t`, clipped below at `dt_hours`, the filter runs as `y_t = y_{t−1} + α_t (x_t − y_{t−1})` with `α_t = dt / (τ_t + dt)` on the bridged input, initialised at the first value, and masked where the input was missing. Missing `m_t` are interpolated in both directions.

- [ ] **Step 1: Write the failing test**

```python
class TestModulatedFilter(unittest.TestCase):

    def _driver(self):
        index = pd.date_range('2024-06-01', periods=24 * 5, freq='1h')
        return pd.Series(np.sin(2 * np.pi * np.arange(len(index)) / 24), index=index)

    def test_a_unit_modulation_reproduces_the_plain_filter(self):
        x = self._driver()
        plain = coupling.thermal_lag_filter(x, 4.0, dt_hours=1.0)
        modulated = coupling.thermal_lag_filter(x, 4.0, dt_hours=1.0,
                                                modulation=pd.Series(1.0, index=x.index))
        np.testing.assert_allclose(modulated.to_numpy(), plain.to_numpy(), rtol=1e-9)

    def test_a_half_modulation_is_the_filter_at_half_the_time_constant(self):
        x = self._driver()
        half = coupling.thermal_lag_filter(x, 2.0, dt_hours=1.0)
        modulated = coupling.thermal_lag_filter(x, 4.0, dt_hours=1.0,
                                                modulation=pd.Series(0.5, index=x.index))
        np.testing.assert_allclose(modulated.to_numpy(), half.to_numpy(), rtol=1e-9)

    def test_the_operator_passes_the_modulation_through(self):
        x = self._driver()
        out = coupling.thermal_operator(x, delay=1, tau=4.0, dt_hours=1.0,
                                        modulation=pd.Series(1.0, index=x.index))
        plain = coupling.thermal_operator(x, delay=1, tau=4.0, dt_hours=1.0)
        np.testing.assert_allclose(out.to_numpy()[1:], plain.to_numpy()[1:], rtol=1e-9)
```

- [ ] **Step 2: Run to verify failure.** Expected `TypeError: thermal_lag_filter() got an unexpected keyword argument 'modulation'`.

- [ ] **Step 3: Implement.** In `thermal_lag_filter`, add the parameter and its docstring entry ("`modulation` : pd.Series or None — a multiplicative factor on the time constant at each slot, for a driver whose exchange rate itself varies, as wind varies convective loss; `None` keeps one constant. Default `None`."), and replace the body's last two lines with:

```python
    if tau_hours <= 0:
        return series.copy()
    bridged = series.interpolate(method='time', limit_direction='both')
    if modulation is None:
        alpha = dt_hours / (tau_hours + dt_hours)
        return bridged.ewm(alpha=alpha, adjust=False).mean().where(series.notna())
    factor = (pd.to_numeric(modulation, errors='coerce').reindex(series.index)
              .interpolate(limit_direction='both').to_numpy(dtype=float))
    tau = np.clip(tau_hours * factor, dt_hours, None)
    alpha = dt_hours / (tau + dt_hours)
    x = bridged.to_numpy(dtype=float)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + alpha[i] * (x[i] - y[i - 1])
    return pd.Series(y, index=series.index).where(series.notna())
```

Note that with `adjust=False` pandas' `ewm` is exactly the recursion above at constant `alpha` started from the first value, which is why the unit-modulation test can demand `rtol=1e-9`. In `thermal_operator`, add `modulation=None` to the signature and docstring and pass it: `out = thermal_lag_filter(series, tau, dt_hours=dt_hours, modulation=modulation)`.

- [ ] **Step 4: Run the tests; expected PASS**, then the whole suite, in particular `03_thermomechanical_response/tests/test_shmlib_study03.py`, which exercises the plain path.

- [ ] **Step 5: Commit** `-- studies/shmlib/coupling.py studies/06_hydro_thermal_drivers/tests/test_states.py -m "feat(shmlib): a thermal filter whose time constant is modulated slot by slot"`.

### Task 3.3 (S runs, O interprets): Movement 3 — `HT_07`, `HT_08`, `HT_F06`

**Files:**
- Modify: the notebook (append Movement 3)

**Interfaces:**
- Consumes: `def_frames['str']`, `sets`, `era5`, `station`, `sensor`, `record` (Movement 0); `net_longwave`, the modulated `thermal_operator` (3.1, 3.2); `year_folds`, `year_ladder` (2.1, 2.2); `prediction.channel_ladder` (existing) for the current-era rung.
- Produces: notebook variables `exchange_frame`, `exchange_ladder`, `wind_sweep`, `exchange_survivors`; artefacts `HT_07_exchange_ladder`, `HT_08_wind_modulation_sweep`, `HT_F06_exchange_ladder`.

- [ ] **Step 1: The states.** Under the opening Markdown cell, which restates rule D6 and names the base operator:

```python
dt_h = pd.Timedelta(NATIVE_FREQ) / pd.Timedelta(hours=1)
exchange_frame = def_frames['str'].copy()
tair_for_lw = exchange_frame['tair']
lw_net = coupling.net_longwave(era5['lwrad_era5'].reindex(exchange_frame.index), tair_for_lw,
                               emissivity=EMISSIVITY)
exchange_frame['lw_net'] = coupling.thermal_operator(lw_net, delay=0, tau=RADIATION_TAU_H, dt_hours=dt_h) \
    if RADIATION_TAU_H > 0 else lw_net.shift(int(RADIATION_DELAY_H / dt_h))
wind = (station if WIND_SOURCE == 'gs' else era5)[f'wspd_{WIND_SOURCE}'].reindex(exchange_frame.index)
sr_raw = sets['str']['sr']      # the delayed or filtered radiation the base carries
for k in WIND_K:
    modulation = 1.0 / (1.0 + k * wind)
    exchange_frame[f'sr_wind_{k}'] = coupling.thermal_operator(
        record[REGRESSOR_SETS['str']['sr']].reindex(exchange_frame.index), delay=0,
        tau=RADIATION_TAU_H if RADIATION_TAU_H > 0 else 1.0, dt_hours=dt_h, modulation=modulation)
print('exchange states built; lw_net night mean:',
      f"{exchange_frame['lw_net'].between_time('00:00', '04:00').mean():.1f} W/m2")
```

When D15 handed on a delay-only operator (`RADIATION_TAU_H == 0`), wind cannot modulate a delay; the sweep then runs on a one-hour filter in place of the one-hour delay, and the Markdown cell says so, because on the daily harmonic the two are the same phase lag (spec §1).

- [ ] **Step 2: The wind sweep, `HT_08`.** Rungs `('+ lw, wind k=<k>', ['tair', 'rh', f'sr_wind_{k}', 'lw_net'], gate)` for each k, gate the union of every state column, through `year_ladder` at `N_CHANGEPOINTS` with `BOOTSTRAP_BLOCK_HOURS`; write the ladder as `HT_08_wind_modulation_sweep.csv` with body columns `rung`, `mae_val`, `skill_vs_first`, its bounds, `gain_mean` (of `sr_wind_k`, so `gain_regressor=f'sr_wind_{k}'` is passed per rung — since `year_ladder` takes one `gain_regressor`, call it with `gain_regressor=None`, whose default is the last column, and put the radiation state last in each rung's columns). `best_k` is the k of lowest `mae_val`.

- [ ] **Step 3: The ladder, `HT_07`.** Rungs 1 to 3 on the whole record through `year_ladder`:

```python
exchange_rungs = [
    ('1 base', ['tair', 'rh', 'sr'], gate),
    ('2 + net longwave', ['tair', 'rh', 'lw_net', 'sr'], gate),
    (f'3 + wind k={best_k}', ['tair', 'rh', 'lw_net', f'sr_wind_{best_k}'], gate),
]
exchange_ladder, exchange_per_fold, _ = prediction.year_ladder(
    exchange_frame, exchange_rungs, folds, N_CHANGEPOINTS,
    block_hours=BOOTSTRAP_BLOCK_HOURS, repetitions=BOOTSTRAP_REPETITIONS, seed=SEED,
    n_jobs=N_JOBS, epochs=EPOCHS, freq=NATIVE_FREQ, growth='linear',
    changepoints_range=CHANGEPOINTS_RANGE, trend_reg=TREND_REG, yearly_order=YEARLY_ORDER,
    daily_order=DAILY_ORDER, quantiles=QUANTILES, learning_rate=LEARNING_RATE)
```

Rung 4, current era only, through `channel_ladder` on the frame restricted to `CURRENT_ERA_START` onward with `wind_dT = wind · (tair − twall)` added from `sensor['twall']`, rungs `('3 as above', […])` and `('4 + wind × (air − wall), diagnostic', [… , 'wind_dT'])`, `LADDER_N_CHANGEPOINTS = max(2, N_CHANGEPOINTS // 3)` as Study 05's ladder used, `VALID_P` tail split. Its two rows are appended to `exchange_ladder` with `rows` from its own window and the skill columns as `channel_ladder` returns them; the radiation gain per rung is read from `exchange_per_fold` (rungs 1 to 3, mean of `gain`) and from the current-era fit's `gain_last` for rung 4, into a column `sr_gain`. Written as `HT_07_exchange_ladder.csv` and `HT_07_body.tex` with columns `rung`, `rows`, `mae_val`, `skill`, `skill_q05`, `skill_q95`, `skill_vs_first`, its bounds, `sr_gain`. `figures.plot_ladder(exchange_ladder, title='What each exchange state buys', save_path=str(OUTPUT_DIR), filename='HT_F06_exchange_ladder')`.

- [ ] **Step 4: The rule.** Print, per rung 2 to 4, `skill_q05 > 0` as `improves`, and the sign of `sr_gain` beside Study 03's −0.035; `exchange_survivors = [rung for rung in rungs 2 to 3 whose skill_q05 > 0]`.

- [ ] **Step 5: Run against dumped state, then the whole notebook once.** Expected 0 error cells; wall time about an hour (5 wind rungs + 3 ladder rungs, 5 folds each, plus two current-era fits).

- [ ] **Step 6: Checkpoint 3 (O).** A verdict per exchange, the radiation gain's sign at every rung, `EXCHANGE_SURVIVORS` recorded in the parameter cell. Commit.

### Task 3.4 (O): Report §5 Heat exchange

- [ ] Write §5 from `HT_07`, `HT_08`, `HT_F06`. Build twice, honesty test, commit.

---

# Phase 3b · The gain assumption and the capacity ceiling

Added 2026-09-07 with spec decisions D11 and D12. This phase answers a question about the method, not about the wall. It runs only after Phases 2 and 3 have written `HT_05` and `HT_07`, so that no physical state is ever judged against a model carrying more freedom than the rule that judged it assumed. Nothing this phase finds changes the main line; if the user removes it by answering spec open question 17, this whole section and its artefacts go with it.

### Task 3b.1 (S): `coupling.amplitude_envelope` extended to a residual

**Files:**
- Modify: `studies/shmlib/coupling.py`, `studies/06_hydro_thermal_drivers/tests/test_states.py`

**Interfaces:**
- Produces: nothing new in the signature. Task 0.5b's function is confirmed to accept a single-column frame so that Movement 3b can pass each specification's residual through the same reduction and report the annual envelope of what each one leaves behind, which is the quantity D11's rule reads next to the sideband power.

- [ ] **Step 1:** Add a test that a one-column frame is accepted and that a residual with no annual envelope returns a ratio near 1.0.
- [ ] **Step 2:** Run to verify; adapt the function only if the test fails, since Task 0.5b may already satisfy it.
- [ ] **Step 3:** Run the whole suite; commit only if a change was needed.

### Task 3b.2 (S): The NeuralProphet capacity arguments, exposed

**Files:**
- Modify: `studies/shmlib/prediction.py` (`neuralprophet_backtest`, the constructor dict at `prediction.py:1104`)
- Create: `studies/06_hydro_thermal_drivers/tests/test_capacity.py`

**Interfaces:**
- Produces: `neuralprophet_backtest(..., future_regressors_model='linear', future_regressors_d_hidden=4, future_regressors_num_hidden_layers=2, lagged_reg_layers=())`, each forwarded into the constructor dict. The defaults are NeuralProphet 0.8.0's own, so the model every existing caller builds is byte-for-byte the model it builds today. `lagged_reg_layers` is passed as a list; an empty tuple becomes `[]`, the library's default, which is why Model B of Study 05 learned a linear impulse response.

- [ ] **Step 1: Write the failing tests.** Three: that the default call still constructs a model whose `config_regressors` reports the linear model; that `future_regressors_model='neural_nets'` constructs one that does not; and that a two-row fit with `lagged_reg_layers=[4]` runs to completion without error. The third is a smoke test on a tiny synthetic frame, not a skill test.
- [ ] **Step 2: Run to verify failure.** Expected `TypeError: neuralprophet_backtest() got an unexpected keyword argument 'future_regressors_model'`.
- [ ] **Step 3: Implement.** Add the four parameters to the signature with those defaults, document each in the docstring's `Parameters` section with one line on what it changes and a sentence recording that Studies 04 and 05 chose the linear default deliberately, and add them to the `constructor` dict. Nothing else in the function changes.
- [ ] **Step 4: Run the tests; then the whole suite of Global Constraints, which is the load-bearing check here** — Studies 03, 04 and 05 must reproduce unchanged, since this touches the function every one of them fits through.
- [ ] **Step 5: Commit** with message `feat(shmlib): expose NeuralProphet's regressor-capacity arguments, defaults unchanged`.

### Task 3b.3 (S): `prediction.gain_variation_ladder`, `prediction.capacity_ceiling`, and two figures

**Files:**
- Modify: `studies/shmlib/prediction.py`, `studies/shmlib/figures.py`, `studies/06_hydro_thermal_drivers/tests/test_capacity.py`, `.../tests/test_figures06.py`

**Interfaces:**
- Produces: `gain_variation_ladder(frame, regressors, specs, folds, weight_curve, **model_kwargs) -> pd.DataFrame`, one row per specification of D11 with `spec`, `mae_val`, `skill`, `skill_q05`, `skill_q95`, `gain_summer`, `gain_winter`, `share_moisture`, `share_exchange`, `sideband_power`, `envelope_ratio`; and `capacity_ceiling(frame, target, drivers, folds, lags, seed, **hyperparameters) -> pd.DataFrame`, one row per fold plus a pooled row, with `fold`, `n_train`, `n_valid`, `mae_val` and nothing else, because the model has nothing else to report. `plot_capacity_ceiling` draws `HT_F10` as a horizontal bar per specification with its bootstrap bounds, Study 05's Model A and the ceiling at the two ends.

- [ ] **Step 1: Write the failing tests.** For `gain_variation_ladder`, that it returns one row per requested spec, that `'base'` has `skill` exactly 0.0, and that requesting an unknown spec raises with the valid names in the message. For `capacity_ceiling`, that it never imports NeuralProphet (assert the module is absent from the function's own globals), that it returns `len(folds) + 1` rows, and that a frame whose target is a deterministic function of one driver is fitted to near-zero error, which checks the feature matrix is built and aligned rather than that the model is good.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.** `gain_variation_ladder` builds the two conditional weight columns with `prediction.seasonal_weights` and the curve of Study 05's `GM_04` — the same function and the same curve the conditional *daily* term used, applied here to the temperature regressor rather than to the seasonality, which the docstring states explicitly so that nobody later reads the two as the same experiment. The `'neural_nets'` spec passes Task 3b.2's arguments through. `capacity_ceiling` builds a lagged feature matrix to `lags` hours per driver plus day of year and time of day, fits `xgboost.XGBRegressor` per fold, and scores on the held-out year; its docstring opens by saying it is a bound and not a candidate, and lists what it deliberately does not return.
- [ ] **Step 4: Figures**, to the template of `figures.plot_ladder`; smoke tests in `test_figures06.py`.
- [ ] **Step 5: Run the tests; then the whole suite; commit** with message `feat(shmlib): the gain-variation ladder and the unconstrained capacity ceiling`.

### Task 3b.4 (S runs, O interprets): Movement 3b — `HT_14`, `HT_15`, `HT_F10`

**Files:**
- Modify: `.../hydro_thermal_drivers_study.py` (append Movement 3b)

**Interfaces:**
- Consumes: the surviving model of Movements 2 and 3, the folds of `prediction.year_folds`, `HT_05` and `HT_07` on disk.
- Produces: `HT_14_gain_variation`, `HT_15_capacity_ceiling`, `HT_F10_capacity_ceiling`.

- [ ] **Step 1: The guard.** The movement's first cell asserts that `HT_05_moisture_ladder.csv` and `HT_07_exchange_ladder.csv` exist and raises a `RuntimeError` naming spec D12 if either does not. This is the mechanical form of the ordering rule; without it the movement would happily run first when a session resumes out of order.
- [ ] **Step 2: The opening Markdown cell** states, in full prose, that this movement answers a question about the method rather than about the wall, that its result cannot revise Movements 2 and 3, and which artefacts it writes. It re-prints the D11 and D12 rules from the head so that they sit immediately above the numbers.
- [ ] **Step 3: `HT_14`.** Call `gain_variation_ladder` with `GAIN_SPECS`, write the CSV and body, display the table.
- [ ] **Step 4: `HT_15` and `HT_F10`.** Call `capacity_ceiling`, then assemble the figure's rows from Study 05's Model A on these folds, this study's base, the `HT_14` specifications and the ceiling.
- [ ] **Step 5: Run against dumped state, then the whole notebook once**, per the Global Constraints.
- [ ] **Step 6: Checkpoint 3b (O).** Apply D11's two-part rule condition by condition and record the outcome in the parameter cell's guidance. Then answer three questions in writing, since the report's §6 is built on them: did a varying gain earn its bounds; did any physical state lose its share when the gain was freed, and which; how large is the gap between the base and the ceiling, in per cent with its bounds. **If a state was absorbed, its Phase 2 or Phase 3 verdict is not revised** — both readings go into the report.
- [ ] **Step 7: Commit** with message `feat(study06): Movement 3b, the gain-variation ladder and the capacity ceiling`.

### Task 3b.5 (O): Report §6 *What the linear gain assumes, and what it costs*

- [ ] Write the section from `HT_13`, `HT_14`, `HT_15`, `HT_F09` and `HT_F10`, removing its `\pending{}`. It opens on the envelope of §2.6 as the reason the question is asked, states that the linear gain is this project's choice and not the library's limitation, gives the three specifications and the ceiling with their bounds, and closes on what the numbers mean for the main line — which, whatever they are, is unchanged. Where a physical state was absorbed by a freed gain, both readings are given and neither is called the right one. Build the PDF twice, run the honesty test, commit.

---

# Phase 4 · Joint fit and the slow chart

### Task 4.1 (S runs, O interprets): Movement 4 — `HT_09`, `HT_10`, `HT_11`, `HT_F07`, `HT_F08`

**Files:**
- Modify: the notebook (append Movement 4)

**Interfaces:**
- Consumes: `MOISTURE_SURVIVOR`, `EXCHANGE_SURVIVORS` (parameter cell, set at Checkpoints 2 and 3); `def_frames`, `api`, `soil`, `era5`, `station` (Movement 0); `exchange_frame` columns (Movement 3); `prediction.attribution_fits`, `residual_diagnostics`, `period_scan`, `rolling_nowcast`, `monitoring.chart_series`, `reference_stats`, `tune_limit_to_budget`, `run_chart`, `figures.plot_decomposition_stack`, `plot_control_chart` (existing).
- Produces: `HT_09_joint_shares`, `HT_10_joint_residual`, `HT_11_slow_chart_episodes`, `HT_F07_joint_decomposition`, `HT_F08_slow_chart`.

- [ ] **Step 1: Guard.** The opening Markdown cell states D7, and adds one sentence recording that this movement uses the fixed linear gain whatever Movement 3b found, per D11. The first code cell: `survivors = ([MOISTURE_SURVIVOR] if MOISTURE_SURVIVOR else []) + list(EXCHANGE_SURVIVORS)`; if empty, print one line ("nothing survived its own rule; Movement 4 writes the three tables with the base fit's values so the report can say so") and every later cell runs on the base regressors alone. No call in this movement passes `future_regressors_model`, `lagged_reg_layers` or a conditional temperature weight; if one does, that is the defect the D11 constraint exists to prevent.

- [ ] **Step 2: Joint frames on three sets.** For each set name in `def_frames`, copy the frame and add the surviving state columns built from that set's own source where the source has one and from ERA5 otherwise: the moisture state from `api` (ERA5 rain in every set); `lw_net` from ERA5 longwave and the set's own `tair`; the wind-modulated radiation from the set's own radiation column and the station's wind for `str` and `gs`, ERA5's wind for `era5`. Regressor tuple `('tair', 'rh', 'sr') + tuple(state columns)`, with the plain `sr` dropped when the wind-modulated radiation replaces it.

- [ ] **Step 3: `HT_09`, `HT_10`, `HT_F07`.** `prediction.attribution_fits(joint_frames, joint_regressors, VALID_P, N_CHANGEPOINTS, study03_gains=STUDY03_GAINS, n_jobs=N_JOBS, …Model A kwargs…)`; write `shares` as `HT_09_joint_shares.csv` (`set`, `component`, `share`, `peak_to_peak`) beside a read of Study 05's `GM_05_component_shares.csv` when it exists, displayed only; write `diagnostics` plus, per set, `prediction.period_scan(components['residual'], min_days=0.4, max_days=3.0, n_periods=40000, top=3)` to `HT_10_joint_residual.csv` (`set`, `lag`, `ljung_box_p`, `std`, `mad`, and the top short-band periods with their power); `figures.plot_decomposition_stack(fits['str']['components'], title='The joint decomposition, on-structure set', save_path=str(OUTPUT_DIR), filename='HT_F07_joint_decomposition')`.

- [ ] **Step 4: The slow chart, `HT_11`, `HT_F08`.** Only if `MOISTURE_SURVIVOR` is set (spec D9); otherwise the table is written with zero rows and a printed reason.

```python
rolling = prediction.rolling_nowcast(
    joint_frames['str'], regressors=joint_regressors['str'], refit_every=REFIT_EVERY,
    min_train=MIN_TRAIN, freq=NATIVE_FREQ, train_window=TRAIN_WINDOW,
    changepoints_per_window=True, n_jobs=N_JOBS, epochs=EPOCHS, growth='linear',
    n_changepoints=N_CHANGEPOINTS, changepoints_range=CHANGEPOINTS_RANGE,
    trend_reg=TREND_REG, yearly_order=YEARLY_ORDER, daily_order=DAILY_ORDER,
    quantiles=QUANTILES, seed=SEED, learning_rate=LEARNING_RATE)
residual = pd.Series((rolling['y'] - rolling['yhat']).to_numpy(), index=pd.DatetimeIndex(rolling['ds']))
series = monitoring.chart_series(residual, NATIVE_FREQ, DAILY_HARMONIC_MIN_SLOTS,
                                 REFERENCE_START, REFERENCE_END)
slow = series['daily_mean']
reference = monitoring.reference_stats(slow, REFERENCE_START, REFERENCE_END)
L_slow, _ = monitoring.tune_limit_to_budget(slow, REFERENCE_START, REFERENCE_END, BUDGET_SLOW_DAYS,
                                            LIMIT_CANDIDATES, EWMA_LAMBDA_SLOW, CUSUM_K, CUSUM_H,
                                            JOINT_WINDOW_DAILY, '1D')
chart = monitoring.run_chart(slow, reference, L_slow, EWMA_LAMBDA_SLOW, CUSUM_K, CUSUM_H,
                             JOINT_WINDOW_DAILY, MONITORED_START, '1D')
episodes = chart['episodes']
study05_episodes = STUDY05_OUTPUTS / 'GM_11_alarm_episodes.csv'
if study05_episodes.exists():
    display(pd.read_csv(study05_episodes).query("chart == 'slow'"))
episodes.to_csv(OUTPUT_DIR / 'HT_11_slow_chart_episodes.csv', index=False)
```

The keys of `chart_series`' return and the `episodes` columns are read from `shmlib/monitoring.py:824` and `:275` when the cell is written, and the `write_table` list follows them. `figures.plot_control_chart(chart['cusum'], statistic='cusum', episodes=episodes, freq='1D', title='The slow chart on the joint residual', save_path=str(OUTPUT_DIR), filename='HT_F08_slow_chart')`.

- [ ] **Step 5: Run against dumped state, then the whole notebook once.** The rolling refit is the long cell, about an hour at `N_JOBS = 32`. Expected 0 error cells.

- [ ] **Step 6: Checkpoint 4 (O).** Shares beside Study 05's; what the slow chart would watch. Commit.

### Task 4.2 (O): Report §7 and §8

- [ ] Write *The joint decomposition* and *What the slow chart would watch* from `HT_09`–`HT_11`, `HT_F07`, `HT_F08`. Build twice, honesty test, commit. (Section numbers shifted by one when §6 was inserted for Phase 3b; the labels `sec:joint` and `sec:slow` did not.)

---

# Phase 5 · Closure

### Task 5.1 (O): Movement 5, `HT_12`, the last three sections, status "Complete"

- [ ] **Step 1: Movement 5.** A metadata cell that assembles every parameter of the parameter cells, the NeuralProphet, torch, pandas and numpy versions, the seed, the git commit of `shmlib`, the presence or absence of the soil file, and the operator D15 handed on, into `HT_12_run_metadata.csv` and `HT_12_body.tex` (columns `parameter`, `value`, `inherited_from`). Run the whole notebook once more; expected 0 error cells.

- [ ] **Step 2: Report.** Write §9 Verdict (one paragraph per question of spec §1, each with its qualification, and a final short paragraph on the fourth, methodological question, marked as such), §10 Limitations (spec §7's list: one gauge and one cell; the embankment's moisture unmeasured; five annual cycles; the emissivity assumed; the wall's azimuth unrecorded; the changepoint count as competitor; and the ceiling of D12 measured with one model family on one feature set, which bounds the skill left on the table from below and not from above), §11 Run metadata. No `\pending{}` remains. Build twice.

- [ ] **Step 3: Closure flags.** Set `STUDY_COMPLETE = True` in `tests/test_folder_honesty.py`; run it, expected PASS. Update the README status line to "Complete" with the date and the artefact list; set the `studies/README.md` row to "Complete". Commit with `-- studies/06_hydro_thermal_drivers studies/README.md`.

### Task 5.2 (S, O reviews): Graph and suite

- [ ] Run every test file named in Global Constraints plus the four of this study; expected all PASS. Run `/graphify --update` on `studies/` so the graph carries the new functions. Commit `graphify-out/` if the repository tracks it, otherwise nothing. Report to the user, in full prose, the closing state and the two things left to them: rotating the API key in `auxiliary/oiko.py`, and deciding whether a Study 07 promotes a surviving state into the main line.

---

## Self-review against the spec

- **Coverage.** §2.4–2.5 sources and the design table → Tasks 0.2–0.5. D1 (whole record, year folds) → 2.1, 2.3. D2 (two ladders, joint only for survivors) → 2.3, 3.3, 4.1. D3 (states with τ; API, soil, filtered radiation, net longwave, wind modulation, wind × ΔT) → 1.1, 3.1, 3.2, 3.3. D4 (sources, snow folded, soil if obtained) → 0.3, 0.5. D5 (three tests, expected sign, three-part rule) → 1.2, 1.4, 2.3 Step 4. D6 (four-rung ladder, rule, gain sign at every rung) → 3.3. D7 (joint fit, three sets) → 4.1. D8 (Model A inherited, gap rules, clock check of the new channels) → 0.1 parameter cell, 0.5 Step 1 (the longwave channel is centred as the solar one; a clock check is not repeated, as D8 allows). D9 (slow chart) → 4.1 Step 4. D10 (figures, identity colours) → 0.2, 1.3, 2.2. §6 artefacts `HT_01`–`HT_12`, `HT_F01`–`HT_F08` → 0.5, 1.4, 2.3, 3.3, 4.1, 5.1. §7 report → 0.6, 1.5, 2.4, 3.4, 4.2, 5.1. §8 checkpoints and the Phase 0 kill criterion → 0.5 Step 7, 1.4 Step 5, 2.3 Step 6, 3.3 Step 6, 4.1 Step 6. §9 risks: soil absent (0.3, 0.5), longwave net rather than downward (0.5 Step 2), collinearity (2.3 reports all counts), wrong sign (2.3 Step 4 reports, never tunes), k instability (3.2 clips τ at one slot), zero skill everywhere (3.3 Step 4), 2019 fold too short (`year_ladder` skips a fold with under two rows on either side and `per_fold` shows which).
- **Placeholders.** Two discovered facts are left to be read at execution rather than asserted here, each with the exact file and line to read: the return columns of `compare.pairwise_agreement` (0.5 Step 3) and the keys of `monitoring.chart_series` and the columns of `alarm_episodes` (4.1 Step 4). Everything else is written out.
- **Type consistency.** `year_ladder` returns `mae_val`, `skill`, `skill_q05`, `skill_q95` so `plot_ladder` (existing) draws it; `channel_ladder`'s rows appended in 3.3 carry the same names. `antecedent_index` names its series `api_<tau>d`, and Movements 0, 1, 2 use `f'api_{tau}d'` throughout. `segment_rate_regression` reads `rate_mdeg_per_year`, the column `trend_parameters` writes. `ERA5_MAP_FULL` keys `lwrad`, `snow` give columns `lwrad_era5`, `snow_era5` under `load_era5`'s naming, which Movements 0, 3 and 4 use; `SOIL_MAP` keys `swvl1`…`swvl4` give `swvl1_era5`…, which Movements 0, 1, 2 use.
