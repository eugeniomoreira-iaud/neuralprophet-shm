# Study 04 · Decomposition, Expectation and Anomaly — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` — chosen for this plan on 2026-08-25. One fresh subagent per task, with a review between tasks. Steps use checkbox (`- [ ]`) syntax for tracking; tick them as they are completed so the ledger in section 0 stays true.

**Goal:** Rebuild `studies/04_neuralprophet_inclination_prediction/` so it decomposes the compensated, cleaned inclination record, states whether a newly arrived 20-minute reading is the expected one, and reports the horizon beyond which forecasting stops beating a trivial baseline.

**Architecture:** Two NeuralProphet models fitted on gap-safe contiguous segments with imputation disabled. Model A decomposes the *level* at the native 20-minute cadence with `n_lags=0` and contemporaneous `tair`/`rh` regressors, yielding trend, daily seasonality, regressor contributions and a residual that feeds EWMA/CUSUM control charts. Model B forecasts the gap-safe *hourly change* with autoregression and past-only lagged regressors, scored against three baselines with a paired block bootstrap. All code lives in `studies/shmlib/`; the notebook orchestrates and owns every path and parameter.

**Tech Stack:** Python 3.10, `neuralprophet==0.8.0`, pandas, numpy, statsmodels, matplotlib/seaborn, jupytext (`py:percent`), `unittest` (no pytest), LaTeX via `pdflatex`.

**Spec:** `docs/superpowers/specs/2026-08-25-study04-decomposition-and-anomaly-design.md`

## 0 · Status, and how to resume

**Execution mode: subagent-driven.** Decided 2026-08-25. One fresh subagent per task, dispatched
under `superpowers:subagent-driven-development`; the orchestrator reviews between tasks and stops
at every checkpoint for the user. Tasks marked *orchestrator* in the Notes are not dispatched.

**Tasks 1 to 12 are complete.** Phases 0, 1, 2 and 3 are done, so every library function the study
needs now exists. Work is on branch `study04-rebuild`, cut from `main` at `9ebcc5c`, and the last
commit of Phase 3 is `8c5f783`. Checkpoints 0 and 1–2 were shown to the user and approved on
2026-08-25, and Checkpoint 3 on 2026-08-26.

**The next action is Task 12A**, which gives the library the three physically motivated
perturbations D12 requires, after which Phase 4 begins at Task 13.

### To resume in a new session

Say, or paste:

> Continue the study 04 rebuild. Read
> `docs/superpowers/plans/2026-08-25-study04-decomposition-and-anomaly.md`, find the first task
> whose steps are unticked in the ledger below, and execute it with
> `superpowers:subagent-driven-development`. Stop at the next checkpoint.

The plan argues from `docs/superpowers/specs/2026-08-25-study04-decomposition-and-anomaly-design.md`,
which carries the reasoning behind every decision here. Read both. Nothing in the plan needs the
conversation that produced it.

**Check out the branch first:** `git checkout study04-rebuild`. It has not been merged to `main`.

**The environment's `python` is the wrong one.** The interpreter on `PATH` has no `neuralprophet`,
which makes two tests in `test_prediction.py` fail spuriously. Every command in this plan that says
`python` must be run as:

```
/Users/eugenio/anaconda3/envs/neuralprophet_env/bin/python
```

`conda activate` does not survive a non-interactive shell, so use the absolute path. Warnings about
`pkg_resources` being deprecated and `Importing plotly failed` are normal noise in this environment.

**Current test state**, all 229 passing, measured 2026-08-26 from `studies/`:

| File | Tests |
|---|---|
| `04_neuralprophet_inclination_prediction/tests/test_gaps.py` | 16 |
| `04_neuralprophet_inclination_prediction/tests/test_decomposition.py` | 22 |
| `04_neuralprophet_inclination_prediction/tests/test_prediction.py` | 20 |
| `04_neuralprophet_inclination_prediction/tests/test_folder_honesty.py` | 4 |
| `shmlib/tests/test_shmlib.py` | 60 |
| `shmlib/tests/test_monitoring.py` | 28 |
| `03_thermomechanical_response/tests/test_shmlib_study03.py` | 44 |
| `02_proxy_forcing_characterization/tests/test_shmlib_study02.py` | 35 |

**Every count above is higher than this plan predicts, and none may be "corrected" downwards.**
Each excess test is a guard added when a review found a defect: `test_gaps.py` holds 16 rather than
14 (Tasks 4 and 5), `test_monitoring.py` 28 rather than 20 (Rulings P24 and P25), and
`test_decomposition.py` 22 rather than the 19 Task 12 predicts as its result (Rulings P21 and P28).

### Ledger

Tick a task only when its final commit exists. A task that stopped mid-way is *not started* for
the purposes of resuming — its steps are ordered so that re-running from Step 1 is safe.

| # | Task | Phase | Runner | Done |
|---|---|---|---|---|
| 1 | Remove fabricated claims and throwaway scripts | 0 | orchestrator | [x] |
| 2 | `prediction.gap_inventory` | 1 | subagent | [x] |
| 3 | `prediction.segment_survival` | 1 | subagent | [x] |
| 4 | `prediction.cadence_evidence` | 2 | subagent | [x] |
| 5 | Phase 1 and 2 figures | 2 | subagent | [x] |
| 6 | Notebook steps 1–3 | 1–2 | orchestrator | [x] |
| 7 | Changepoints and the decomposition path | 3 | subagent | [x] |
| 8 | Component shares and residual diagnostics | 3 | subagent | [x] |
| 9 | Extend `score_predictions` | 3 | subagent | [x] |
| 10 | `shmlib.monitoring` — charts and episodes | 3 | subagent | [x] |
| 11 | Anomaly injection and detectability | 3 | subagent | [x] |
| 12 | The remaining figures | 3 | subagent | [x] |
| 12A | Perturbations with a physical mechanism | 3 | subagent | [ ] |
| 13 | Fit Model A, confront Study 03 | 4 | orchestrator | [ ] |
| 14 | The expectation and its calibration | 4 | orchestrator | [ ] |
| 15 | Charts tuned to a false-alarm budget | 5 | orchestrator | [ ] |
| 16 | Detectability and the known event | 5 | orchestrator | [ ] |
| 17 | The ablation ladder | 6 | subagent | [ ] |
| 18 | The gap-closure verdict | 7 | subagent | [ ] |
| 19 | Run metadata and table bodies | 8 | orchestrator | [ ] |
| 20 | Write the report | 8 | **orchestrator only** | [ ] |

Task 12A carries no checkpoint of its own: it is library work inside Phase 3, whose Checkpoint 3 has
already been shown, and its acceptance gate is its own test suite plus the standing suites it must
leave untouched.

Checkpoints, each shown to the user before the next task starts: **0** after Task 1, **1–2** after
Task 6, **3** after Task 12, **4** after Task 13, **5** after Task 16, **6** after Task 17,
**7** after Task 18, **8** after Task 20.

### Where the repository stands

- `studies/` is under version control as of `555da8c`; the previous run's notebook and report were
  destroyed before it existed and are not recoverable. This plan and its spec are committed.
- `.gitignore` excludes the local `.adc` cache, study `outputs/`, the graph build artefacts,
  `.DS_Store` and `.vscode/`. Only `graph.json`, `GRAPH_REPORT.md` and `manifest.json` are kept
  from `studies/graphify-out/`.
- **Study `outputs/` is gitignored, so the artefacts of Task 6 are on local disk only.** A clone
  elsewhere must re-run the notebook to regenerate `NP_01`–`NP_04` and `NP_F01`–`NP_F04`. The
  archive it reads, `data/interim/archive/gubbio_archive_20min.csv` (45 MB), is likewise not in
  the repository.
- `.claude/settings.local.json` carries uncommitted permission entries accumulated during
  execution. Left uncommitted deliberately — harness configuration, not study work.
- One open item deliberately left for the user, blocking nothing: whether the ~26 MB of
  retired-study PDFs and executed notebooks under `studies/obsolete/` stay in the repository.

### Commits so far, in order

| Commit | Task | What |
|---|---|---|
| `5ac1df9` | 1 | Removed the fabricated report paragraph, the four throwaway scripts, and restated the folder's real status; added `test_folder_honesty.py`. Report PDF rebuilt from the corrected source |
| `7413880` | — | Plan ledger tick |
| `80b03eb` | 2 | `prediction.gap_inventory` and `DEFAULT_GAP_CLASSES` |
| `cdecba2` | 2 | Docstring: `gap_inventory`'s fallback for a duration matching no bin |
| `56d1655` | 3 | `prediction.segment_survival` |
| `0434706` | 4 | `prediction.cadence_evidence` |
| `dc32058` | 4 | Fix: `cadence_evidence` reindexes onto a complete grid before the level autocorrelation |
| `e4dd4c7` | 5 | `figures.plot_gap_anatomy`, `plot_segment_survival`, `plot_cadence_evidence` |
| `f9d7ffc` | 5 | Fix: line styles cycle across forecast-horizon groups |
| `e4f35e0` | 6 | Notebook steps 1–3; the record, the gap anatomy, the cadence evidence |
| `95edd35` | — | Plan ledger tick |
| `de073bd` | — | Execution state recorded so the rebuild could resume elsewhere |
| `ab9b6b4` | 7 | `covered_changepoints`, `decompose_components`; five additive arguments on `neuralprophet_backtest`, one on `neuralprophet_predict` |
| `1814125` | 8 | `component_variance_shares`, `residual_diagnostics` |
| `d986d73` | 8 | Fix: an untestable Ljung–Box lag is reported as `NaN` instead of raising |
| `f00904a` | 9 | `score_predictions` gains `mase`, `pinball_q05`, `pinball_q95`, `interval_score` |
| `1fdc5a0` | 10 | `shmlib/monitoring.py`: reference statistics, EWMA, CUSUM, joint alarm, episodes, run length |
| `f7789a5` | 10 | Fix: raise on a degenerate `sigma`; enforce the regular-grid precondition |
| `0faf499` | 11 | `inject_anomaly`, `detectability_curve`, with an attributable detection rule |
| `4f45e35` | 11 | Fix: a docstring that described behaviour the code does not have |
| `eff8416` | 12 | `plot_decomposition_stack`, `plot_prediction_band`, `plot_control_chart`, `plot_metric_vs_horizon`, `plot_detectability` |
| `8c5f783` | 12 | Fix: the accent colour stops encoding a data category in two figures |

### Rulings taken during execution, which later tasks must honour

These were decided by the orchestrator while executing, each against the plan text that prompted
them, with the design document as the binding authority. They are recorded here because the
execution ledger they were first written to lives under `.superpowers/`, which is gitignored and
does not survive to another clone.

1. **`monitoring` is not yet importable.** Task 6's Step 1 adds `monitoring` to the notebook's
   imports cell, but `studies/shmlib/monitoring.py` is *created* by Task 10. It was therefore
   omitted, and the notebook currently imports
   `from shmlib import adc, figures, prediction, proxies, site, tables, viz`. **Task 10 or Task 15
   must add `monitoring` to that line**, or the control-chart step will fail on a missing name.
2. **`NP_F01` keeps its five panels, and both channel maps stay whole.** Task 6's replacement
   parameter cell would have dropped `sr` and `twall` from `STR_MAP_CURRENT` and cut
   `ON_STRUCTURE_COLUMNS` to three entries, which would have made the notebook overwrite the
   standing five-panel record figure with a three-panel one. `NP_F01` is not in Task 6's Produces
   list, and section 10.3 of the design document states that the standing introduction and `NP_F01`
   are sound and reused. The maps and the five-panel tuple were kept. `PREDICTOR_COLUMNS =
   ('tair', 'rh')` governs every model and the segment-survival requirement, so **D11 is intact**:
   it excludes wall temperature and solar radiation from *models*, not from a figure describing what
   the instrument package records. `NP_01_window_coverage.csv` consequently reports five channels.
3. **The reference table reproduces, with one six-slot difference.** Every quantity a decision rests
   on is exact: 1,647 gaps, level autocorrelation 0.99771, change autocorrelation −0.0581 at 20 min
   and +0.3494 hourly, `corr(inc, tair)` −0.8647, 72.80 % of missing time inside the four outages
   over seven days, median segment 1.67 h, and 101 segments giving 15,000 windows at a 24 h lag with
   a 24 h horizon. Two entries differ: **62,931 accepted inclination slots against the table's
   62,937, and 6,791 missing hours against 6,789** — one fact, six slots this pipeline masks that
   the design-time script accepted, 0.0095 % of accepted values. The window is exactly the table's
   83,305 slots and the gap count is identical, so those six extended existing gaps rather than
   creating new ones. **Where the report quotes a number it quotes `NP_01`–`NP_04`, not section 2.4
   of the design document**, since the notebook's outputs come from the study's own versioned
   library. Coverage therefore reads 75.5 %, not 75.6 %.
4. **Task 3's cross-reference is misnumbered.** Its Interfaces section says "Task 18 reads
   `n_windows` to choose `n_lags`". The task that chooses `n_lags` for the forecasting model is
   **Task 17**; Task 18 is the gap-closure verdict.
5. **`python .../tests/*.py` does not do what it looks like.** A shell glob handed to `python` runs
   only the first file and passes the rest as `argv`. Task 5's Step 5 and any other step using that
   form must be run as a loop:
   `for t in <dir>/tests/test_*.py; do <interpreter> "$t"; done`.
6. **Three header/interface errors in later tasks were checked against their step code and are
   prose-only — the code is correct.** Task 13's Produces omits `predictions_a`, which its step code
   does bind (`model_a, predictions_a = fits[MODEL_A_YEARLY]`). Task 13 and Task 14 both list
   `NP_F06_daily_cycle_and_response` as an output, but only Task 14's step code writes it — Task 13
   writes `NP_F05_decomposition_stack`, and **Task 14 owns `NP_F06`**. Task 17's Produces omits
   `segmented_b` and `hourly`, which its step code does create. No code change is needed for any of
   the three.
7. **Two known-weak spots left as they are**, both recorded for the final review rather than fixed:
   `segment_survival` truncates `lag_hours`/`forecast_hours` with `int()` while computing `need`
   from the float, so a fractional hour would be labelled differently from how it was used — latent,
   as the documented contract is int-only; and `gap_inventory`'s classification falls back to the
   last class's label when a duration matches no bin, which is documented rather than prevented.

### Rulings taken during Phase 3

Same status as the list above: decided by the orchestrator while executing, against the plan text
that prompted them, with the design document as the binding authority. Six of the nine were taken
because a mandated test or a mandated line was wrong, and in every case the measurement that settled
it is quoted, because the alternative — trusting the plan over the evidence — is what these rulings
exist to prevent.

8. **The plan's line numbers for Phase 3 are stale by about +221 lines.** The preflight claimed
   Tasks 2 to 4 append at end of file; they append after `contiguous_segments`, so everything below
   moved. Every Phase 3 dispatch located its target by function name instead. Current positions:
   `_score_group` 434, `score_predictions` 470, `_model_frame` 592, `_analysis_freq` 606,
   `neuralprophet_backtest` 675, `neuralprophet_predict` 775.
9. **`residual_diagnostics` reports an untestable lag rather than raising.** `acorr_ljungbox` raises
   when a lag is not shorter than the surviving series. The guard is a no-op wherever the old code
   returned, and keeps one row per requested lag.
10. **`_standardise` raises on a degenerate `sigma`.** It used to return an all-NaN `z`, and
    `ewma_chart` then filled its alarm column with `False` — a broken reference window read as a
    quiet structure, which is the worst failure a monitoring system has.
11. **`alarm_episodes` and `average_run_length` enforce the regular grid they document.** Both
    derived durations from one spacing without checking it, so an alarm series with rows dropped
    across an outage would report an episode spanning time the record does not cover.
12. **The CUSUM test asserted luck, not behaviour.** `assertFalse(alarm.iloc[:900].any())` on pure
    noise holds for 5 of 50 seeds — a two-sided CUSUM with `k = 0.5`, `h = 5` has an in-control
    average run length of a few hundred samples. Replaced with a rate bound and a pre/post contrast.
13. **`detectability_curve` counted alarms it had not caused.** It searched to the end of the
    record, so a **0.01 mdeg** pulse was reported detected with a **114.7 h** delay off a false alarm
    on the clean series — an error inflating the study's headline sensitivity. Detection is now
    attributable, counting only slots the uncontaminated run leaves silent, and bounded by a
    documented `response_window='24h'` argument that the notebook passes explicitly.
14. **`assertGreater(delay_h, 0.0)` demanded the detector be late.** A 5-sigma step alarms on its
    first contaminated sample. The "detected sooner" claim moved to its own test, where it is
    verifiable: 1.0 detected at 1.67 h against 2.0 at 0.67 h.
15. **The legend tests used an accessor that cannot answer the question.**
    `legend.get_bbox_to_anchor().y1` is a display-pixel position, positive for any legend on the
    canvas, so `assertLess(..., 0.0)` can never pass for a legend that exists. Both test files now
    compare `legend.get_window_extent(renderer).y1` against `ax.get_window_extent().y0`. This also
    repaired the dead guard Ruling 7 above recorded in `test_gaps.py`, which is now pointed at a
    figure that draws a legend.
16. **The accent colour stopped encoding data.** `plot_decomposition_stack` painted the residual
    panel Vermilion and `plot_prediction_band` painted the expected trajectory Vermilion; the
    convention reserves it for annotations and event markers. Both take the inclination identity
    colour, and the expected line is distinguished by its dash. The control-limit lines stay
    Vermilion deliberately — a control limit is a reference line, which is the role the accent is for.

### Decisions already taken, not to be reopened

Wall temperature and solar radiation are out of scope. The study is rebuilt in place as study 04,
keeping the `NP_` prefix. The report is written in English. `studies/` is version-controlled.
These were settled on 2026-08-25 and are recorded in section 10 of the design document.

---

## Global Constraints

- **Caveman ultra in chat; full prose in code, docstrings, comments, commit messages, and every word of the report.**
- **Every function lives in `studies/shmlib/`.** No code in the study folder. No `*_lib.py`. A function that encodes a choice takes that choice as a documented argument.
- **The notebook's parameter cell holds every path and every governing parameter.** Re-aiming the study at another station or window must require editing that cell alone.
- **Never edit `.ipynb` directly.** Edit `neuralprophet_inclination_prediction_study.py`; the pair is synced by `jupytext --sync` or `auto_watcher.py`.
- **Tests are `unittest`,** run as `python studies/shmlib/tests/test_shmlib.py` and `python 04_neuralprophet_inclination_prediction/tests/test_prediction.py` from `studies/`. Do not introduce pytest.
- **Library adaptations are additive.** Existing parameters keep their names, defaults and behaviour; existing output columns keep their names, order and values.
- **Artefact naming:** `NP_NN_<description>.csv` for tables, `NP_FNN_<description>` for figures (PNG **and** SVG, both written by `viz.finish`).
- **Figure rules** (binding, `instructions-pipeline.md`): Okabe–Ito for categories; Cividis for scalars; fixed channel identity colours — inclination `#0072B2`, air temperature `#E69F00`, relative humidity `#009E73`, wall temperature `#DAA520`, solar radiation `#CC79A7`; Vermilion `#D55E00` reserved for annotations and event markers; span highlights black at `alpha=0.05` with no edge; **legends below the axes, never inside**, via `ax.legend(fontsize='small', ncol=N, loc='upper center', bbox_to_anchor=(0.5, -0.30), frameon=False)`; no outlines or halos on data lines.
- **The raw `.adc` archive is read-only and never committed.** This study reads only `data/interim/archive/gubbio_archive_20min.csv`.
- **Excluded from this study** (decision D11): wall temperature `twall` and solar radiation `sr`.
- **Window:** `SEGMENT_START = '2023-06-21'` to the end of the archive.
- **Response:** `inc_comp_cleaned`, read with `honour_spike=True` so interpolated values are returned missing.
- **Instrument eras are out of scope entirely** (D9). Study 01's cleaned, compensated series is this
  study's raw data. No era term, no era column read, no era label passed to any function —
  `cadence_evidence` and `hourly_change` are called with `era=None`.
- **Compensation is taken as given** (D10). The compensated channel is modelled as delivered. No
  raw-channel fit, no comparison against `adc.DOCUMENTED_COEFF`, no claim about the sign
  contradiction in `docs/raw-data-format.md` §7.5. Study 03's −2.79 mdeg/°C is the external check.
- **The summer-2026 event is not detection evidence** (D12). It is excluded from the reference
  window as a precaution, and no figure, table or sentence reports whether the detector found it.
  Sensitivity is stated from injected perturbations shaped like damage instead.
- **Commit messages** end with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  ```

## Reference values every task may need

Measured 2026-08-25 over `2023-06-21 → 2026-08-21`, 83,305 slots on the 20-minute grid. Tasks assert against these.

| Quantity | 20-min grid | 1-h grid |
|---|---|---|
| Inclination coverage | 75.6 % (62,937) | 78.3 % (21,749) |
| `tair`, `rh` coverage | 77.8 % | — |
| Gaps | 1,647 | 391 |
| Missing hours | 6,789 | 6,020 |
| Missing time in 4 gaps > 7 d | 72.8 % | 82.1 % |
| Median contiguous segment | 1.7 h | 19 h |
| Segments at 24 h lags + 24 h horizon | 101 → 15,000 windows | 101 → 13,243 windows |
| Level lag-1 autocorrelation | 0.99771 | — |
| Change lag-1 autocorrelation | −0.0581 | +0.3494 |
| Change std / MAD | 2.5717 / 0.78 mdeg | 3.6998 / 1.30 mdeg |
| `corr(Δinc, Δtair)` | −0.7781 | −0.8925 |
| `corr(inc, tair)` | −0.8647 | −0.8665 |

Study 03's independently measured diurnal-band gain, which Model A must reproduce: **−2.79 mdeg/°C** (`r = −0.957`). The documented compensation coefficient is deliberately absent: D10 puts compensation outside this study.

---

## File Structure

**Created**

| File | Responsibility |
|---|---|
| `studies/shmlib/monitoring.py` | Residual control charts, alarm episodes, run lengths, anomaly injection, detectability sweeps |
| `studies/shmlib/tests/test_monitoring.py` | Unit tests for the above |
| `studies/04_neuralprophet_inclination_prediction/tests/test_gaps.py` | Unit tests for the gap-anatomy and cadence functions |
| `studies/04_neuralprophet_inclination_prediction/tests/test_decomposition.py` | Unit tests for the decomposition, changepoint and metric additions |

**Modified**

| File | Change |
|---|---|
| `studies/shmlib/prediction.py` | `gap_inventory`, `segment_survival`, `cadence_evidence`, `covered_changepoints`, `decompose_components`, `component_variance_shares`, `residual_diagnostics`; additive parameters on `neuralprophet_backtest`, `neuralprophet_predict`, `score_predictions` |
| `studies/shmlib/figures.py` | `plot_gap_anatomy`, `plot_segment_survival`, `plot_cadence_evidence`, `plot_decomposition_stack`, `plot_prediction_band`, `plot_control_chart`, `plot_metric_vs_horizon`, `plot_detectability` |
| `studies/04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.py` | Rebuilt: parameter cell plus nine orchestration steps |
| `studies/04_neuralprophet_inclination_prediction/report/neuralprophet_inclination_prediction_report.tex` | Rebuilt to the eleven-section structure of the spec |
| `studies/04_neuralprophet_inclination_prediction/README.md` | Rewritten to describe what the folder actually contains |
| `studies/README.md` | Study 04 row corrected |

**Deleted**

`studies/04_neuralprophet_inclination_prediction/clean_up.py`, `studies/04_neuralprophet_inclination_prediction/replace_report.py`, `replace_panels2.py`, `replace_panels3.py`.

---

## Phase 0 · Truth restoration

### Task 1: Remove fabricated claims and throwaway scripts

**Files:**
- Delete: `studies/04_neuralprophet_inclination_prediction/clean_up.py`
- Delete: `studies/04_neuralprophet_inclination_prediction/replace_report.py`
- Delete: `replace_panels2.py`, `replace_panels3.py`
- Modify: `studies/04_neuralprophet_inclination_prediction/report/neuralprophet_inclination_prediction_report.tex:86`
- Modify: `studies/04_neuralprophet_inclination_prediction/README.md`
- Modify: `studies/README.md:32`
- Modify: `studies/04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.py:44-45`

**Interfaces:**
- Consumes: nothing.
- Produces: a folder in which no claim is unsupported by an artefact on disk. Later tasks assume the report `.tex` ends after the `NP_F01` figure and that `README.md` describes the rebuild in progress.

- [x] **Step 1: Write the failing test**

Create `studies/04_neuralprophet_inclination_prediction/tests/test_folder_honesty.py`:

```python
"""
Guards that the study folder never again claims a result it does not hold.

These are repository hygiene tests rather than numerical ones: they assert that
no code lives outside shmlib, and that no artefact is named in prose without a
file behind it.
"""
import re
import unittest
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
REPORT = STUDY / 'report' / 'neuralprophet_inclination_prediction_report.tex'
README = STUDY / 'README.md'


class TestNoCodeOutsideShmlib(unittest.TestCase):

    def test_only_the_notebook_source_is_a_python_file_in_the_study_root(self):
        found = sorted(p.name for p in STUDY.glob('*.py'))
        self.assertEqual(
            found, ['neuralprophet_inclination_prediction_study.py'],
            'A study folder holds its notebook and nothing else executable; '
            'every function belongs in studies/shmlib/.')


class TestReportClaimsAreSupported(unittest.TestCase):

    def test_report_does_not_assert_unproduced_results(self):
        text = REPORT.read_text(encoding='utf-8')
        for phrase in ('definitive 48-hour limit', 'pinball quantiles',
                       'horizon-adaptive block-bootstrapping'):
            self.assertNotIn(
                phrase, text,
                f'The report asserts {phrase!r} with no artefact behind it.')

    def test_every_included_graphic_exists(self):
        text = REPORT.read_text(encoding='utf-8')
        outputs = STUDY / 'outputs'
        for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text):
            stem = Path(name).stem
            self.assertTrue(
                list(outputs.glob(stem + '.*')),
                f'The report includes {name} but outputs/ holds no such file.')


class TestReadmeMatchesTheFolder(unittest.TestCase):

    def test_readme_names_only_artefacts_that_exist(self):
        text = README.read_text(encoding='utf-8')
        outputs = STUDY / 'outputs'
        for name in re.findall(r'`(NP_F?\d\d[A-Za-z0-9_]*)', text):
            self.assertTrue(
                list(outputs.glob(name + '*')),
                f'README names {name}, which outputs/ does not contain.')


if __name__ == '__main__':
    unittest.main(verbosity=2)
```

- [x] **Step 2: Run it to confirm it fails**

Run from `studies/`:
```bash
python 04_neuralprophet_inclination_prediction/tests/test_folder_honesty.py
```
Expected: three failures — `clean_up.py`/`replace_report.py` present, the report asserting `definitive 48-hour limit`, and the README naming `NP_01` … `NP_10` which do not exist.

- [x] **Step 3: Delete the throwaway scripts**

```bash
cd studies/04_neuralprophet_inclination_prediction
git rm clean_up.py replace_report.py
cd ../..
git rm replace_panels2.py replace_panels3.py
```

- [x] **Step 4: Remove the fabricated paragraph from the report**

Delete the whole paragraph at `report/neuralprophet_inclination_prediction_report.tex:86` beginning `This updated study specifically executes this fourth movement`. Nothing replaces it; the introduction ends with the paragraph describing the four movements, and the `NP_F01` figure follows.

- [x] **Step 5: Correct the stale sentence in the notebook**

In `neuralprophet_inclination_prediction_study.py`, replace lines 44–45:

```python
# Movements 2 to 4 are not implemented yet. Everything below the isolation
# marker at the end of this notebook belongs to the previous experiment and is
# held there, inert, until it is rewritten.
```

with:

```python
# Movements 2 to 4 are being rebuilt to the design in
# docs/superpowers/specs/2026-08-25-study04-decomposition-and-anomaly-design.md.
# The previous experiment's cells were removed rather than held inert; its
# reusable logic survives in shmlib.prediction.
```

- [x] **Step 6: Rewrite the study README**

Replace `studies/04_neuralprophet_inclination_prediction/README.md` with:

```markdown
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
```

- [x] **Step 7: Correct the study 04 row of the studies index**

In `studies/README.md:32`, replace the row with:

```markdown
| 4 | [`04_neuralprophet_inclination_prediction/`](04_neuralprophet_inclination_prediction/) | What is the compensated inclination record made of, is a newly arrived reading the one the measured environment predicts, and how far ahead is forecasting worth anything? | In progress |
```

- [x] **Step 8: Run the tests to verify they pass**

Run from `studies/`:
```bash
python 04_neuralprophet_inclination_prediction/tests/test_folder_honesty.py
```
Expected: `OK`, 4 tests.

- [x] **Step 9: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
fix(study04): remove fabricated results and code outside shmlib

The study README, the studies index and one paragraph of the report described a
completed experiment whose outputs do not exist on disk. Two throwaway scripts in
the study folder had generated that report prose and then truncated the file.

Remove the scripts and the unsupported claims, restate the folder's real status,
and add hygiene tests that fail if either recurs.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

> **CHECKPOINT 0 — show the user:** `git show --stat HEAD`, the new README, and passing
> `test_folder_honesty.py`. No claim in the folder is unsupported by an artefact; no code outside
> `shmlib`. Wait for approval before Task 2.

---

## Phase 1 · The record and the anatomy of what is missing

### Task 2: `prediction.gap_inventory`

**Files:**
- Modify: `studies/shmlib/prediction.py` (append after `contiguous_segments`)
- Test: `studies/04_neuralprophet_inclination_prediction/tests/test_gaps.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `gap_inventory(series, freq='20min', classes=DEFAULT_GAP_CLASSES) -> pd.DataFrame` with
  columns `start`, `end`, `duration_h`, `n_slots`, `gap_class`, one row per maximal run of missing
  slots, plus the module constant `DEFAULT_GAP_CLASSES`. Task 4 plots it; Task 5 exports it as
  `NP_02`.

- [x] **Step 1: Write the failing test**

Create `studies/04_neuralprophet_inclination_prediction/tests/test_gaps.py`:

```python
"""
Unit tests for the gap-anatomy and cadence functions this study relies on.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import prediction


def _series(values, start='2024-01-01', freq='20min'):
    index = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=index, dtype=float)


class TestGapInventory(unittest.TestCase):

    def test_each_run_of_missing_slots_becomes_one_row(self):
        values = [1.0, np.nan, np.nan, 2.0, 3.0, np.nan, 4.0]
        table = prediction.gap_inventory(_series(values), freq='20min')
        self.assertEqual(len(table), 2)
        self.assertEqual(list(table['n_slots']), [2, 1])
        self.assertAlmostEqual(table['duration_h'].iloc[0], 2 / 3)
        self.assertEqual(table['start'].iloc[0],
                         pd.Timestamp('2024-01-01 00:20:00'))
        self.assertEqual(table['end'].iloc[0],
                         pd.Timestamp('2024-01-01 00:40:00'))

    def test_a_series_with_no_gaps_returns_an_empty_typed_table(self):
        table = prediction.gap_inventory(_series([1.0, 2.0, 3.0]), freq='20min')
        self.assertTrue(table.empty)
        self.assertEqual(list(table.columns),
                         ['start', 'end', 'duration_h', 'n_slots', 'gap_class'])

    def test_gaps_are_classified_by_duration(self):
        values = [1.0] + [np.nan] * 3 + [1.0] + [np.nan] * 60 + [1.0]
        table = prediction.gap_inventory(_series(values), freq='20min')
        self.assertEqual(list(table['gap_class']), ['<=1h', '6-24h'])

    def test_an_irregular_index_is_reindexed_onto_the_grid_first(self):
        index = pd.DatetimeIndex(['2024-01-01 00:00', '2024-01-01 01:00'])
        table = prediction.gap_inventory(
            pd.Series([1.0, 2.0], index=index), freq='20min')
        self.assertEqual(len(table), 1)
        self.assertEqual(table['n_slots'].iloc[0], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
```

- [x] **Step 2: Run it to verify it fails**

Run from `studies/`:
```bash
python 04_neuralprophet_inclination_prediction/tests/test_gaps.py
```
Expected: `AttributeError: module 'shmlib.prediction' has no attribute 'gap_inventory'`.

- [x] **Step 3: Implement**

Append to `studies/shmlib/prediction.py`:

```python
DEFAULT_GAP_CLASSES = (
    (0.0, 1.0, '<=1h'),
    (1.0, 6.0, '1-6h'),
    (6.0, 24.0, '6-24h'),
    (24.0, 168.0, '1-7d'),
    (168.0, np.inf, '>7d'),
)


def gap_inventory(series, freq='20min', classes=DEFAULT_GAP_CLASSES):
    """
    One row per maximal run of missing slots, classified by duration.

    The series is first reindexed onto a regular grid of spacing ``freq``, so
    that time absent from the index counts as missing rather than disappearing.
    A coverage percentage says how much time is missing; this says how that time
    is shaped, which is what decides whether filling it is interpolation or
    reconstruction.

    Parameters
    ----------
    series : pd.Series
        Numeric signal indexed by timestamp. Missing is ``NaN`` or an absent
        timestamp.
    freq : str, optional
        Spacing of the analysis grid. Default ``'20min'``.
    classes : sequence of (float, float, str), optional
        Half-open duration bins in hours, as ``(low, high, label)``; a gap falls
        in the first bin with ``low < duration_h <= high``. Default
        ``DEFAULT_GAP_CLASSES``.

    Returns
    -------
    pd.DataFrame
        Columns ``start``, ``end``, ``duration_h``, ``n_slots``, ``gap_class``,
        in chronological order. Empty with those columns when nothing is
        missing.
    """
    columns = ['start', 'end', 'duration_h', 'n_slots', 'gap_class']
    values = pd.to_numeric(series, errors='coerce')
    index = pd.DatetimeIndex(values.index)
    if len(index) == 0:
        return pd.DataFrame(columns=columns)

    grid = pd.date_range(index.min(), index.max(), freq=freq)
    values = values.reindex(grid)
    step_hours = pd.Timedelta(freq) / pd.Timedelta(hours=1)

    missing = values.isna().to_numpy()
    if not missing.any():
        return pd.DataFrame(columns=columns)

    positions = np.flatnonzero(missing)
    breaks = np.flatnonzero(np.diff(positions) != 1)
    starts = positions[np.r_[0, breaks + 1]]
    ends = positions[np.r_[breaks, positions.size - 1]]
    n_slots = ends - starts + 1
    duration_h = n_slots * step_hours

    labels = []
    for hours in duration_h:
        label = classes[-1][2]
        for low, high, name in classes:
            if low < hours <= high:
                label = name
                break
        labels.append(label)

    return pd.DataFrame({
        'start': grid[starts],
        'end': grid[ends],
        'duration_h': duration_h,
        'n_slots': n_slots,
        'gap_class': labels,
    }, columns=columns)
```

- [x] **Step 4: Run the tests to verify they pass**

```bash
python 04_neuralprophet_inclination_prediction/tests/test_gaps.py
```
Expected: `OK`, 4 tests.

- [x] **Step 5: Verify nothing else broke**

```bash
python shmlib/tests/test_shmlib.py
python 04_neuralprophet_inclination_prediction/tests/test_prediction.py
python 03_thermomechanical_response/tests/test_shmlib_study03.py
```
Expected: all `OK`.

- [x] **Step 6: Commit**

```bash
git add studies/shmlib/prediction.py studies/04_neuralprophet_inclination_prediction/tests/test_gaps.py
git commit -m "$(cat <<'EOF'
feat(shmlib): inventory missing time as classified gaps

A coverage percentage says how much of the record is absent; it does not say
whether the absence arrives as scattered samples or as seasons, which is what
decides whether filling it is interpolation or reconstruction.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `prediction.segment_survival`

**Files:**
- Modify: `studies/shmlib/prediction.py` (append after `gap_inventory`)
- Test: `studies/04_neuralprophet_inclination_prediction/tests/test_gaps.py`

**Interfaces:**
- Consumes: `contiguous_segments` (`prediction.py:76`).
- Produces: `segment_survival(frame, required, lag_hours, forecast_hours, freq='20min') -> pd.DataFrame`
  with columns `lag_hours`, `forecast_hours`, `n_rows`, `coverage`, `n_segments`, `median_segment_h`,
  `max_segment_h`, `n_surviving`, `n_windows`. One row per `(lag_hours, forecast_hours)` pair.
  Task 4 plots it; Task 5 exports it as `NP_03`; Task 18 reads `n_windows` to choose `n_lags`.

- [x] **Step 1: Write the failing test**

Append to `studies/04_neuralprophet_inclination_prediction/tests/test_gaps.py`, above the
`if __name__` block:

```python
class TestSegmentSurvival(unittest.TestCase):

    def _frame(self):
        # Two complete runs of 12 slots (4 h at 20 min) separated by one gap.
        index = pd.date_range('2024-01-01', periods=25, freq='20min')
        y = np.arange(25, dtype=float)
        y[12] = np.nan
        return pd.DataFrame({'y': y, 'x': np.ones(25)}, index=index)

    def test_counts_windows_that_fit_inside_a_segment(self):
        table = prediction.segment_survival(
            self._frame(), required=['y', 'x'],
            lag_hours=1, forecast_hours=1, freq='20min')
        row = table.iloc[0]
        self.assertEqual(row['n_segments'], 2)
        self.assertEqual(row['n_rows'], 24)
        # Each segment holds 12 slots; a window needs 3 + 3 = 6, so 7 fit.
        self.assertEqual(row['n_surviving'], 2)
        self.assertEqual(row['n_windows'], 14)

    def test_a_window_longer_than_every_segment_survives_nowhere(self):
        table = prediction.segment_survival(
            self._frame(), required=['y', 'x'],
            lag_hours=24, forecast_hours=24, freq='20min')
        self.assertEqual(table['n_surviving'].iloc[0], 0)
        self.assertEqual(table['n_windows'].iloc[0], 0)

    def test_several_configurations_return_several_rows(self):
        table = prediction.segment_survival(
            self._frame(), required=['y'],
            lag_hours=[1, 2], forecast_hours=[1], freq='20min')
        self.assertEqual(len(table), 2)
        self.assertEqual(list(table['lag_hours']), [1, 2])

    def test_a_required_column_that_is_never_present_yields_zero_rows(self):
        frame = self._frame()
        frame['z'] = np.nan
        table = prediction.segment_survival(
            frame, required=['y', 'z'], lag_hours=1, forecast_hours=1,
            freq='20min')
        self.assertEqual(table['n_rows'].iloc[0], 0)
        self.assertEqual(table['n_segments'].iloc[0], 0)
```

- [x] **Step 2: Run it to verify it fails**

```bash
python 04_neuralprophet_inclination_prediction/tests/test_gaps.py
```
Expected: `AttributeError: module 'shmlib.prediction' has no attribute 'segment_survival'`.

- [x] **Step 3: Implement**

Append to `studies/shmlib/prediction.py`:

```python
def segment_survival(frame, required, lag_hours, forecast_hours, freq='20min'):
    """
    How many training windows survive contiguous segmentation, per configuration.

    Segmenting a gapped record is not free: a model that consumes ``lag_hours``
    of history and predicts ``forecast_hours`` ahead can only be trained inside a
    run of complete rows long enough to hold both. This counts what is left, so
    that a lag length is chosen against the record rather than against habit.

    Parameters
    ----------
    frame : pd.DataFrame
        Candidate modelling frame indexed by timestamp on a regular grid.
    required : sequence of str
        Columns that must be present for a row to count as complete.
    lag_hours, forecast_hours : int or sequence of int
        Configurations to evaluate. Scalars are broadcast, and every combination
        of the two is reported.
    freq : str, optional
        Spacing of the analysis grid. Default ``'20min'``.

    Returns
    -------
    pd.DataFrame
        One row per configuration, with ``lag_hours``, ``forecast_hours``,
        ``n_rows``, ``coverage``, ``n_segments``, ``median_segment_h``,
        ``max_segment_h``, ``n_surviving`` and ``n_windows``.
    """
    required = list(required)
    step_hours = pd.Timedelta(freq) / pd.Timedelta(hours=1)
    segmented = contiguous_segments(frame, required, min_length=1, freq=freq)

    if segmented.empty:
        lengths = np.array([], dtype=int)
    else:
        lengths = segmented.groupby('segment_id').size().to_numpy()

    lags = np.atleast_1d(lag_hours)
    horizons = np.atleast_1d(forecast_hours)
    rows = []
    for lag in lags:
        for horizon in horizons:
            need = int(round((float(lag) + float(horizon)) / step_hours))
            usable = lengths[lengths >= need] if lengths.size else lengths
            windows = int((usable - need + 1).sum()) if usable.size else 0
            rows.append({
                'lag_hours': int(lag),
                'forecast_hours': int(horizon),
                'n_rows': int(len(segmented)),
                'coverage': (len(segmented) / len(frame)) if len(frame) else np.nan,
                'n_segments': int(lengths.size),
                'median_segment_h': (float(np.median(lengths) * step_hours)
                                     if lengths.size else np.nan),
                'max_segment_h': (float(lengths.max() * step_hours)
                                  if lengths.size else np.nan),
                'n_surviving': int(usable.size),
                'n_windows': windows,
            })
    return pd.DataFrame(rows)
```

- [x] **Step 4: Run the tests to verify they pass**

```bash
python 04_neuralprophet_inclination_prediction/tests/test_gaps.py
```
Expected: `OK`, 8 tests.

- [x] **Step 5: Verify nothing else broke**

```bash
python shmlib/tests/test_shmlib.py
python 04_neuralprophet_inclination_prediction/tests/test_prediction.py
python 03_thermomechanical_response/tests/test_shmlib_study03.py
```
Expected: all `OK`.

- [x] **Step 6: Commit**

```bash
git add studies/shmlib/prediction.py studies/04_neuralprophet_inclination_prediction/tests/test_gaps.py
git commit -m "$(cat <<'EOF'
feat(shmlib): count training windows surviving segmentation

A lag length must be chosen against the record's contiguity rather than by
habit: on a gapped series most of the cost of a long autoregressive window is
paid in segments that become unusable entirely.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 · Cadence and target, settled on evidence

### Task 4: `prediction.cadence_evidence`

**Files:**
- Modify: `studies/shmlib/prediction.py` (append after `segment_survival`)
- Test: `studies/04_neuralprophet_inclination_prediction/tests/test_gaps.py`

**Interfaces:**
- Consumes: `hourly_change` (`prediction.py:25`).
- Produces: `cadence_evidence(response, driver, era=None, cadences=('20min', '1h'), freq='20min') -> pd.DataFrame`
  with columns `cadence`, `n_level`, `n_change`, `level_autocorr1`, `change_autocorr1`,
  `change_std`, `change_mad`, `corr_level`, `corr_change`, `drift_per_year`. Task 5 plots it;
  Task 6 exports it as `NP_04`.

Design decisions D1, D2 and D3 rest entirely on this table. It exists so that the choice of target
and cadence is a reported measurement rather than a preference.

- [x] **Step 1: Write the failing test**

Append to `studies/04_neuralprophet_inclination_prediction/tests/test_gaps.py`, above the
`if __name__` block:

```python
class TestCadenceEvidence(unittest.TestCase):

    def _pair(self, n=6 * 24 * 30):
        index = pd.date_range('2024-01-01', periods=n, freq='20min')
        phase = np.arange(n) / (3 * 24) * 2 * np.pi
        driver = pd.Series(10.0 * np.sin(phase), index=index)
        response = pd.Series(-2.0 * driver.to_numpy() + 0.01 * np.arange(n),
                             index=index)
        return response, driver

    def test_reports_one_row_per_cadence_with_the_documented_columns(self):
        response, driver = self._pair()
        table = prediction.cadence_evidence(response, driver)
        self.assertEqual(list(table['cadence']), ['20min', '1h'])
        for column in ('level_autocorr1', 'change_autocorr1', 'change_std',
                       'change_mad', 'corr_level', 'corr_change',
                       'drift_per_year'):
            self.assertIn(column, table.columns)

    def test_recovers_the_sign_of_a_known_coupling(self):
        response, driver = self._pair()
        table = prediction.cadence_evidence(response, driver).set_index('cadence')
        self.assertLess(table.loc['1h', 'corr_level'], -0.9)
        self.assertLess(table.loc['1h', 'corr_change'], -0.9)

    def test_white_noise_on_the_level_shows_as_negative_change_autocorrelation(self):
        response, driver = self._pair()
        rng = np.random.default_rng(0)
        noisy = response + rng.normal(0.0, 20.0, len(response))
        table = prediction.cadence_evidence(noisy, driver).set_index('cadence')
        self.assertLess(table.loc['20min', 'change_autocorr1'], 0.0)

    def test_an_era_boundary_is_never_differenced_across(self):
        response, driver = self._pair(n=200)
        era = pd.Series('legacy', index=response.index)
        era.iloc[100:] = 'current'
        table = prediction.cadence_evidence(response, driver, era=era)
        without = prediction.cadence_evidence(response, driver)
        self.assertLess(table['n_change'].iloc[0], without['n_change'].iloc[0])
```

- [x] **Step 2: Run it to verify it fails**

```bash
python 04_neuralprophet_inclination_prediction/tests/test_gaps.py
```
Expected: `AttributeError: module 'shmlib.prediction' has no attribute 'cadence_evidence'`.

- [x] **Step 3: Implement**

Append to `studies/shmlib/prediction.py`:

```python
def cadence_evidence(response, driver, era=None, cadences=('20min', '1h'),
                     freq='20min'):
    """
    Statistics that decide the modelling cadence and the prediction target.

    Reported per candidate cadence: the persistence of the level, the memory
    left in its gap-safe first difference, the scale of that difference, its
    coupling to a driver, and the drift implied by its mean. A level whose
    lag-one autocorrelation is near unity cannot be scored honestly, and a
    difference whose lag-one autocorrelation is negative is dominated by
    measurement noise rather than by the increment it is meant to carry.

    Parameters
    ----------
    response : pd.Series
        Structural response, on the finest available grid.
    driver : pd.Series
        Environmental driver to correlate against, same index.
    era : pd.Series or None, optional
        Instrument-era label per timestamp; a difference crossing a change of
        label is discarded. Default ``None``.
    cadences : sequence of str, optional
        Grids to evaluate. The first must be the native one. Default
        ``('20min', '1h')``.
    freq : str, optional
        Native spacing of the inputs. Default ``'20min'``.

    Returns
    -------
    pd.DataFrame
        One row per cadence, with ``cadence``, ``n_level``, ``n_change``,
        ``level_autocorr1``, ``change_autocorr1``, ``change_std``,
        ``change_mad``, ``corr_level``, ``corr_change`` and ``drift_per_year``.
    """
    response = pd.to_numeric(response, errors='coerce')
    driver = pd.to_numeric(driver, errors='coerce').reindex(response.index)
    era_values = _as_series(era, response.index, name='era')

    rows = []
    for cadence in cadences:
        if cadence == freq:
            level, force = response, driver
            labels = era_values
        else:
            level = response.resample(cadence).mean()
            force = driver.resample(cadence).mean()
            labels = (era_values.resample(cadence).first()
                      if era_values is not None else None)

        change = hourly_change(level, era=labels, freq=cadence)
        driver_change = hourly_change(force, era=labels, freq=cadence)
        steps_per_year = pd.Timedelta(days=365) / pd.Timedelta(cadence)

        level_pair = pd.concat([level, force], axis=1).dropna()
        change_pair = pd.concat([change, driver_change], axis=1).dropna()

        rows.append({
            'cadence': cadence,
            'n_level': int(level.notna().sum()),
            'n_change': int(change.notna().sum()),
            'level_autocorr1': float(level.autocorr(1)),
            'change_autocorr1': float(change.autocorr(1)),
            'change_std': float(change.std()),
            'change_mad': float((change - change.median()).abs().median()),
            'corr_level': (float(level_pair.iloc[:, 0].corr(level_pair.iloc[:, 1]))
                           if len(level_pair) > 1 else np.nan),
            'corr_change': (float(change_pair.iloc[:, 0].corr(change_pair.iloc[:, 1]))
                            if len(change_pair) > 1 else np.nan),
            'drift_per_year': float(change.mean() * steps_per_year),
        })
    return pd.DataFrame(rows)
```

- [x] **Step 4: Run the tests to verify they pass**

```bash
python 04_neuralprophet_inclination_prediction/tests/test_gaps.py
```
Expected: `OK`, 12 tests.

- [x] **Step 5: Verify nothing else broke**

```bash
python shmlib/tests/test_shmlib.py
python 04_neuralprophet_inclination_prediction/tests/test_prediction.py
python 03_thermomechanical_response/tests/test_shmlib_study03.py
```
Expected: all `OK`.

- [x] **Step 6: Commit**

```bash
git add studies/shmlib/prediction.py studies/04_neuralprophet_inclination_prediction/tests/test_gaps.py
git commit -m "$(cat <<'EOF'
feat(shmlib): measure the evidence that fixes cadence and target

Whether a record is modelled on its level or its change, and at which spacing,
is a decision that can be measured: a level near unit autocorrelation cannot be
scored, and a difference with negative lag-one autocorrelation is dominated by
measurement noise.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Phase 1 and 2 figures

**Files:**
- Modify: `studies/shmlib/figures.py` (append at end of module)
- Test: `studies/04_neuralprophet_inclination_prediction/tests/test_gaps.py`

**Interfaces:**
- Consumes: `gap_inventory`, `segment_survival`, `cadence_evidence` outputs; `viz.finish`,
  `viz.format_spines`, `viz.figsize`, `viz.INC_COLOUR`, `viz.MARK_COLOUR`, `viz.FIGURE_WIDTH`.
- Produces:
  - `plot_gap_anatomy(inventory, classes=None, title='', save_path=None, filename=None) -> Figure`
  - `plot_segment_survival(survival, title='', save_path=None, filename=None) -> Figure`
  - `plot_cadence_evidence(evidence, title='', save_path=None, filename=None) -> Figure`

  Task 6 calls all three.

- [x] **Step 1: Write the failing test**

Append to `studies/04_neuralprophet_inclination_prediction/tests/test_gaps.py`, above the
`if __name__` block:

```python
class TestPhaseOneFigures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')

    def test_each_figure_builds_and_saves_both_formats(self):
        import tempfile
        from pathlib import Path

        import matplotlib.pyplot as plt

        from shmlib import figures

        index = pd.date_range('2024-01-01', periods=300, freq='20min')
        values = np.sin(np.arange(300) / 10.0)
        values[50:70] = np.nan
        series = pd.Series(values, index=index)
        frame = pd.DataFrame({'y': series, 'x': 1.0}, index=index)

        inventory = prediction.gap_inventory(series, freq='20min')
        survival = prediction.segment_survival(
            frame, ['y', 'x'], lag_hours=[1, 2], forecast_hours=[1],
            freq='20min')
        evidence = prediction.cadence_evidence(
            series.ffill(), pd.Series(np.arange(300.0), index=index))

        with tempfile.TemporaryDirectory() as tmp:
            for name, call in (
                    ('NP_F02_gap_anatomy',
                     lambda p, f: figures.plot_gap_anatomy(
                         inventory, title='t', save_path=p, filename=f)),
                    ('NP_F03_segment_survival',
                     lambda p, f: figures.plot_segment_survival(
                         survival, title='t', save_path=p, filename=f)),
                    ('NP_F04_cadence_evidence',
                     lambda p, f: figures.plot_cadence_evidence(
                         evidence, title='t', save_path=p, filename=f))):
                call(tmp, name)
                for ext in ('png', 'svg'):
                    self.assertTrue((Path(tmp) / f'{name}.{ext}').exists(),
                                    f'{name}.{ext} was not written')
            plt.close('all')

    def test_no_legend_is_drawn_inside_the_axes(self):
        import matplotlib.pyplot as plt

        from shmlib import figures

        index = pd.date_range('2024-01-01', periods=300, freq='20min')
        values = np.sin(np.arange(300) / 10.0)
        values[50:70] = np.nan
        inventory = prediction.gap_inventory(
            pd.Series(values, index=index), freq='20min')
        fig = figures.plot_gap_anatomy(inventory, title='t')
        for ax in fig.axes:
            legend = ax.get_legend()
            if legend is not None:
                self.assertLess(legend.get_bbox_to_anchor().y1, 0.0,
                                'A legend must sit below its axes.')
        plt.close(fig)
```

- [x] **Step 2: Run it to verify it fails**

```bash
python 04_neuralprophet_inclination_prediction/tests/test_gaps.py
```
Expected: `AttributeError: module 'shmlib.figures' has no attribute 'plot_gap_anatomy'`.

- [x] **Step 3: Implement**

Append to `studies/shmlib/figures.py`:

```python
def plot_gap_anatomy(inventory, classes=None, title='', save_path=None,
                     filename=None):
    """
    How the missing time is shaped: gap count and missing hours, by duration class.

    Two panels answer two different questions about the same table. The left
    counts gaps, which is what governs how badly contiguity is broken; the right
    sums their hours, which is what governs how much record is absent. A record
    can be dominated by one class on the left and another on the right, and the
    difference decides what kind of problem filling it is.

    Parameters
    ----------
    inventory : pd.DataFrame
        Output of ``prediction.gap_inventory``.
    classes : sequence of str or None, optional
        Class order along the category axis. Default: the order in which the
        classes appear in ``prediction.DEFAULT_GAP_CLASSES``.
    title : str, optional
        Figure title. Default ``''``.
    save_path, filename : str or None, optional
        Passed to ``viz.finish``; nothing is written when either is ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from shmlib import prediction as _prediction

    if classes is None:
        classes = [label for _, _, label in _prediction.DEFAULT_GAP_CLASSES]

    counts = (inventory.groupby('gap_class')['n_slots'].size()
              .reindex(classes).fillna(0.0))
    hours = (inventory.groupby('gap_class')['duration_h'].sum()
             .reindex(classes).fillna(0.0))

    fig, axes = plt.subplots(1, 2, figsize=viz.figsize(viz.FIGURE_WIDTH, 2.4))
    for ax, values, label in ((axes[0], counts, 'Number of gaps'),
                              (axes[1], hours, 'Missing time [h]')):
        ax.bar(range(len(classes)), values.to_numpy(), color=viz.INC_COLOUR,
               width=0.72)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_ylabel(label)
        ax.set_xlabel('Gap duration')
        viz.format_spines(ax)

    total = hours.sum()
    if total > 0:
        share = hours / total
        for position, value in enumerate(share.to_numpy()):
            axes[1].annotate(f'{value:.0%}',
                             (position, hours.to_numpy()[position]),
                             ha='center', va='bottom', fontsize='small',
                             color=viz.MARK_COLOUR)

    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_segment_survival(survival, title='', save_path=None, filename=None):
    """
    Training windows surviving segmentation, against the length of window asked for.

    Parameters
    ----------
    survival : pd.DataFrame
        Output of ``prediction.segment_survival``, one row per configuration.
    title : str, optional
        Figure title. Default ``''``.
    save_path, filename : str or None, optional
        Passed to ``viz.finish``; nothing is written when either is ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=viz.figsize(viz.FIGURE_WIDTH, 2.4))
    handles = []
    for position, (horizon, group) in enumerate(
            survival.groupby('forecast_hours')):
        style = '-' if position == 0 else '--'
        line, = ax.plot(group['lag_hours'], group['n_windows'],
                        color=viz.INC_COLOUR, linestyle=style, marker='o',
                        linewidth=1.6, label=f'{int(horizon)} h horizon')
        handles.append(line)
    ax.set_xlabel('Autoregressive window [h]')
    ax.set_ylabel('Training windows')
    viz.format_spines(ax)
    if title:
        ax.set_title(title)
    ax.legend(fontsize='small', ncol=len(handles), loc='upper center',
              bbox_to_anchor=(0.5, -0.30), frameon=False)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_cadence_evidence(evidence, title='', save_path=None, filename=None):
    """
    The three measurements that fix the cadence and the target, side by side.

    Parameters
    ----------
    evidence : pd.DataFrame
        Output of ``prediction.cadence_evidence``.
    title : str, optional
        Figure title. Default ``''``.
    save_path, filename : str or None, optional
        Passed to ``viz.finish``; nothing is written when either is ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    panels = (('level_autocorr1', 'Level lag-1\nautocorrelation'),
              ('change_autocorr1', 'Change lag-1\nautocorrelation'),
              ('corr_change', 'corr(change, driver change)'))
    fig, axes = plt.subplots(1, len(panels),
                             figsize=viz.figsize(viz.FIGURE_WIDTH, 2.2))
    positions = range(len(evidence))
    for ax, (column, label) in zip(axes, panels):
        ax.bar(positions, evidence[column].to_numpy(), color=viz.INC_COLOUR,
               width=0.6)
        ax.axhline(0.0, color='black', linewidth=0.8)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(evidence['cadence'])
        ax.set_ylabel(label)
        viz.format_spines(ax)
    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig
```

- [x] **Step 4: Run the tests to verify they pass**

```bash
python 04_neuralprophet_inclination_prediction/tests/test_gaps.py
```
Expected: `OK`, 14 tests.

- [x] **Step 5: Verify nothing else broke**

```bash
python shmlib/tests/test_shmlib.py
python 03_thermomechanical_response/tests/test_shmlib_study03.py
python 02_proxy_forcing_characterization/tests/*.py
```
Expected: all `OK`.

- [x] **Step 6: Commit**

```bash
git add studies/shmlib/figures.py studies/04_neuralprophet_inclination_prediction/tests/test_gaps.py
git commit -m "$(cat <<'EOF'
feat(shmlib): figures for gap anatomy, segment survival and cadence evidence

Each figure draws one table produced by the corresponding prediction helper, so
that the decisions those tables settle can be read rather than asserted.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Notebook steps 1–3 — the record, its gaps, and the cadence decision

**Files:**
- Modify: `studies/04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.py`
- Produces: `outputs/NP_01_window_coverage.csv`, `NP_02_gap_inventory.csv`,
  `NP_03_segment_survival.csv`, `NP_04_cadence_evidence.csv`,
  `NP_F02_gap_anatomy.{png,svg}`, `NP_F03_segment_survival.{png,svg}`,
  `NP_F04_cadence_evidence.{png,svg}`

**Interfaces:**
- Consumes: `proxies.load_response`, `proxies.load_sensor_forcings`, `proxies.join_eras`,
  `proxies.harmonise`, `prediction.gap_inventory`, `prediction.segment_survival`,
  `prediction.cadence_evidence`, `figures.plot_channel_panels`, `figures.plot_gap_anatomy`,
  `figures.plot_segment_survival`, `figures.plot_cadence_evidence`, `tables.write_table`.
- Produces: the notebook variables `window` (20-minute frame with `inc`, `tair`, `rh`, `era`),
  `SEGMENT_START`, `ANALYSIS_FREQ`, `MODEL_FREQ_A`, `MODEL_FREQ_B`, consumed by Tasks 14 onward.

The existing loaders default to `site.ANALYSIS_FREQ = '1h'`. This study needs the native
20-minute grid for Model A, so every loader call passes `freq=NATIVE_FREQ` explicitly. That is a
parameter the notebook owns, not a library default to change.

- [x] **Step 1: Extend the imports cell**

Replace line 69 of the notebook source:

```python
from shmlib import figures, prediction, proxies, site, viz
```

with:

```python
from shmlib import adc, figures, monitoring, prediction, proxies, site, tables, viz
```

`adc` supplies the documented compensation coefficient quoted in step 5, `monitoring` the control
charts of step 7, and `tables` the LaTeX bodies of step 10. Importing them here keeps the imports
cell the single place a reader learns what the study depends on.

- [x] **Step 2: Replace the parameter cell**

Replace the cell at lines 77–123 of the notebook source with:

```python
# %%
# ---------------------------------------------------------------------------
# Data pointers
# ---------------------------------------------------------------------------
ARCHIVE_CSV = '../../data/interim/archive/gubbio_archive_20min.csv'
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------
# The archive is written every twenty minutes and this study reads it at that
# spacing, because that is the rate at which a deployed system receives a new
# reading and therefore the rate at which it must judge one. Model A works on
# this grid. Model B works hourly: measured on this record, the twenty-minute
# first difference has a lag-one autocorrelation of -0.058, the signature of
# measurement noise dominating the increment, while the hourly difference
# retains +0.349 of memory and couples to air temperature at -0.893 rather
# than -0.778. Notebook step 3 re-measures both and exports the comparison.
NATIVE_FREQ = '20min'
MODEL_FREQ_A = '20min'          # decomposition and expectation
MODEL_FREQ_B = '1h'             # forecast skill

# ---------------------------------------------------------------------------
# The response
# ---------------------------------------------------------------------------
# Study 01's final inclination product: compensated, anchored once across the
# whole record, and cleaned of impulsive noise. Read with honour_spike=True, so
# every value the cleaning interpolated is returned missing rather than as a
# measurement. No earlier or intermediate version of this channel is used.
TARGET_COLUMN = 'inc_comp_cleaned'

# No raw channel is read. Compensation is Study 01's discussion and this study
# is blind to it: the compensated, cleaned channel above is the raw material
# here, and the sign contradiction of docs/raw-data-format.md section 7.5 stays
# Study 01's open question (spec section 11.2).

# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------
# The archive carries a 271-day outage beginning 2022-09-23 whose last missing
# day is 2023-06-20. This study starts on the day after it and runs to the end
# of the archive: the stretch closest to the present, and the one a system
# deployed today would have to work with.
SEGMENT_START = '2023-06-21'

# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------
# Air temperature and relative humidity only. Measured on this record, their
# missingness is perfectly nested inside the inclination's, so they cost no
# coverage at all: requiring them leaves the same 62,937 complete rows in the
# same 1,647 segments. Wall temperature and solar radiation are excluded by
# design decision D11 - requiring them collapses coverage from 75.6% to 17.0%.
PREDICTOR_COLUMNS = ('tair', 'rh')

# Battery voltage is carried as the negative control only. Study 03 established
# that it is not a silent channel - it reaches r = -0.673 in the diurnal band -
# so it marks a conservative floor rather than a zero.
CONTROL_COLUMN = 'batt'

# ---------------------------------------------------------------------------
# Channel maps
# ---------------------------------------------------------------------------
# The archive names the package's channels differently before and after the
# instrument installed on 2025-02-21, so each set is loaded under its own map
# and the two are joined. This is a column-naming detail and not an analytical
# split: Study 01 already compensated and anchored the inclination once across
# the whole record, and no era offset is estimated anywhere in this study.
STR_MAP_CURRENT = {'tair': 'tair', 'rh': 'n_rh_ok', 'batt': 'n_batt_ok'}
STR_MAP_LEGACY = {'tair': 'tair',
                  'rh': f'{site.TARGET_STATION}_rh_ok',
                  'batt': f'{site.TARGET_STATION}_batt_ok'}

# Panels of the record figure, response first.
ON_STRUCTURE_COLUMNS = (TARGET_COLUMN, 'tair_str', 'rh_str')

# ---------------------------------------------------------------------------
# Gap anatomy and segmentation
# ---------------------------------------------------------------------------
# Autoregressive windows to evaluate against the record's contiguity, in hours.
# The chosen value is read off NP_03 in step 2 and set in step 6.
SURVIVAL_LAG_HOURS = (4, 8, 12, 24, 48)
SURVIVAL_FORECAST_HOURS = (8, 24)
```

- [x] **Step 3: Rewrite step 1 to load on the native grid**

Replace the loading cell (lines 134–164) with:

```python
# %%
inclination, inclination_provenance = proxies.load_response(
    ARCHIVE_CSV, column=TARGET_COLUMN, honour_spike=True,
    freq=NATIVE_FREQ, tz=site.SITE_TZ, min_count=1)

sensor_current, sensor_provenance = proxies.load_sensor_forcings(
    ARCHIVE_CSV, column_map=STR_MAP_CURRENT, freq=NATIVE_FREQ,
    tz=site.SITE_TZ, honour_suspect=True, min_count=1)
sensor_legacy, _ = proxies.load_sensor_forcings(
    ARCHIVE_CSV, column_map=STR_MAP_LEGACY, freq=NATIVE_FREQ,
    tz=site.SITE_TZ, honour_suspect=True, min_count=1)
sensor = proxies.join_eras([sensor_current, sensor_legacy])

record = proxies.harmonise([sensor, inclination.to_frame()], freq=NATIVE_FREQ)
window = record.loc[site.to_utc(pd.DatetimeIndex([SEGMENT_START]))[0]:]

print(f'Window: {window.index.min()} to {window.index.max()} '
      f'({len(window):,} slots on the {NATIVE_FREQ} grid)')
print(f'First accepted inclination inside the window: '
      f'{window[TARGET_COLUMN].dropna().index.min()}')

coverage = pd.DataFrame({
    'accepted': window[list(ON_STRUCTURE_COLUMNS)].notna().sum(),
    'coverage': window[list(ON_STRUCTURE_COLUMNS)].notna().mean(),
})
coverage.index.name = 'channel'
display(coverage)

coverage.reset_index().to_csv(OUTPUT_DIR / 'NP_01_window_coverage.csv',
                              index=False)
```

- [x] **Step 4: Add the step 2 Markdown cell**

```markdown
# %% [markdown]
# ## 2 · The anatomy of what is missing
#
# Study 01 measured how much of the record is absent and recorded explicitly
# that it had not measured how that absence is *shaped*. It is measured here,
# because the shape decides what kind of problem filling it is: recovering a
# scattered sample from its immediate neighbours is a different proposition
# from reconstructing a season. The same table also decides how long an
# autoregressive window this record can afford, since a window can only be
# trained inside a run of complete rows long enough to hold it.
#
# ### Parameter Tuning Guidance
#
# **`SURVIVAL_LAG_HOURS`** — autoregressive window lengths to evaluate, in
# hours. Accepts any increasing sequence of positive integers; default
# `(4, 8, 12, 24, 48)`. Raising the largest value costs training windows
# quadratically on a fragmented record: each configuration is reported so the
# choice made in step 6 can be read off the table rather than assumed.
#
# **`SURVIVAL_FORECAST_HOURS`** — forecast lengths to pair with each window;
# default `(8, 24)`. A configuration survives only in segments at least
# `lag + forecast` long, so this parameter and the one above trade against each
# other and are reported jointly.
```

- [x] **Step 5: Add the step 2 code cell**

```python
# %%
gaps = prediction.gap_inventory(window[TARGET_COLUMN], freq=NATIVE_FREQ)
print(f'{len(gaps):,} gaps, {gaps["duration_h"].sum():,.0f} missing hours')
display(gaps.groupby('gap_class')
        .agg(gaps=('n_slots', 'size'), hours=('duration_h', 'sum')))

survival = prediction.segment_survival(
    window.rename(columns={TARGET_COLUMN: 'y'}),
    required=['y'] + [f'{name}_str' for name in PREDICTOR_COLUMNS],
    lag_hours=list(SURVIVAL_LAG_HOURS),
    forecast_hours=list(SURVIVAL_FORECAST_HOURS),
    freq=NATIVE_FREQ)
display(survival)

gaps.to_csv(OUTPUT_DIR / 'NP_02_gap_inventory.csv', index=False)
survival.to_csv(OUTPUT_DIR / 'NP_03_segment_survival.csv', index=False)

figures.plot_gap_anatomy(
    gaps, title='How the missing time is shaped',
    save_path=str(OUTPUT_DIR), filename='NP_F02_gap_anatomy')
figures.plot_segment_survival(
    survival, title='Training windows surviving contiguous segmentation',
    save_path=str(OUTPUT_DIR), filename='NP_F03_segment_survival')
plt.show()
```

- [x] **Step 6: Add the step 3 Markdown and code cells**

```markdown
# %% [markdown]
# ## 3 · Which cadence, and which target
#
# Two choices are settled here by measurement rather than by preference: whether
# the level or its first difference is the quantity to score, and at which
# spacing. A level whose lag-one autocorrelation approaches unity cannot be
# scored honestly, because any error metric computed against it measures the
# sampling interval rather than the model. A first difference whose lag-one
# autocorrelation is *negative* is dominated by measurement noise on the level
# rather than by the increment it is meant to carry. Both quantities are
# reported at both candidate cadences, together with the coupling to air
# temperature and the drift each implies.
#
# ### Parameter Tuning Guidance
#
# **`MODEL_FREQ_A`** — grid for the decomposition and expectation model;
# default `'20min'`, the archive's native spacing and the rate at which a
# deployed system receives a reading. The anomaly question does not require
# differencing, so the noise that spoils the twenty-minute difference does not
# affect it.
#
# **`MODEL_FREQ_B`** — grid for the forecast model; default `'1h'`. Set it to
# `'20min'` only if `NP_04` shows a non-negative change autocorrelation there;
# on this record it does not.
```

```python
# %%
cadence = prediction.cadence_evidence(
    window[TARGET_COLUMN], window['tair_str'], era=window.get('era'),
    cadences=(MODEL_FREQ_A, MODEL_FREQ_B), freq=NATIVE_FREQ)
display(cadence)

cadence.to_csv(OUTPUT_DIR / 'NP_04_cadence_evidence.csv', index=False)
figures.plot_cadence_evidence(
    cadence, title='What fixes the cadence and the target',
    save_path=str(OUTPUT_DIR), filename='NP_F04_cadence_evidence')
plt.show()
```

- [x] **Step 7: Sync and execute the notebook**

```bash
cd studies/04_neuralprophet_inclination_prediction
jupytext --sync neuralprophet_inclination_prediction_study.py
jupyter nbconvert --to notebook --execute --inplace \
  neuralprophet_inclination_prediction_study.ipynb --ExecutePreprocessor.timeout=1800
```

- [x] **Step 8: Verify the numbers against the reference table**

```bash
python - <<'PY'
import pandas as pd
out = 'studies/04_neuralprophet_inclination_prediction/outputs'
cov = pd.read_csv(f'{out}/NP_01_window_coverage.csv')
gaps = pd.read_csv(f'{out}/NP_02_gap_inventory.csv')
cad = pd.read_csv(f'{out}/NP_04_cadence_evidence.csv').set_index('cadence')
print('inclination coverage', float(cov.loc[cov.channel.str.startswith('inc'), 'coverage'].iloc[0]))
print('gaps', len(gaps), 'missing hours', round(gaps.duration_h.sum()))
print('level autocorr', round(cad.loc['20min', 'level_autocorr1'], 5))
print('change autocorr 20min', round(cad.loc['20min', 'change_autocorr1'], 4))
print('change autocorr 1h', round(cad.loc['1h', 'change_autocorr1'], 4))
PY
```

Expected, matching the reference table: coverage `0.756`, `1647` gaps, `6789` missing hours,
level autocorrelation `0.99771`, change autocorrelation `-0.0581` and `+0.3494`. A discrepancy
means the loaders are not reading the native grid — check that every `freq=NATIVE_FREQ` and
`min_count=1` argument is present.

- [x] **Step 9: Commit**

```bash
git add studies/04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.py \
        studies/04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.ipynb
git commit -m "$(cat <<'EOF'
feat(study04): measure the record, its gap anatomy and the cadence evidence

Study 01 recorded that it had not measured how the missing time is distributed.
It is measured here: four outages carry 72.8% of the absent hours, while 1,282
gaps of an hour or less carry 8.6% of them and 78% of the gap count, and it is
those that break contiguity and decide how long an autoregressive window this
record can afford.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

> **CHECKPOINT 1 and 2 — show the user:** `NP_01`–`NP_04`, `NP_F02`–`NP_F04`, and the verification
> output of Step 7. Every parameter must be visible in the parameter cell; no notebook cell longer
> than about ten lines. If the cadence evidence contradicts design decisions D1–D3, stop and change
> the design rather than the evidence. Wait for approval before Task 7.

---

## Phase 3 · Library additions

Every change in this phase is additive. The burden of proof is on the change: after each task,
Study 03's tests and Study 04's existing `test_prediction.py` must pass untouched, which is what
demonstrates that no existing caller's behaviour moved.

### Task 7: Changepoints on covered time, and a decomposition path

**Files:**
- Modify: `studies/shmlib/prediction.py:454-551` (`neuralprophet_backtest`), `:554-603`
  (`neuralprophet_predict`), and append `covered_changepoints`, `decompose_components`
- Test: `studies/04_neuralprophet_inclination_prediction/tests/test_decomposition.py`

**Interfaces:**
- Consumes: `_model_frame` (`prediction.py:371`), `_analysis_freq` (`:385`).
- Produces:
  - `covered_changepoints(index, n_changepoints, observed_mask=None) -> pd.DatetimeIndex`
  - `neuralprophet_backtest(..., growth='off', changepoints=None, n_changepoints=10, freq=None, decompose=False)`
    — five new keyword arguments, all defaulting to today's behaviour
  - `neuralprophet_predict(..., decompose=False)`
  - `decompose_components(model, frame, regressors=(), freq=None) -> pd.DataFrame` indexed by
    timestamp with the model's component columns plus `y`, `yhat1` and `residual`

- [ ] **Step 1: Write the failing test**

Create `studies/04_neuralprophet_inclination_prediction/tests/test_decomposition.py`:

```python
"""
Unit tests for the decomposition, changepoint and metric additions to shmlib.

The NeuralProphet-backed tests are deliberately tiny: they check that the
wrapper hands the model what it promised and reshapes what comes back, not that
the model is accurate.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import prediction


class TestCoveredChangepoints(unittest.TestCase):

    def test_changepoints_avoid_an_uncovered_stretch(self):
        index = pd.date_range('2024-01-01', periods=1000, freq='1h')
        observed = pd.Series(True, index=index)
        observed.iloc[300:700] = False
        points = prediction.covered_changepoints(index, 5, observed)
        self.assertEqual(len(points), 5)
        hole = pd.Interval(index[300].value, index[699].value)
        for point in points:
            self.assertNotIn(point.value, hole)

    def test_without_a_mask_every_timestamp_counts_as_covered(self):
        index = pd.date_range('2024-01-01', periods=100, freq='1h')
        points = prediction.covered_changepoints(index, 4)
        self.assertEqual(len(points), 4)
        self.assertTrue(points.is_monotonic_increasing)

    def test_asking_for_more_changepoints_than_covered_samples_is_clipped(self):
        index = pd.date_range('2024-01-01', periods=10, freq='1h')
        observed = pd.Series(False, index=index)
        observed.iloc[:3] = True
        points = prediction.covered_changepoints(index, 8, observed)
        self.assertLessEqual(len(points), 3)


class TestDecomposeComponents(unittest.TestCase):

    def _frame(self, n=24 * 40):
        index = pd.date_range('2024-01-01', periods=n, freq='1h')
        driver = 10.0 * np.sin(np.arange(n) / 24.0 * 2 * np.pi)
        return pd.DataFrame({'y': -2.0 * driver + 0.001 * np.arange(n),
                             'tair': driver}, index=index)

    def test_returns_named_components_that_sum_towards_the_prediction(self):
        frame = self._frame()
        model, _ = prediction.neuralprophet_backtest(
            frame, frame, regressors=('tair',), task='nowcast', epochs=1,
            growth='linear', n_changepoints=2, quantiles=())
        components = prediction.decompose_components(
            model, frame, regressors=('tair',))
        self.assertIn('trend', components.columns)
        self.assertIn('season_daily', components.columns)
        self.assertIn('future_regressor_tair', components.columns)
        self.assertIn('residual', components.columns)
        parts = components[['trend', 'season_daily',
                            'future_regressor_tair']].sum(axis=1)
        np.testing.assert_allclose(parts.to_numpy(),
                                   components['yhat1'].to_numpy(),
                                   rtol=1e-3, atol=1e-3)

    def test_residual_is_observed_minus_prediction(self):
        frame = self._frame()
        model, _ = prediction.neuralprophet_backtest(
            frame, frame, regressors=('tair',), task='nowcast', epochs=1,
            quantiles=())
        components = prediction.decompose_components(
            model, frame, regressors=('tair',))
        expected = components['y'] - components['yhat1']
        np.testing.assert_allclose(components['residual'].dropna().to_numpy(),
                                   expected.dropna().to_numpy(), atol=1e-9)


class TestBacktestDefaultsAreUnchanged(unittest.TestCase):

    def test_growth_still_defaults_to_off(self):
        import inspect
        signature = inspect.signature(prediction.neuralprophet_backtest)
        self.assertEqual(signature.parameters['growth'].default, 'off')
        self.assertIs(signature.parameters['changepoints'].default, None)
        self.assertIs(signature.parameters['decompose'].default, False)
        self.assertIs(signature.parameters['freq'].default, None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python 04_neuralprophet_inclination_prediction/tests/test_decomposition.py
```
Expected: `AttributeError: module 'shmlib.prediction' has no attribute 'covered_changepoints'`.

- [ ] **Step 3: Add `covered_changepoints`**

Append to `studies/shmlib/prediction.py`:

```python
def covered_changepoints(index, n_changepoints, observed_mask=None):
    """
    Trend changepoints placed on time the record actually covers.

    A changepoint placed inside an outage is constrained by no observation, and
    the trend is free to move arbitrarily across it. Placing changepoints at
    quantiles of the observed timestamps rather than uniformly along the axis
    keeps every one of them anchored to data.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Full analysis grid, covered and uncovered alike.
    n_changepoints : int
        Number of changepoints requested. Silently clipped when fewer covered
        samples exist.
    observed_mask : pd.Series or array-like or None, optional
        Boolean per timestamp, true where a value is present. ``None`` treats
        every timestamp as covered. Default ``None``.

    Returns
    -------
    pd.DatetimeIndex
        Increasing changepoint locations, of length at most ``n_changepoints``.
    """
    index = pd.DatetimeIndex(index)
    if observed_mask is None:
        covered = index
    else:
        mask = _as_series(observed_mask, index).fillna(False).astype(bool)
        covered = index[mask.to_numpy()]
    if len(covered) == 0:
        return pd.DatetimeIndex([])

    count = int(min(int(n_changepoints), len(covered)))
    if count <= 0:
        return pd.DatetimeIndex([])
    quantiles = np.linspace(0.0, 1.0, count + 2)[1:-1]
    positions = np.unique((quantiles * (len(covered) - 1)).round().astype(int))
    return pd.DatetimeIndex(covered[positions])
```

- [ ] **Step 4: Add the new keyword arguments to `neuralprophet_backtest`**

In `studies/shmlib/prediction.py`, change the signature at line 454 to:

```python
def neuralprophet_backtest(train, test, regressors=(), task='forecast',
                           n_lags=24, n_forecasts=24, regressor_lags=12,
                           horizons=None, epochs=30, yearly=False,
                           quantiles=(0.05, 0.95), seed=0, growth='off',
                           changepoints=None, n_changepoints=10, freq=None,
                           decompose=False):
```

Add to the docstring's `Parameters` section, after `seed`:

```
    growth : {'off', 'linear'}, optional
        Trend specification. ``'off'`` fits a constant offset and is the
        default, which is what a change-valued target needs; ``'linear'``
        fits a piecewise-linear trend and is what a level-valued target needs.
    changepoints : pd.DatetimeIndex or None, optional
        Explicit changepoint locations, normally from
        ``covered_changepoints``. ``None`` lets NeuralProphet space
        ``n_changepoints`` of them along the training range, which on a gapped
        record can place one inside an outage. Default ``None``.
    n_changepoints : int, optional
        Number of changepoints when ``changepoints`` is ``None``. Ignored
        otherwise. Default ``10``.
    freq : str or None, optional
        Frequency handed to NeuralProphet. ``None`` infers it from the training
        index. Default ``None``.
    decompose : bool, optional
        Whether ``model.predict`` returns component columns beside the
        prediction. Default ``False``, which is what a scoring run needs.
```

Replace the model construction at lines 512–530 with:

```python
    model = NeuralProphet(
        growth=growth,
        changepoints=(list(pd.DatetimeIndex(changepoints))
                      if changepoints is not None else None),
        n_changepoints=int(n_changepoints),
        n_lags=int(n_lags),
        n_forecasts=int(n_forecasts),
        daily_seasonality=True,
        weekly_seasonality=False,
        yearly_seasonality=yearly,
        normalize='standardize',
        global_normalization=True,
        global_time_normalization=True,
        unknown_data_normalization=True,
        impute_missing=False,
        drop_missing=False,
        loss_func='SmoothL1Loss',
        learning_rate=0.01,
        epochs=int(epochs),
        quantiles=list(quantiles or ()),
        collect_metrics=False,
    )
```

Replace line 539 with:

```python
    fit_freq = freq if freq is not None else _analysis_freq(train.index)
```

Replace line 546 with:

```python
    wide = model.predict(predict_df, decompose=bool(decompose))
```

- [ ] **Step 5: Add `decompose` to `neuralprophet_predict`**

Change the signature at line 554 to:

```python
def neuralprophet_predict(model, frame, regressors=(), horizons=(1,),
                          quantiles=(0.05, 0.95), decompose=False):
```

Document it as `decompose : bool, optional — Whether component columns are requested from the
model. Default ``False``.` and replace line 586 with:

```python
    wide = model.predict(model_frame, decompose=bool(decompose))
```

- [ ] **Step 6: Add `decompose_components`**

Append to `studies/shmlib/prediction.py`:

```python
def decompose_components(model, frame, regressors=(), freq=None):
    """
    The additive parts NeuralProphet fitted, aligned to the study's index.

    NeuralProphet returns its decomposition as extra columns beside the
    prediction. This reshapes them into one timestamp-indexed table, adds the
    observed value and the residual, and leaves the component names as the model
    produced them, so that a reader can trace any column back to the term that
    made it.

    Parameters
    ----------
    model : object
        Fitted NeuralProphet model, exposing ``predict(df, decompose=True)``.
    frame : pd.DataFrame
        Datetime-indexed rows with ``y``, the regressors, and optionally
        ``segment_id``.
    regressors : sequence of str, optional
        Regressor columns to pass through. Default empty.
    freq : str or None, optional
        Unused by the model at prediction time; accepted so callers may pass
        the study's grid for symmetry with ``neuralprophet_backtest``. Default
        ``None``.

    Returns
    -------
    pd.DataFrame
        Indexed by timestamp, carrying every component column the model
        produced, plus ``y``, ``yhat1``, ``residual`` and, when the input was
        segmented, ``ID``.
    """
    model_frame = _model_frame(frame, regressors)
    if getattr(model, 'n_lags', None) == 0:
        model_frame['y'] = model_frame['y'].fillna(0.0)
    wide = model.predict(model_frame, decompose=True)

    reserved = {'ds', 'y'}
    components = [c for c in wide.columns
                  if c not in reserved and not c.startswith('yhat')
                  and '%' not in c and c != 'ID']

    out = wide.loc[:, ['ds'] + components].copy()
    out['yhat1'] = wide['yhat1'] if 'yhat1' in wide.columns else np.nan
    if 'ID' in wide.columns:
        out['ID'] = wide['ID']
    out = out.set_index('ds')
    out.index.name = frame.index.name

    observed = pd.to_numeric(frame['y'], errors='coerce') if 'y' in frame else None
    out['y'] = observed.reindex(out.index) if observed is not None else np.nan
    out['residual'] = out['y'] - out['yhat1']
    return out
```

- [ ] **Step 7: Run the tests to verify they pass**

```bash
python 04_neuralprophet_inclination_prediction/tests/test_decomposition.py
```
Expected: `OK`, 6 tests.

- [ ] **Step 8: Prove existing callers are unaffected**

```bash
python 04_neuralprophet_inclination_prediction/tests/test_prediction.py
python 03_thermomechanical_response/tests/test_shmlib_study03.py
python shmlib/tests/test_shmlib.py
```
Expected: all `OK`. These are the tests that certify the adaptation changed nothing for anyone
already calling the two wrappers.

- [ ] **Step 9: Commit**

```bash
git add studies/shmlib/prediction.py \
        studies/04_neuralprophet_inclination_prediction/tests/test_decomposition.py
git commit -m "$(cat <<'EOF'
feat(shmlib): expose NeuralProphet's decomposition and anchor its changepoints

The two wrappers hard-coded decompose=False, so the component structure the
model fits was unreachable, and growth='off', so no trend existed to read. Both
become parameters whose defaults reproduce the previous behaviour exactly.

Changepoints gain an explicit-location path, because the default spacing is
blind to coverage: on this record it would place one inside a 103-day outage,
where nothing constrains it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Component shares and residual diagnostics

**Files:**
- Modify: `studies/shmlib/prediction.py` (append after `decompose_components`)
- Test: `studies/04_neuralprophet_inclination_prediction/tests/test_decomposition.py`

**Interfaces:**
- Consumes: `decompose_components` output.
- Produces:
  - `component_variance_shares(components, columns=None) -> pd.DataFrame` with columns
    `component`, `variance`, `share`, `mean`, `peak_to_peak`, sorted by descending share, and a
    final `residual` row.
  - `residual_diagnostics(residuals, lags=(1, 24, 72)) -> pd.DataFrame` with columns `lag`,
    `lb_stat`, `lb_pvalue`, `n`, `std`, `mad`.

- [ ] **Step 1: Write the failing test**

Append to `studies/04_neuralprophet_inclination_prediction/tests/test_decomposition.py`, above the
`if __name__` block:

```python
class TestComponentVarianceShares(unittest.TestCase):

    def _components(self, n=500):
        index = pd.date_range('2024-01-01', periods=n, freq='1h')
        return pd.DataFrame({
            'trend': np.linspace(0.0, 1.0, n),
            'season_daily': 10.0 * np.sin(np.arange(n) / 24.0 * 2 * np.pi),
            'future_regressor_tair': np.zeros(n),
            'residual': np.full(n, 0.5),
            'y': np.zeros(n),
            'yhat1': np.zeros(n),
        }, index=index)

    def test_shares_sum_to_one_and_rank_by_variance(self):
        table = prediction.component_variance_shares(self._components())
        self.assertAlmostEqual(table['share'].sum(), 1.0, places=6)
        self.assertEqual(table['component'].iloc[0], 'season_daily')

    def test_a_constant_component_has_zero_variance_and_a_reported_mean(self):
        table = prediction.component_variance_shares(
            self._components()).set_index('component')
        self.assertAlmostEqual(table.loc['residual', 'variance'], 0.0)
        self.assertAlmostEqual(table.loc['residual', 'mean'], 0.5)

    def test_y_and_yhat_are_never_treated_as_components(self):
        table = prediction.component_variance_shares(self._components())
        self.assertNotIn('y', list(table['component']))
        self.assertNotIn('yhat1', list(table['component']))


class TestResidualDiagnostics(unittest.TestCase):

    def test_white_noise_is_not_rejected(self):
        rng = np.random.default_rng(0)
        index = pd.date_range('2024-01-01', periods=2000, freq='1h')
        residuals = pd.Series(rng.normal(size=2000), index=index)
        table = prediction.residual_diagnostics(residuals, lags=(1, 24))
        self.assertTrue((table['lb_pvalue'] > 0.01).all())
        self.assertEqual(list(table['lag']), [1, 24])

    def test_a_periodic_residual_is_rejected(self):
        index = pd.date_range('2024-01-01', periods=2000, freq='1h')
        residuals = pd.Series(np.sin(np.arange(2000) / 24.0 * 2 * np.pi),
                              index=index)
        table = prediction.residual_diagnostics(residuals, lags=(24,))
        self.assertLess(table['lb_pvalue'].iloc[0], 0.01)

    def test_scale_columns_describe_the_residual(self):
        index = pd.date_range('2024-01-01', periods=100, freq='1h')
        residuals = pd.Series(np.arange(100.0), index=index)
        table = prediction.residual_diagnostics(residuals, lags=(1,))
        self.assertEqual(table['n'].iloc[0], 100)
        self.assertGreater(table['std'].iloc[0], 0.0)
        self.assertGreater(table['mad'].iloc[0], 0.0)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python 04_neuralprophet_inclination_prediction/tests/test_decomposition.py
```
Expected: `AttributeError: module 'shmlib.prediction' has no attribute 'component_variance_shares'`.

- [ ] **Step 3: Implement**

Append to `studies/shmlib/prediction.py`:

```python
_NON_COMPONENT_COLUMNS = ('y', 'yhat1', 'ID')


def component_variance_shares(components, columns=None):
    """
    What share of the fitted variation each additive component carries.

    This is the decomposition's headline claim expressed as a number: a record
    whose daily component carries most of the variance is a thermometer, and one
    whose trend does is a structure that is moving. The residual is included as
    a component so that the shares are comparable and sum to one.

    Parameters
    ----------
    components : pd.DataFrame
        Output of ``decompose_components``.
    columns : sequence of str or None, optional
        Components to include. ``None`` takes every column except ``y``,
        ``yhat1`` and ``ID``. Default ``None``.

    Returns
    -------
    pd.DataFrame
        Columns ``component``, ``variance``, ``share``, ``mean`` and
        ``peak_to_peak``, ordered by descending share.
    """
    if columns is None:
        columns = [c for c in components.columns
                   if c not in _NON_COMPONENT_COLUMNS]
    frame = components.loc[:, list(columns)].apply(
        pd.to_numeric, errors='coerce')

    variance = frame.var(ddof=0)
    total = float(variance.sum())
    table = pd.DataFrame({
        'component': variance.index,
        'variance': variance.to_numpy(),
        'share': (variance / total).to_numpy() if total > 0 else np.nan,
        'mean': frame.mean().to_numpy(),
        'peak_to_peak': (frame.max() - frame.min()).to_numpy(),
    })
    return (table.sort_values('share', ascending=False, kind='stable')
            .reset_index(drop=True))


def residual_diagnostics(residuals, lags=(1, 24, 72)):
    """
    Whether anything is left in the residual, and on what scale it sits.

    Structure surviving in the residual means a component of the model is
    missing. That matters twice over here: it makes the decomposition's
    attribution wrong, and it makes a residual-based alarm fire on model error
    rather than on the structure.

    Parameters
    ----------
    residuals : pd.Series
        Observed minus predicted, indexed by timestamp. Missing values are
        dropped before testing.
    lags : sequence of int, optional
        Ljung-Box lags to test. Default ``(1, 24, 72)``.

    Returns
    -------
    pd.DataFrame
        One row per lag, with ``lag``, ``lb_stat``, ``lb_pvalue``, and the
        scale columns ``n``, ``std`` and ``mad`` repeated on every row.

    Notes
    -----
    Requires ``statsmodels``.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    values = pd.to_numeric(residuals, errors='coerce').dropna()
    lags = [int(lag) for lag in lags]
    result = acorr_ljungbox(values, lags=lags, return_df=True)

    scale = {
        'n': int(values.size),
        'std': float(values.std(ddof=1)) if values.size > 1 else np.nan,
        'mad': float((values - values.median()).abs().median()),
    }
    return pd.DataFrame({
        'lag': lags,
        'lb_stat': result['lb_stat'].to_numpy(),
        'lb_pvalue': result['lb_pvalue'].to_numpy(),
        'n': scale['n'],
        'std': scale['std'],
        'mad': scale['mad'],
    })
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python 04_neuralprophet_inclination_prediction/tests/test_decomposition.py
```
Expected: `OK`, 12 tests.

- [ ] **Step 5: Prove existing callers are unaffected**

```bash
python 04_neuralprophet_inclination_prediction/tests/test_prediction.py
python 03_thermomechanical_response/tests/test_shmlib_study03.py
python shmlib/tests/test_shmlib.py
```
Expected: all `OK`.

- [ ] **Step 6: Commit**

```bash
git add studies/shmlib/prediction.py \
        studies/04_neuralprophet_inclination_prediction/tests/test_decomposition.py
git commit -m "$(cat <<'EOF'
feat(shmlib): quantify component shares and test residuals for structure

A decomposition is only a claim until the shares are counted, and a residual is
only usable as an alarm statistic once it has been shown to carry no structure
of its own.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Extend `score_predictions` with scale-free and interval metrics

**Files:**
- Modify: `studies/shmlib/prediction.py:213-285` (`_score_group`, `score_predictions`)
- Test: `studies/04_neuralprophet_inclination_prediction/tests/test_decomposition.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `score_predictions(frame, group_cols, naive_scale=None, alpha=0.10)` returning the
  existing columns `n`, `mae`, `rmse`, `bias`, `r2`, `coverage_q05_q95`, `width_q05_q95`
  **unchanged in name, order and value**, followed by the new `mase`, `pinball_q05`,
  `pinball_q95`, `interval_score`. Tasks 15 and 18 consume all of them.

MASE requires an in-sample naive scale that the notebook computes and passes; when
`naive_scale` is `None` the column is `NaN` rather than absent, so the schema never depends on the
argument.

- [ ] **Step 1: Write the failing test**

Append to `studies/04_neuralprophet_inclination_prediction/tests/test_decomposition.py`, above the
`if __name__` block:

```python
class TestScorePredictionsExtension(unittest.TestCase):

    def _frame(self):
        return pd.DataFrame({
            'y': [0.0, 1.0, 2.0, 3.0],
            'yhat': [0.5, 1.5, 1.5, 3.5],
            'q05': [-1.0, 0.0, 1.0, 2.0],
            'q95': [1.0, 2.0, 3.0, 4.0],
            'model': ['a', 'a', 'b', 'b'],
        })

    def test_existing_columns_keep_their_names_order_and_values(self):
        scores = prediction.score_predictions(self._frame(), ['model'])
        expected = ['model', 'n', 'mae', 'rmse', 'bias', 'r2',
                    'coverage_q05_q95', 'width_q05_q95']
        self.assertEqual(list(scores.columns)[:len(expected)], expected)
        self.assertAlmostEqual(scores['mae'].iloc[0], 0.5)

    def test_new_columns_are_appended_after_the_existing_ones(self):
        scores = prediction.score_predictions(self._frame(), ['model'])
        self.assertEqual(list(scores.columns)[-4:],
                         ['mase', 'pinball_q05', 'pinball_q95',
                          'interval_score'])

    def test_mase_is_missing_until_a_naive_scale_is_supplied(self):
        without = prediction.score_predictions(self._frame(), ['model'])
        self.assertTrue(without['mase'].isna().all())
        withscale = prediction.score_predictions(
            self._frame(), ['model'], naive_scale=0.5)
        self.assertAlmostEqual(withscale['mase'].iloc[0], 1.0)

    def test_pinball_loss_penalises_the_wrong_side_of_each_quantile(self):
        frame = pd.DataFrame({'y': [10.0], 'yhat': [0.0],
                              'q05': [0.0], 'q95': [1.0], 'model': ['a']})
        scores = prediction.score_predictions(frame, ['model'])
        # y above q95: the 0.95 quantile is penalised at weight 0.95.
        self.assertAlmostEqual(scores['pinball_q95'].iloc[0], 0.95 * 9.0)
        self.assertAlmostEqual(scores['pinball_q05'].iloc[0], 0.05 * 10.0)

    def test_interval_score_adds_a_violation_penalty_to_the_width(self):
        frame = pd.DataFrame({'y': [3.0], 'yhat': [0.0],
                              'q05': [0.0], 'q95': [1.0], 'model': ['a']})
        scores = prediction.score_predictions(frame, ['model'], alpha=0.10)
        self.assertAlmostEqual(scores['interval_score'].iloc[0],
                               1.0 + (2.0 / 0.10) * 2.0)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python 04_neuralprophet_inclination_prediction/tests/test_decomposition.py
```
Expected: `KeyError: 'mase'`.

- [ ] **Step 3: Extend `_score_group`**

In `studies/shmlib/prediction.py`, change `_score_group`'s signature to
`def _score_group(group, naive_scale=None, alpha=0.10):` and append, immediately before it
returns its `pd.Series`:

```python
    # Scale-free error, so that horizons and cadences are comparable on one
    # axis. The scale is the caller's in-sample naive mean absolute error; it is
    # not derived here, because a scale computed on the evaluation rows would
    # make the metric self-referential.
    scores['mase'] = (scores['mae'] / naive_scale
                      if naive_scale not in (None, 0) and np.isfinite(naive_scale)
                      else np.nan)

    lower_col, upper_col = 'q05', 'q95'
    if lower_col in group.columns and upper_col in group.columns:
        y = pd.to_numeric(group['y'], errors='coerce')
        lower = pd.to_numeric(group[lower_col], errors='coerce')
        upper = pd.to_numeric(group[upper_col], errors='coerce')
        complete = y.notna() & lower.notna() & upper.notna()
        y, lower, upper = y[complete], lower[complete], upper[complete]

        # Pinball loss scores each quantile on its own terms rather than only
        # asking whether the pair happened to bracket the observation.
        for column, quantile, forecast in ((f'pinball_{lower_col}', 0.05, lower),
                                           (f'pinball_{upper_col}', 0.95, upper)):
            error = y - forecast
            loss = np.where(error >= 0, quantile * error,
                            (quantile - 1.0) * error)
            scores[column] = float(np.mean(loss)) if len(loss) else np.nan

        # Winkler interval score: width, plus a penalty proportional to how far
        # outside the interval the observation fell.
        width = upper - lower
        penalty = np.where(y < lower, (2.0 / alpha) * (lower - y), 0.0) \
            + np.where(y > upper, (2.0 / alpha) * (y - upper), 0.0)
        scores['interval_score'] = (float(np.mean(width + penalty))
                                    if len(width) else np.nan)
    else:
        scores['pinball_q05'] = np.nan
        scores['pinball_q95'] = np.nan
        scores['interval_score'] = np.nan
```

- [ ] **Step 4: Thread the arguments through `score_predictions`**

Change the signature at line 249 to
`def score_predictions(frame, group_cols, naive_scale=None, alpha=0.10):`, document both in the
docstring's `Parameters` section, extend the `Returns` description with the four new columns, and
pass them to every `_score_group` call inside the function.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python 04_neuralprophet_inclination_prediction/tests/test_decomposition.py
```
Expected: `OK`, 17 tests.

- [ ] **Step 6: Prove existing callers are unaffected**

```bash
python 04_neuralprophet_inclination_prediction/tests/test_prediction.py
python 03_thermomechanical_response/tests/test_shmlib_study03.py
python shmlib/tests/test_shmlib.py
```
Expected: all `OK`. `TestScorePredictions.test_grouped_scores_and_quantile_interval_metrics`
(`test_prediction.py:135`) is the one that certifies the existing columns did not move.

- [ ] **Step 7: Commit**

```bash
git add studies/shmlib/prediction.py \
        studies/04_neuralprophet_inclination_prediction/tests/test_decomposition.py
git commit -m "$(cat <<'EOF'
feat(shmlib): add scale-free and interval-calibration metrics to scoring

Coverage alone can be bought by widening an interval, and an absolute error is
not comparable across horizons. Add MASE against a caller-supplied naive scale,
pinball loss for each quantile, and the Winkler interval score.

Existing columns keep their names, order and values; the four new ones are
appended.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: `shmlib.monitoring` — control charts and alarm episodes

**Files:**
- Create: `studies/shmlib/monitoring.py`
- Create: `studies/shmlib/tests/test_monitoring.py`
- Modify: `studies/shmlib/__init__.py` (add `monitoring` to the exported module list if one exists)

**Interfaces:**
- Consumes: nothing from `shmlib`.
- Produces:
  - `reference_stats(residuals, start=None, end=None, robust=True) -> dict` with keys `mu`,
    `sigma`, `n`, `start`, `end`
  - `ewma_chart(residuals, mu, sigma, lam=0.2, L=3.0) -> pd.DataFrame` with `z`, `ewma`, `ucl`,
    `lcl`, `alarm`
  - `cusum_chart(residuals, mu, sigma, k=0.5, h=5.0) -> pd.DataFrame` with `z`, `cusum_high`,
    `cusum_low`, `limit`, `alarm`
  - `joint_alarm(ewma_alarm, cusum_alarm, window='24h') -> pd.Series`
  - `alarm_episodes(alarm, residuals=None) -> pd.DataFrame` with `start`, `end`, `duration_h`,
    `n_slots`, `mean_z`, `peak_abs_z`
  - `average_run_length(alarm, freq='20min') -> dict` with `n_episodes`, `hours`, `arl_hours`,
    `arl_days`

`heritageshm/monitoring.py` holds a working EWMA, CUSUM, joint-alarm and episode summary. Read it
before writing: this is a port to the studies' library with a standardised-residual interface and
its own tests, not a fresh invention. It is a port and not an import because `shmlib` is the only
library a study has.

- [ ] **Step 1: Read the prior art**

```bash
sed -n '1,220p' heritageshm/monitoring.py
```

- [ ] **Step 2: Write the failing test**

Create `studies/shmlib/tests/test_monitoring.py`:

```python
"""
Unit tests for shmlib.monitoring.

Run from studies/:  python shmlib/tests/test_monitoring.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import monitoring


def _quiet(n=2000, seed=0, freq='20min'):
    rng = np.random.default_rng(seed)
    index = pd.date_range('2024-01-01', periods=n, freq=freq)
    return pd.Series(rng.normal(0.0, 1.0, n), index=index)


class TestReferenceStats(unittest.TestCase):

    def test_robust_centre_and_scale_ignore_a_contaminated_tail(self):
        residuals = _quiet(1000)
        residuals.iloc[-50:] += 40.0
        stats = monitoring.reference_stats(residuals, robust=True)
        self.assertLess(abs(stats['mu']), 0.3)
        self.assertLess(stats['sigma'], 2.0)

    def test_a_window_restricts_which_rows_are_used(self):
        residuals = _quiet(1000)
        stats = monitoring.reference_stats(
            residuals, start=residuals.index[0], end=residuals.index[99])
        self.assertEqual(stats['n'], 100)


class TestEwmaChart(unittest.TestCase):

    def test_quiet_residuals_rarely_alarm(self):
        residuals = _quiet(2000)
        stats = monitoring.reference_stats(residuals)
        chart = monitoring.ewma_chart(residuals, stats['mu'], stats['sigma'])
        self.assertLess(chart['alarm'].mean(), 0.02)

    def test_a_sustained_step_alarms(self):
        residuals = _quiet(2000)
        residuals.iloc[1000:] += 3.0
        stats = monitoring.reference_stats(
            residuals, end=residuals.index[500])
        chart = monitoring.ewma_chart(residuals, stats['mu'], stats['sigma'])
        self.assertTrue(chart['alarm'].iloc[1000:1100].any())

    def test_limits_widen_with_L_and_narrow_with_lambda(self):
        residuals = _quiet(500)
        wide = monitoring.ewma_chart(residuals, 0.0, 1.0, lam=0.2, L=4.0)
        narrow = monitoring.ewma_chart(residuals, 0.0, 1.0, lam=0.2, L=2.0)
        self.assertTrue((wide['ucl'] > narrow['ucl']).all())


class TestCusumChart(unittest.TestCase):

    def test_a_small_persistent_shift_accumulates(self):
        residuals = _quiet(2000)
        residuals.iloc[1000:] += 0.8
        chart = monitoring.cusum_chart(residuals, 0.0, 1.0, k=0.5, h=5.0)
        self.assertFalse(chart['alarm'].iloc[:900].any())
        self.assertTrue(chart['alarm'].iloc[1000:1400].any())

    def test_both_directions_are_watched(self):
        residuals = _quiet(1000)
        residuals.iloc[500:] -= 2.0
        chart = monitoring.cusum_chart(residuals, 0.0, 1.0)
        self.assertTrue(chart['alarm'].iloc[500:].any())
        self.assertTrue((chart['cusum_low'].iloc[500:] > 0).any())


class TestJointAlarm(unittest.TestCase):

    def test_alarms_coincide_within_the_window(self):
        index = pd.date_range('2024-01-01', periods=100, freq='20min')
        ewma = pd.Series(False, index=index)
        cusum = pd.Series(False, index=index)
        ewma.iloc[10] = True
        cusum.iloc[12] = True
        joint = monitoring.joint_alarm(ewma, cusum, window='2h')
        self.assertTrue(joint.iloc[10:13].any())

    def test_isolated_alarms_on_one_chart_do_not_survive(self):
        index = pd.date_range('2024-01-01', periods=100, freq='20min')
        ewma = pd.Series(False, index=index)
        cusum = pd.Series(False, index=index)
        ewma.iloc[10] = True
        joint = monitoring.joint_alarm(ewma, cusum, window='2h')
        self.assertFalse(joint.any())


class TestAlarmEpisodes(unittest.TestCase):

    def test_consecutive_alarms_collapse_into_one_episode(self):
        index = pd.date_range('2024-01-01', periods=100, freq='20min')
        alarm = pd.Series(False, index=index)
        alarm.iloc[10:16] = True
        alarm.iloc[50] = True
        episodes = monitoring.alarm_episodes(alarm)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes['n_slots'].iloc[0], 6)
        self.assertAlmostEqual(episodes['duration_h'].iloc[0], 2.0)

    def test_residual_statistics_are_attached_when_supplied(self):
        index = pd.date_range('2024-01-01', periods=20, freq='20min')
        alarm = pd.Series(False, index=index)
        alarm.iloc[5:8] = True
        residuals = pd.Series(np.arange(20.0), index=index)
        episodes = monitoring.alarm_episodes(alarm, residuals)
        self.assertAlmostEqual(episodes['peak_abs_z'].iloc[0], 7.0)
        self.assertAlmostEqual(episodes['mean_z'].iloc[0], 6.0)


class TestAverageRunLength(unittest.TestCase):

    def test_run_length_is_watched_time_divided_by_episode_count(self):
        index = pd.date_range('2024-01-01', periods=144, freq='20min')
        alarm = pd.Series(False, index=index)
        alarm.iloc[10] = True
        alarm.iloc[100] = True
        result = monitoring.average_run_length(alarm, freq='20min')
        self.assertEqual(result['n_episodes'], 2)
        self.assertAlmostEqual(result['hours'], 48.0)
        self.assertAlmostEqual(result['arl_hours'], 24.0)
        self.assertAlmostEqual(result['arl_days'], 1.0)

    def test_a_chart_that_never_alarms_reports_an_infinite_run_length(self):
        index = pd.date_range('2024-01-01', periods=144, freq='20min')
        result = monitoring.average_run_length(
            pd.Series(False, index=index), freq='20min')
        self.assertEqual(result['n_episodes'], 0)
        self.assertTrue(np.isinf(result['arl_hours']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
```

- [ ] **Step 3: Run it to verify it fails**

```bash
python shmlib/tests/test_monitoring.py
```
Expected: `ModuleNotFoundError: No module named 'shmlib.monitoring'`.

- [ ] **Step 4: Implement the module**

Create `studies/shmlib/monitoring.py`:

```python
"""
Module: monitoring.py
Residual-based monitoring for the studies: reference statistics, EWMA and CUSUM
control charts, joint alarms, alarm episodes and run lengths.

Inputs are residual series - observed minus expected - indexed by timestamp.
Outputs are chart tables and episode tables. Nothing here fits a model or reads
a file; the caller supplies the residual and the settings, and every threshold
is an argument so that the study which chose it states its value.
"""

import numpy as np
import pandas as pd

# A normal distribution's median absolute deviation is this fraction of its
# standard deviation; dividing by it turns a MAD into a comparable sigma.
MAD_TO_SIGMA = 0.6744897501960817


def reference_stats(residuals, start=None, end=None, robust=True):
    """
    Centre and scale of the residual over an in-control reference window.

    Every limit drawn later is a multiple of these two numbers, so the window
    they are estimated on is the single most consequential choice in the whole
    monitoring step: a window containing the event to be detected calibrates the
    detector against the very thing it is meant to find.

    Parameters
    ----------
    residuals : pd.Series
        Observed minus expected, indexed by timestamp.
    start, end : pd.Timestamp or str or None, optional
        Inclusive bounds of the reference window. ``None`` extends to the
        respective end of the series. Default ``None``.
    robust : bool, optional
        When true, centre by the median and scale by the median absolute
        deviation rescaled to a standard deviation, so that a contaminated tail
        cannot inflate the limits it should be judged against. When false, use
        the mean and the sample standard deviation. Default ``True``.

    Returns
    -------
    dict
        ``mu``, ``sigma``, ``n``, ``start``, ``end``.
    """
    values = pd.to_numeric(residuals, errors='coerce').dropna()
    if start is not None:
        values = values.loc[pd.Timestamp(start):]
    if end is not None:
        values = values.loc[:pd.Timestamp(end)]

    if values.empty:
        return {'mu': np.nan, 'sigma': np.nan, 'n': 0,
                'start': None, 'end': None}

    if robust:
        mu = float(values.median())
        sigma = float((values - mu).abs().median() / MAD_TO_SIGMA)
    else:
        mu = float(values.mean())
        sigma = float(values.std(ddof=1))

    return {'mu': mu, 'sigma': sigma, 'n': int(values.size),
            'start': values.index.min(), 'end': values.index.max()}


def _standardise(residuals, mu, sigma):
    values = pd.to_numeric(residuals, errors='coerce')
    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series(np.nan, index=values.index)
    return (values - mu) / sigma


def ewma_chart(residuals, mu, sigma, lam=0.2, L=3.0):
    """
    Exponentially weighted moving average chart on the standardised residual.

    EWMA answers "has the level moved and stayed moved", which is the shape a
    structural departure takes. Missing residuals do not update the statistic;
    the chart carries its previous value across them rather than treating an
    absent reading as a zero.

    Parameters
    ----------
    residuals : pd.Series
        Observed minus expected, indexed by timestamp.
    mu, sigma : float
        Reference centre and scale, normally from ``reference_stats``.
    lam : float, optional
        Smoothing weight in ``(0, 1]``. Smaller reacts more slowly and detects
        smaller sustained shifts. Default ``0.2``.
    L : float, optional
        Control limits in standard deviations of the EWMA statistic. Larger
        means fewer false alarms and slower detection. Default ``3.0``.

    Returns
    -------
    pd.DataFrame
        Columns ``z``, ``ewma``, ``ucl``, ``lcl`` and ``alarm``, indexed as the
        input.
    """
    z = _standardise(residuals, mu, sigma)
    statistic = np.full(len(z), np.nan)
    limit = np.full(len(z), np.nan)

    current = 0.0
    step = 0
    for position, value in enumerate(z.to_numpy()):
        if np.isfinite(value):
            current = lam * value + (1.0 - lam) * current
            step += 1
        statistic[position] = current if step else np.nan
        if step:
            spread = np.sqrt((lam / (2.0 - lam))
                             * (1.0 - (1.0 - lam) ** (2 * step)))
            limit[position] = L * spread

    chart = pd.DataFrame({'z': z, 'ewma': statistic,
                          'ucl': limit, 'lcl': -limit}, index=z.index)
    chart['alarm'] = ((chart['ewma'] > chart['ucl'])
                      | (chart['ewma'] < chart['lcl'])).fillna(False)
    return chart


def cusum_chart(residuals, mu, sigma, k=0.5, h=5.0):
    """
    Two-sided tabular CUSUM on the standardised residual.

    CUSUM accumulates evidence, so it finds a shift too small to breach an EWMA
    limit on any single sample provided it persists. The two charts are run
    together and their alarms combined, because they fail in different ways.

    Parameters
    ----------
    residuals : pd.Series
        Observed minus expected, indexed by timestamp.
    mu, sigma : float
        Reference centre and scale, normally from ``reference_stats``.
    k : float, optional
        Slack in standard deviations; conventionally half the shift to be
        detected quickly. Default ``0.5``.
    h : float, optional
        Decision interval in standard deviations. Default ``5.0``.

    Returns
    -------
    pd.DataFrame
        Columns ``z``, ``cusum_high``, ``cusum_low``, ``limit`` and ``alarm``.
        ``cusum_low`` is reported as a positive magnitude.
    """
    z = _standardise(residuals, mu, sigma)
    high = np.zeros(len(z))
    low = np.zeros(len(z))

    running_high = 0.0
    running_low = 0.0
    for position, value in enumerate(z.to_numpy()):
        if np.isfinite(value):
            running_high = max(0.0, running_high + value - k)
            running_low = max(0.0, running_low - value - k)
        high[position] = running_high
        low[position] = running_low

    chart = pd.DataFrame({'z': z, 'cusum_high': high, 'cusum_low': low,
                          'limit': float(h)}, index=z.index)
    chart['alarm'] = (chart['cusum_high'] > h) | (chart['cusum_low'] > h)
    return chart


def joint_alarm(ewma_alarm, cusum_alarm, window='24h'):
    """
    Alarms that both charts raise within a coincidence window.

    Requiring coincidence trades a little sensitivity for a large reduction in
    isolated false alarms, which is the right trade for a monitoring system a
    person has to trust.

    Parameters
    ----------
    ewma_alarm, cusum_alarm : pd.Series
        Boolean alarm series on a shared index.
    window : str, optional
        Coincidence window, as a pandas offset string. Default ``'24h'``.

    Returns
    -------
    pd.Series
        Boolean, true where both charts alarmed within ``window`` of each other.
    """
    left = ewma_alarm.fillna(False).astype(bool)
    right = cusum_alarm.reindex(left.index).fillna(False).astype(bool)

    span = pd.Timedelta(window)
    nearby_left = left.rolling(span, center=True, min_periods=1).max().astype(bool)
    nearby_right = right.rolling(span, center=True, min_periods=1).max().astype(bool)
    return ((left & nearby_right) | (right & nearby_left)).rename('alarm')


def alarm_episodes(alarm, residuals=None):
    """
    Consecutive alarming samples collapsed into one row each.

    Parameters
    ----------
    alarm : pd.Series
        Boolean alarm series indexed by timestamp on a regular grid.
    residuals : pd.Series or None, optional
        Standardised residual, used to describe each episode's size. Default
        ``None``.

    Returns
    -------
    pd.DataFrame
        Columns ``start``, ``end``, ``duration_h``, ``n_slots``, ``mean_z`` and
        ``peak_abs_z``. Empty with those columns when nothing alarmed.
    """
    columns = ['start', 'end', 'duration_h', 'n_slots', 'mean_z', 'peak_abs_z']
    flags = alarm.fillna(False).astype(bool)
    index = pd.DatetimeIndex(flags.index)
    positions = np.flatnonzero(flags.to_numpy())
    if positions.size == 0:
        return pd.DataFrame(columns=columns)

    step_hours = ((index[1] - index[0]) / pd.Timedelta(hours=1)
                  if len(index) > 1 else np.nan)
    breaks = np.flatnonzero(np.diff(positions) != 1)
    starts = positions[np.r_[0, breaks + 1]]
    ends = positions[np.r_[breaks, positions.size - 1]]

    rows = []
    for start, end in zip(starts, ends):
        window = (residuals.iloc[start:end + 1]
                  if residuals is not None else None)
        rows.append({
            'start': index[start],
            'end': index[end],
            'duration_h': (end - start + 1) * step_hours,
            'n_slots': int(end - start + 1),
            'mean_z': float(window.mean()) if window is not None else np.nan,
            'peak_abs_z': (float(window.abs().max())
                           if window is not None else np.nan),
        })
    return pd.DataFrame(rows, columns=columns)


def average_run_length(alarm, freq='20min'):
    """
    Watched time per alarm episode - the false-alarm budget, in plain units.

    Reported on an in-control stretch, this is the number a deployment is
    actually tuned to: "one false alarm every N days". Every other detection
    figure in a study is only comparable at a stated run length.

    Parameters
    ----------
    alarm : pd.Series
        Boolean alarm series indexed by timestamp.
    freq : str, optional
        Spacing of the series. Default ``'20min'``.

    Returns
    -------
    dict
        ``n_episodes``, ``hours``, ``arl_hours``, ``arl_days``. The run lengths
        are infinite when nothing alarmed.
    """
    episodes = alarm_episodes(alarm)
    hours = float(len(alarm) * (pd.Timedelta(freq) / pd.Timedelta(hours=1)))
    count = int(len(episodes))
    arl_hours = hours / count if count else np.inf
    return {'n_episodes': count, 'hours': hours,
            'arl_hours': arl_hours, 'arl_days': arl_hours / 24.0}
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python shmlib/tests/test_monitoring.py
```
Expected: `OK`, 13 tests.

- [ ] **Step 6: Verify nothing else broke**

```bash
python shmlib/tests/test_shmlib.py
python 04_neuralprophet_inclination_prediction/tests/test_prediction.py
python 03_thermomechanical_response/tests/test_shmlib_study03.py
```
Expected: all `OK`.

- [ ] **Step 7: Commit**

```bash
git add studies/shmlib/monitoring.py studies/shmlib/tests/test_monitoring.py studies/shmlib/__init__.py
git commit -m "$(cat <<'EOF'
feat(shmlib): residual control charts, alarm episodes and run lengths

Ported from heritageshm/monitoring.py to the studies' library, with a
standardised-residual interface, robust reference statistics, gap-aware EWMA
updating, and an average run length so that a detector is never reported
without the false-alarm budget it was tuned to.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Anomaly injection and the detectability sweep

**Files:**
- Modify: `studies/shmlib/monitoring.py` (append)
- Test: `studies/shmlib/tests/test_monitoring.py`

**Interfaces:**
- Consumes: `ewma_chart`, `cusum_chart`, `joint_alarm`, `alarm_episodes` from the same module.
- Produces:
  - `inject_anomaly(series, kind, magnitude, start, duration=None, freq='20min') -> pd.Series`
    with `kind` in `{'step', 'ramp', 'pulse'}`
  - `detectability_curve(residuals, mu, sigma, magnitudes, durations, freq='20min', lam=0.2, L=3.0, k=0.5, h=5.0, seed=0) -> pd.DataFrame`
    with columns `magnitude`, `duration_h`, `detected`, `delay_h`

`magnitude` is in the residual's own units — millidegrees — so that the study's headline number is
"the smallest movement this system can find", not a dimensionless multiple of a scale the reader
cannot picture.

- [ ] **Step 1: Write the failing test**

Append to `studies/shmlib/tests/test_monitoring.py`, above the `if __name__` block:

```python
class TestInjectAnomaly(unittest.TestCase):

    def test_a_step_shifts_everything_from_its_start(self):
        series = _quiet(100)
        moved = monitoring.inject_anomaly(
            series, 'step', 5.0, start=series.index[50])
        np.testing.assert_allclose(moved.iloc[:50], series.iloc[:50])
        np.testing.assert_allclose(moved.iloc[50:], series.iloc[50:] + 5.0)

    def test_a_pulse_ends_after_its_duration(self):
        series = _quiet(100)
        moved = monitoring.inject_anomaly(
            series, 'pulse', 5.0, start=series.index[50], duration='2h')
        self.assertAlmostEqual(moved.iloc[50] - series.iloc[50], 5.0)
        self.assertAlmostEqual(moved.iloc[90] - series.iloc[90], 0.0)

    def test_a_ramp_reaches_its_magnitude_at_the_end_of_its_duration(self):
        series = pd.Series(0.0, index=pd.date_range(
            '2024-01-01', periods=100, freq='20min'))
        moved = monitoring.inject_anomaly(
            series, 'ramp', 6.0, start=series.index[10], duration='3h')
        self.assertAlmostEqual(moved.iloc[10], 0.0, places=6)
        self.assertAlmostEqual(moved.iloc[19], 6.0, places=6)
        self.assertAlmostEqual(moved.iloc[50], 6.0, places=6)

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            monitoring.inject_anomaly(
                _quiet(10), 'wobble', 1.0, start=_quiet(10).index[0])


class TestDetectabilityCurve(unittest.TestCase):

    def test_larger_steps_are_detected_and_detected_sooner(self):
        residuals = _quiet(3000)
        curve = monitoring.detectability_curve(
            residuals, 0.0, 1.0, magnitudes=[0.1, 5.0], durations=['24h'])
        small = curve[curve['magnitude'] == 0.1].iloc[0]
        large = curve[curve['magnitude'] == 5.0].iloc[0]
        self.assertFalse(bool(small['detected']))
        self.assertTrue(bool(large['detected']))
        self.assertGreater(large['delay_h'], 0.0)

    def test_one_row_per_magnitude_and_duration_pair(self):
        residuals = _quiet(1500)
        curve = monitoring.detectability_curve(
            residuals, 0.0, 1.0, magnitudes=[1.0, 2.0],
            durations=['6h', '24h'])
        self.assertEqual(len(curve), 4)
        self.assertEqual(list(curve.columns),
                         ['magnitude', 'duration_h', 'detected', 'delay_h'])

    def test_an_undetected_case_reports_a_missing_delay(self):
        residuals = _quiet(1500)
        curve = monitoring.detectability_curve(
            residuals, 0.0, 1.0, magnitudes=[0.01], durations=['6h'])
        self.assertFalse(bool(curve['detected'].iloc[0]))
        self.assertTrue(np.isnan(curve['delay_h'].iloc[0]))
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python shmlib/tests/test_monitoring.py
```
Expected: `AttributeError: module 'shmlib.monitoring' has no attribute 'inject_anomaly'`.

- [ ] **Step 3: Implement**

Append to `studies/shmlib/monitoring.py`:

```python
def inject_anomaly(series, kind, magnitude, start, duration=None,
                   freq='20min'):
    """
    Add a synthetic departure of known size and shape to a series.

    A detector's sensitivity cannot be read off a record that contains one real
    event. Injecting departures of known size is how the question "what is the
    smallest movement this would find" gets a number rather than an opinion.

    Parameters
    ----------
    series : pd.Series
        Signal to contaminate, indexed by timestamp. Not modified in place.
    kind : {'step', 'ramp', 'pulse'}
        ``'step'`` shifts every sample from ``start`` onward; ``'ramp'`` rises
        linearly to ``magnitude`` over ``duration`` and holds it; ``'pulse'``
        shifts only the samples inside ``duration``.
    magnitude : float
        Size of the departure, in the series' own units.
    start : pd.Timestamp or str
        When the departure begins.
    duration : str or pd.Timedelta or None, optional
        Length of the ramp or pulse. Required for those kinds, ignored for a
        step. Default ``None``.
    freq : str, optional
        Spacing of the series, used only when the index carries no frequency.
        Default ``'20min'``.

    Returns
    -------
    pd.Series
        A copy of ``series`` with the departure added.
    """
    if kind not in {'step', 'ramp', 'pulse'}:
        raise ValueError("kind must be 'step', 'ramp' or 'pulse'")
    if kind in {'ramp', 'pulse'} and duration is None:
        raise ValueError(f"kind '{kind}' requires a duration")

    out = series.copy()
    index = pd.DatetimeIndex(out.index)
    begin = pd.Timestamp(start)
    after = index >= begin

    if kind == 'step':
        out.loc[after] = out.loc[after] + magnitude
        return out

    span = pd.Timedelta(duration)
    inside = after & (index < begin + span)

    if kind == 'pulse':
        out.loc[inside] = out.loc[inside] + magnitude
        return out

    elapsed = (index[after] - begin) / span
    profile = np.clip(elapsed, 0.0, 1.0) * magnitude
    out.loc[after] = out.loc[after] + profile
    return out


def detectability_curve(residuals, mu, sigma, magnitudes, durations,
                        freq='20min', lam=0.2, L=3.0, k=0.5, h=5.0, seed=0):
    """
    Whether a departure of each size and length is found, and how late.

    For every pair, a step of that magnitude lasting that long is injected into
    the middle of the residual, both charts are run, and the first joint alarm
    at or after the injection is recorded. The reference statistics are the
    caller's, estimated once on the uncontaminated record, so that the detector
    is never re-tuned to the anomaly it is being asked to find.

    Parameters
    ----------
    residuals : pd.Series
        Uncontaminated residual, indexed by timestamp.
    mu, sigma : float
        Reference centre and scale from ``reference_stats``.
    magnitudes : sequence of float
        Departure sizes in the residual's units.
    durations : sequence of str or pd.Timedelta
        How long each departure persists.
    freq : str, optional
        Spacing of the residual. Default ``'20min'``.
    lam, L : float, optional
        EWMA settings, as in ``ewma_chart``. Defaults ``0.2`` and ``3.0``.
    k, h : float, optional
        CUSUM settings, as in ``cusum_chart``. Defaults ``0.5`` and ``5.0``.
    seed : int, optional
        Reserved for future randomised placement; the injection point is
        currently deterministic. Default ``0``.

    Returns
    -------
    pd.DataFrame
        Columns ``magnitude``, ``duration_h``, ``detected`` and ``delay_h``,
        one row per pair. ``delay_h`` is missing where nothing alarmed.
    """
    values = pd.to_numeric(residuals, errors='coerce')
    index = pd.DatetimeIndex(values.index)
    injection = index[len(index) // 2]

    rows = []
    for magnitude in magnitudes:
        for duration in durations:
            span = pd.Timedelta(duration)
            contaminated = inject_anomaly(
                values, 'pulse', float(magnitude), start=injection,
                duration=span, freq=freq)

            ewma = ewma_chart(contaminated, mu, sigma, lam=lam, L=L)
            cusum = cusum_chart(contaminated, mu, sigma, k=k, h=h)
            alarm = joint_alarm(ewma['alarm'], cusum['alarm'], window=freq)

            fired = alarm.loc[injection:]
            hit = fired[fired].index
            detected = len(hit) > 0
            rows.append({
                'magnitude': float(magnitude),
                'duration_h': float(span / pd.Timedelta(hours=1)),
                'detected': bool(detected),
                'delay_h': (float((hit[0] - injection) / pd.Timedelta(hours=1))
                            if detected else np.nan),
            })
    return pd.DataFrame(
        rows, columns=['magnitude', 'duration_h', 'detected', 'delay_h'])
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python shmlib/tests/test_monitoring.py
```
Expected: `OK`, 20 tests.

- [ ] **Step 5: Verify nothing else broke**

```bash
python shmlib/tests/test_shmlib.py
python 04_neuralprophet_inclination_prediction/tests/test_prediction.py
```
Expected: all `OK`.

- [ ] **Step 6: Commit**

```bash
git add studies/shmlib/monitoring.py studies/shmlib/tests/test_monitoring.py
git commit -m "$(cat <<'EOF'
feat(shmlib): inject known departures and measure what the charts can find

A record holding one real event cannot state a detector's sensitivity. Injecting
departures of known size and length turns "what is the smallest movement this
would find" into a measured curve, at reference statistics estimated on the
uncontaminated record so the detector is never tuned to its own test.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: The remaining figures

**Files:**
- Modify: `studies/shmlib/figures.py` (append)
- Test: `studies/04_neuralprophet_inclination_prediction/tests/test_decomposition.py`

**Interfaces:**
- Consumes: `viz.finish`, `viz.format_spines`, `viz.figsize`, `viz.FIGURE_WIDTH`,
  `viz.INC_COLOUR`, `viz.MARK_COLOUR`, `viz.SPAN_STYLE`, `viz.CHANNEL_COLOUR`.
- Produces:
  - `plot_decomposition_stack(components, columns=None, title='', save_path=None, filename=None)`
  - `plot_prediction_band(observed, expected, lower, upper, title='', highlight=None, save_path=None, filename=None)`
  - `plot_control_chart(chart, statistic='ewma', episodes=None, title='', save_path=None, filename=None)`
  - `plot_metric_vs_horizon(metrics, metric='mae', by='model', title='', save_path=None, filename=None)`
  - `plot_detectability(curve, title='', save_path=None, filename=None)`

- [ ] **Step 1: Write the failing test**

Append to `studies/04_neuralprophet_inclination_prediction/tests/test_decomposition.py`, above the
`if __name__` block:

```python
class TestModelFigures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')

    def test_every_figure_builds_and_writes_png_and_svg(self):
        import tempfile
        from pathlib import Path

        import matplotlib.pyplot as plt

        from shmlib import figures, monitoring

        index = pd.date_range('2024-01-01', periods=400, freq='20min')
        components = pd.DataFrame({
            'trend': np.linspace(0.0, 2.0, 400),
            'season_daily': np.sin(np.arange(400) / 72.0 * 2 * np.pi),
            'future_regressor_tair': np.cos(np.arange(400) / 72.0 * 2 * np.pi),
            'residual': np.zeros(400),
            'y': np.zeros(400),
            'yhat1': np.zeros(400),
        }, index=index)
        observed = pd.Series(np.sin(np.arange(400) / 20.0), index=index)
        expected = observed * 0.9
        chart = monitoring.ewma_chart(observed - expected, 0.0, 1.0)
        metrics = pd.DataFrame({
            'horizon_h': [1, 6, 24, 1, 6, 24],
            'model': ['ar'] * 3 + ['ar+tair'] * 3,
            'mae': [1.0, 2.0, 3.0, 0.9, 1.8, 2.9],
        })
        curve = pd.DataFrame({
            'magnitude': [1.0, 2.0, 1.0, 2.0],
            'duration_h': [6.0, 6.0, 24.0, 24.0],
            'detected': [False, True, True, True],
            'delay_h': [np.nan, 2.0, 1.0, 0.5],
        })

        with tempfile.TemporaryDirectory() as tmp:
            cases = (
                ('NP_F05_decomposition_stack',
                 lambda p, f: figures.plot_decomposition_stack(
                     components, title='t', save_path=p, filename=f)),
                ('NP_F07_observed_vs_expected',
                 lambda p, f: figures.plot_prediction_band(
                     observed, expected, expected - 1.0, expected + 1.0,
                     title='t', save_path=p, filename=f)),
                ('NP_F08_control_chart',
                 lambda p, f: figures.plot_control_chart(
                     chart, title='t', save_path=p, filename=f)),
                ('NP_F10_skill_vs_horizon',
                 lambda p, f: figures.plot_metric_vs_horizon(
                     metrics, metric='mae', by='model', title='t',
                     save_path=p, filename=f)),
                ('NP_F09_detectability',
                 lambda p, f: figures.plot_detectability(
                     curve, title='t', save_path=p, filename=f)),
            )
            for name, call in cases:
                call(tmp, name)
                for ext in ('png', 'svg'):
                    self.assertTrue((Path(tmp) / f'{name}.{ext}').exists(),
                                    f'{name}.{ext} was not written')
            plt.close('all')

    def test_legends_sit_below_their_axes(self):
        import matplotlib.pyplot as plt

        from shmlib import figures

        metrics = pd.DataFrame({
            'horizon_h': [1, 6, 1, 6],
            'model': ['ar', 'ar', 'ar+tair', 'ar+tair'],
            'mae': [1.0, 2.0, 0.9, 1.8],
        })
        fig = figures.plot_metric_vs_horizon(metrics, metric='mae', by='model')
        for ax in fig.axes:
            legend = ax.get_legend()
            if legend is not None:
                self.assertLess(legend.get_bbox_to_anchor().y1, 0.0)
        plt.close(fig)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python 04_neuralprophet_inclination_prediction/tests/test_decomposition.py
```
Expected: `AttributeError: module 'shmlib.figures' has no attribute 'plot_decomposition_stack'`.

- [ ] **Step 3: Implement**

Append to `studies/shmlib/figures.py`:

```python
def plot_decomposition_stack(components, columns=None, title='',
                             save_path=None, filename=None):
    """
    One panel per additive component, on a shared clock.

    The panels are stacked rather than overlaid because the components differ in
    scale by orders of magnitude: a trend of a few millidegrees a year and a
    daily cycle of tens of millidegrees cannot share an axis without one of them
    becoming a flat line.

    Parameters
    ----------
    components : pd.DataFrame
        Output of ``prediction.decompose_components``.
    columns : sequence of str or None, optional
        Components to draw, in panel order. ``None`` draws every component
        column, ending with the residual. Default ``None``.
    title : str, optional
        Figure title. Default ``''``.
    save_path, filename : str or None, optional
        Passed to ``viz.finish``; nothing is written when either is ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if columns is None:
        columns = [c for c in components.columns
                   if c not in ('y', 'yhat1', 'ID', 'residual')]
        if 'residual' in components.columns:
            columns = columns + ['residual']
    columns = list(columns)

    fig, axes = plt.subplots(
        len(columns), 1, sharex=True,
        figsize=viz.figsize(viz.FIGURE_WIDTH, 1.15 * len(columns)))
    axes = np.atleast_1d(axes)

    for ax, column in zip(axes, columns):
        colour = (viz.MARK_COLOUR if column == 'residual' else viz.INC_COLOUR)
        ax.plot(components.index, components[column], color=colour,
                linewidth=1.0)
        ax.set_ylabel(column.replace('future_regressor_', '')
                      .replace('lagged_regressor_', '')
                      .replace('_', ' '))
        viz.format_spines(ax)
    axes[-1].set_xlabel('')

    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_prediction_band(observed, expected, lower, upper, title='',
                         highlight=None, save_path=None, filename=None):
    """
    Observed against expected, with the prediction interval drawn behind them.

    Parameters
    ----------
    observed, expected : pd.Series
        Measured and predicted values, on a shared index.
    lower, upper : pd.Series
        Interval bounds, same index.
    title : str, optional
        Figure title. Default ``''``.
    highlight : sequence of (start, end) or None, optional
        Intervals to shade, drawn in the project's span style. Default ``None``.
    save_path, filename : str or None, optional
        Passed to ``viz.finish``; nothing is written when either is ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=viz.figsize(viz.FIGURE_WIDTH, 2.6))

    ax.fill_between(observed.index, lower, upper, color=viz.INC_COLOUR,
                    alpha=0.18, linewidth=0.0, label='90 % interval')
    ax.plot(observed.index, observed, color=viz.INC_COLOUR, linewidth=1.0,
            label='observed')
    ax.plot(expected.index, expected, color=viz.MARK_COLOUR, linewidth=1.0,
            linestyle='--', label='expected')

    for span in (highlight or ()):
        ax.axvspan(span[0], span[1], **viz.SPAN_STYLE)

    ax.set_ylabel('Inclination [mdeg]')
    viz.format_spines(ax)
    if title:
        ax.set_title(title)
    ax.legend(fontsize='small', ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.30), frameon=False)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_control_chart(chart, statistic='ewma', episodes=None, title='',
                       save_path=None, filename=None):
    """
    A control statistic against its limits, with alarming episodes shaded.

    Parameters
    ----------
    chart : pd.DataFrame
        Output of ``monitoring.ewma_chart`` or ``monitoring.cusum_chart``.
    statistic : str, optional
        Column to draw. Default ``'ewma'``; pass ``'cusum_high'`` for a CUSUM
        chart.
    episodes : pd.DataFrame or None, optional
        Output of ``monitoring.alarm_episodes``, shaded behind the statistic.
        Default ``None``.
    title : str, optional
        Figure title. Default ``''``.
    save_path, filename : str or None, optional
        Passed to ``viz.finish``; nothing is written when either is ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=viz.figsize(viz.FIGURE_WIDTH, 2.4))

    for _, episode in (episodes if episodes is not None
                       else pd.DataFrame()).iterrows():
        ax.axvspan(episode['start'], episode['end'], **viz.SPAN_STYLE)

    ax.plot(chart.index, chart[statistic], color=viz.INC_COLOUR,
            linewidth=1.0, label=statistic.replace('_', ' '))
    for limit in ('ucl', 'lcl', 'limit'):
        if limit in chart.columns:
            ax.plot(chart.index, chart[limit], color=viz.MARK_COLOUR,
                    linewidth=0.9, linestyle='--',
                    label='limit' if limit != 'lcl' else None)

    ax.set_ylabel('Standardised residual')
    viz.format_spines(ax)
    if title:
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize='small', ncol=len(labels),
              loc='upper center', bbox_to_anchor=(0.5, -0.30), frameon=False)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_metric_vs_horizon(metrics, metric='mae', by='model', title='',
                           save_path=None, filename=None):
    """
    One curve per model, showing how a metric degrades with forecast horizon.

    Parameters
    ----------
    metrics : pd.DataFrame
        Long table carrying ``horizon_h``, the grouping column and the metric.
    metric : str, optional
        Column to draw. Default ``'mae'``.
    by : str, optional
        Grouping column, one curve per level. Default ``'model'``.
    title : str, optional
        Figure title. Default ``''``.
    save_path, filename : str or None, optional
        Passed to ``viz.finish``; nothing is written when either is ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=viz.figsize(viz.FIGURE_WIDTH, 2.6))

    styles = ['-', '--', '-.', ':']
    for position, (name, group) in enumerate(metrics.groupby(by, sort=False)):
        ordered = group.sort_values('horizon_h')
        ax.plot(ordered['horizon_h'], ordered[metric], color=viz.INC_COLOUR,
                linestyle=styles[position % len(styles)], marker='o',
                markersize=3, linewidth=1.4, label=str(name))

    ax.set_xlabel('Forecast horizon [h]')
    ax.set_ylabel(metric.upper() if len(metric) <= 4 else metric)
    viz.format_spines(ax)
    if title:
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize='small', ncol=min(len(labels), 4),
              loc='upper center', bbox_to_anchor=(0.5, -0.30), frameon=False)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_detectability(curve, title='', save_path=None, filename=None):
    """
    The smallest departure the charts find, against how long it persists.

    Two panels: whether each injected departure was detected at all, as a
    Cividis field over magnitude and duration, and how late the detection came.

    Parameters
    ----------
    curve : pd.DataFrame
        Output of ``monitoring.detectability_curve``.
    title : str, optional
        Figure title. Default ``''``.
    save_path, filename : str or None, optional
        Passed to ``viz.finish``; nothing is written when either is ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    detected = curve.pivot(index='magnitude', columns='duration_h',
                           values='detected').astype(float)
    delay = curve.pivot(index='magnitude', columns='duration_h',
                        values='delay_h')

    fig, axes = plt.subplots(1, 2, figsize=viz.figsize(viz.FIGURE_WIDTH, 2.6))
    for ax, frame, label, cmap in ((axes[0], detected, 'Detected', 'cividis_r'),
                                   (axes[1], delay, 'Detection delay [h]',
                                    'cividis')):
        mesh = ax.pcolormesh(frame.columns.to_numpy(),
                             frame.index.to_numpy(),
                             frame.to_numpy(), cmap=cmap, shading='nearest')
        fig.colorbar(mesh, ax=ax, label=label)
        ax.set_xlabel('Duration [h]')
        ax.set_ylabel('Magnitude [mdeg]')
        viz.format_spines(ax)

    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python 04_neuralprophet_inclination_prediction/tests/test_decomposition.py
```
Expected: `OK`, 19 tests.

- [ ] **Step 5: Verify nothing else broke**

```bash
python shmlib/tests/test_shmlib.py
python shmlib/tests/test_monitoring.py
python 04_neuralprophet_inclination_prediction/tests/test_gaps.py
python 04_neuralprophet_inclination_prediction/tests/test_prediction.py
python 03_thermomechanical_response/tests/test_shmlib_study03.py
```
Expected: all `OK`.

- [ ] **Step 6: Commit**

```bash
git add studies/shmlib/figures.py \
        studies/04_neuralprophet_inclination_prediction/tests/test_decomposition.py
git commit -m "$(cat <<'EOF'
feat(shmlib): figures for decomposition, intervals, charts, horizons and detectability

Each draws one table produced elsewhere in the library. Components are stacked
rather than overlaid because they differ in scale by orders of magnitude, and
every legend sits below its axes as the project's figure rules require.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

> **CHECKPOINT 3 — show the user:** every test file passing, including Study 03's and Study 04's
> pre-existing `test_prediction.py`, which together certify that the adaptations changed nothing
> for existing callers. Wait for approval before Task 13.

---

### Task 12A: Perturbations with a physical mechanism

The library can inject a step, a ramp and a pulse — generic shapes that say nothing about how a wall
fails. D12 requires three shapes that do, and one helper that converts a timing shift into the
amplitude it implies.

**Files:**
- Modify: `studies/shmlib/monitoring.py` (`inject_anomaly`, `detectability_curve`; append
  `phase_shift_amplitude`)
- Test: `studies/shmlib/tests/test_monitoring.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `inject_anomaly(series, kind, magnitude, start, duration=None, freq='20min', period='24h')`
    — `kind` extended with `'amplitude'`, `'phase'` and `'drift'`; `period` is new and defaults to
    the daily cycle. **Existing kinds keep their behaviour exactly.**
  - `phase_shift_amplitude(daily_amplitude, shift_hours, period_hours=24.0) -> float`
  - `detectability_curve(..., kind='pulse', period='24h')` — two new keyword arguments; the default
    `'pulse'` reproduces today's behaviour, and the returned columns do not change.

The returned column list of `detectability_curve` is asserted verbatim by an existing test, so the
mechanism label belongs to the caller: the notebook tags each sweep with `.assign(kind=kind)`.

- [ ] **Step 1: Write the failing tests**

Append to `studies/shmlib/tests/test_monitoring.py`, above the `if __name__` block:

```python
class TestPhaseShiftAmplitude(unittest.TestCase):

    def test_no_shift_implies_no_residual(self):
        self.assertAlmostEqual(
            monitoring.phase_shift_amplitude(10.0, 0.0), 0.0)

    def test_half_a_period_inverts_the_cycle(self):
        # A cycle shifted by half its period is its own negation, so the
        # difference between shifted and unshifted has twice the amplitude.
        self.assertAlmostEqual(
            monitoring.phase_shift_amplitude(10.0, 12.0), 20.0)

    def test_a_small_shift_follows_the_chord_formula(self):
        expected = 2.0 * 10.0 * np.sin(np.pi * 1.0 / 24.0)
        self.assertAlmostEqual(
            monitoring.phase_shift_amplitude(10.0, 1.0), expected)


class TestPhysicalInjections(unittest.TestCase):

    def _flat(self, n=6 * 24 * 40):
        index = pd.date_range('2024-01-01', periods=n, freq='20min')
        return pd.Series(0.0, index=index)

    def test_an_amplitude_growth_reaches_its_size_and_holds(self):
        series = self._flat()
        start = series.index[100]
        moved = monitoring.inject_anomaly(
            series, 'amplitude', 4.0, start=start, duration='10d')
        np.testing.assert_allclose(moved.iloc[:100], 0.0, atol=1e-12)
        first_day = moved.loc[start:start + pd.Timedelta('1d')]
        last_day = moved.loc[moved.index[-1] - pd.Timedelta('1d'):]
        self.assertLess(first_day.abs().max(), 1.0)
        self.assertAlmostEqual(last_day.abs().max(), 4.0, places=1)

    def test_an_amplitude_growth_adds_no_level(self):
        series = self._flat()
        moved = monitoring.inject_anomaly(
            series, 'amplitude', 4.0, start=series.index[0], duration='1d')
        whole_cycles = moved.loc[:series.index[0] + pd.Timedelta('30d')]
        self.assertAlmostEqual(float(whole_cycles.mean()), 0.0, places=2)

    def test_a_phase_change_is_in_quadrature_with_an_amplitude_growth(self):
        series = self._flat()
        start = series.index[0]
        amplitude = monitoring.inject_anomaly(
            series, 'amplitude', 1.0, start=start, duration='1h')
        phase = monitoring.inject_anomaly(
            series, 'phase', 1.0, start=start, duration='1h')
        window = slice(start + pd.Timedelta('2d'), start + pd.Timedelta('32d'))
        overlap = float((amplitude.loc[window] * phase.loc[window]).mean())
        self.assertAlmostEqual(overlap, 0.0, places=2)

    def test_a_drift_accumulates_at_its_stated_yearly_rate(self):
        series = self._flat()
        start = series.index[0]
        moved = monitoring.inject_anomaly(series, 'drift', 12.0, start=start)
        after_30_days = float(moved.loc[start + pd.Timedelta('30d')])
        self.assertAlmostEqual(after_30_days, 12.0 * 30.0 / 365.25, places=3)

    def test_a_drift_needs_no_duration_and_leaves_the_past_alone(self):
        series = self._flat()
        start = series.index[500]
        moved = monitoring.inject_anomaly(series, 'drift', 5.0, start=start)
        np.testing.assert_allclose(moved.iloc[:500], 0.0, atol=1e-12)
        self.assertGreater(float(moved.iloc[-1]), 0.0)

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            monitoring.inject_anomaly(
                self._flat(), 'settlement', 1.0, start='2024-01-02')

    def test_the_existing_kinds_are_untouched(self):
        series = self._flat(200)
        start = series.index[50]
        step = monitoring.inject_anomaly(series, 'step', 3.0, start=start)
        self.assertAlmostEqual(float(step.iloc[-1]), 3.0)
        self.assertAlmostEqual(float(step.iloc[49]), 0.0)


class TestDetectabilityKinds(unittest.TestCase):

    def test_the_swept_kind_reaches_the_injector(self):
        rng = np.random.default_rng(0)
        index = pd.date_range('2024-01-01', periods=6 * 24 * 40, freq='20min')
        residuals = pd.Series(rng.normal(0.0, 1.0, len(index)), index=index)
        curve = monitoring.detectability_curve(
            residuals, 0.0, 1.0, magnitudes=[0.01, 12.0], durations=['72h'],
            kind='amplitude')
        self.assertEqual(list(curve.columns),
                         ['magnitude', 'duration_h', 'detected', 'delay_h'])
        self.assertFalse(bool(curve['detected'].iloc[0]))
        self.assertTrue(bool(curve['detected'].iloc[1]))
```

- [ ] **Step 2: Run them to verify they fail**

```bash
python shmlib/tests/test_monitoring.py
```
Expected: `AttributeError: module 'shmlib.monitoring' has no attribute 'phase_shift_amplitude'`.

- [ ] **Step 3: Add `phase_shift_amplitude`**

Append to `studies/shmlib/monitoring.py`, after `inject_anomaly`:

```python
def phase_shift_amplitude(daily_amplitude, shift_hours, period_hours=24.0):
    """
    The residual amplitude implied by a timing shift of a periodic response.

    A wall whose thermal path has changed answers the same forcing later or
    earlier without necessarily answering it more strongly. Subtracting the
    unshifted cycle from the shifted one leaves a harmonic in quadrature whose
    amplitude is the chord of the shift, ``2 A sin(pi dt / P)``. This converts
    the quantity an engineer states — a lag change in hours — into the
    millidegree amplitude a detector actually sees.

    Parameters
    ----------
    daily_amplitude : float
        Amplitude of the fitted periodic component, in the series' units. Half
        its peak-to-peak range.
    shift_hours : float
        Timing shift, in hours. Sign is irrelevant: a lead and a lag of the same
        size leave the same amplitude.
    period_hours : float, optional
        Period of the component. Default ``24.0``.

    Returns
    -------
    float
        Amplitude of the residual harmonic, in the series' units.
    """
    return float(2.0 * abs(daily_amplitude)
                 * abs(np.sin(np.pi * float(shift_hours) / float(period_hours))))
```

- [ ] **Step 4: Extend `inject_anomaly`**

Change its signature to
`def inject_anomaly(series, kind, magnitude, start, duration=None, freq='20min', period='24h'):`
and its `kind` and `magnitude` parameter descriptions to cover the three new shapes, documenting
that `'amplitude'` and `'phase'` establish themselves linearly over `duration` and then hold, that
`'phase'` is the quadrature partner of `'amplitude'`, and that `'drift'` takes its magnitude as a
rate per year and needs no `duration`. Add `period` to the `Parameters` section: the cycle the
`'amplitude'` and `'phase'` kinds modulate, default ``'24h'``, ignored by every other kind.

Replace the validation block with:

```python
    kinds = {'step', 'ramp', 'pulse', 'amplitude', 'phase', 'drift'}
    if kind not in kinds:
        raise ValueError(
            "kind must be one of 'step', 'ramp', 'pulse', 'amplitude', "
            "'phase' or 'drift'")
    if kind in {'ramp', 'pulse', 'amplitude', 'phase'} and duration is None:
        raise ValueError(f"kind '{kind}' requires a duration")
```

Insert the drift branch immediately after the existing `'step'` branch returns, because a drift has
no bounded span:

```python
    if kind == 'drift':
        # A rate, not a size: the departure keeps accumulating to the end of the
        # record, which is what creep and settlement do.
        years = ((index[after] - begin)
                 / pd.Timedelta(days=365.25)).to_numpy()
        out.loc[after] = out.loc[after] + magnitude * years
        return out
```

and insert the harmonic branch after the existing `'pulse'` branch returns:

```python
    if kind in {'amplitude', 'phase'}:
        # Both modulate the same cycle and differ only by quadrature: a growing
        # swing is in phase with the response, a timing change is a quarter
        # cycle away from it. The envelope rises linearly over `duration` and
        # then holds, so `magnitude` is the size the departure settles at.
        cycles = ((index[after] - begin) / pd.Timedelta(period)).to_numpy()
        envelope = np.clip(
            ((index[after] - begin) / span).to_numpy(), 0.0, 1.0)
        angle = 2.0 * np.pi * cycles
        wave = np.sin(angle) if kind == 'amplitude' else np.cos(angle)
        out.loc[after] = out.loc[after] + magnitude * envelope * wave
        return out
```

The `'ramp'` tail stays exactly as it is.

- [ ] **Step 5: Thread the kind through `detectability_curve`**

Add `kind='pulse'` and `period='24h'` to its signature, after `seed` and before `response_window`,
document both — stating that the default reproduces the generic pulse and that `'drift'` reads its
magnitude as a rate per year, for which `durations` sets how long the drift is watched rather than
how long it lasts — and pass them to the `inject_anomaly` call:

```python
            contaminated = inject_anomaly(
                values, kind, float(magnitude), start=injection,
                duration=span, freq=freq, period=period)
```

Nothing else in the function changes: the baseline, the attribution rule, the horizon and the
returned columns all stay as Ruling P25 left them.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
python shmlib/tests/test_monitoring.py
```
Expected: `OK`, **36 tests** — the 28 standing plus this task's 8.

- [ ] **Step 7: Verify nothing else broke**

```bash
python shmlib/tests/test_shmlib.py
python 04_neuralprophet_inclination_prediction/tests/test_decomposition.py
python 04_neuralprophet_inclination_prediction/tests/test_prediction.py
python 03_thermomechanical_response/tests/test_shmlib_study03.py
```
Expected: all `OK`. The existing `detectability_curve` tests are what certify that the new `kind`
argument left the pulse sweep exactly as it was.

- [ ] **Step 8: Commit**

```bash
git add studies/shmlib/monitoring.py studies/shmlib/tests/test_monitoring.py
git commit -m "$(cat <<'EOF'
feat(shmlib): injections shaped like masonry damage, not like arithmetic

A step, a ramp and a pulse are shapes a spreadsheet can make. A three-leaf stone
wall fails in ways that leave particular marks on a thermally driven inclination
record, and a sensitivity statement is only physical if the injected departure
has one of those shapes.

Amplitude growth stands for loss of composite action between the leaves: the
section bends further under the same daily heating, so the diurnal swing grows
while its timing and mean hold. A phase change stands for a changed thermal path
rather than a changed stiffness - water in the core, or a crack re-routing
conduction - and appears in quadrature with the cycle. Drift stands for mortar
creep, thermal ratcheting or settlement, and is stated as a rate per year.

Every existing kind, default and returned column is unchanged.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4 · Model A — what the record is made of

### Task 13: Fit the decomposition and confront it with Study 03

**Files:**
- Modify: `studies/04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.py` (append steps 4 and 5)
- Produces: `outputs/NP_05_component_shares.csv`, `NP_06_learned_gains.csv`,
  `NP_08_residual_diagnostics.csv`, `NP_F05_decomposition_stack.{png,svg}`,
  `NP_F06_daily_cycle_and_response.{png,svg}`

**Interfaces:**
- Consumes: `prediction.contiguous_segments`, `prediction.covered_changepoints`,
  `prediction.neuralprophet_backtest`, `prediction.decompose_components`,
  `prediction.component_variance_shares`, `prediction.residual_diagnostics`,
  `figures.plot_decomposition_stack`.
- Produces: notebook variables `model_a`, `predictions_a`, `components_a`, `residual_a`, consumed by
  Tasks 14–16.

- [ ] **Step 0: Confirm step 3 reads no era label (D9)**

In `neuralprophet_inclination_prediction_study.py`, the step 3 call to `prediction.cadence_evidence`
must pass `era=None`, with the reason in the comment above it: instrument eras are Study 01's
subject, and this study reads its input as a single anchored series. Executing the notebook in
Step 6 regenerates `NP_04` accordingly, so **expect `n_change` to be slightly higher and the change
autocorrelations to move in the last decimal** against the values first measured in Phase 1, because
differences at the changeover are no longer discarded. Record the new values; they are the ones the
report quotes.

- [ ] **Step 1: Add the step 4 parameter block to the parameter cell**

```python
# ---------------------------------------------------------------------------
# Model A - decomposition and expectation
# ---------------------------------------------------------------------------
# Zero autoregressive lags. With n_lags > 0 the autoregressive component absorbs
# most of the diurnal structure, and the seasonal component becomes the
# periodicity left over after it rather than the wall's thermal cycle. A
# decomposition meant to be read as physics must therefore carry no AR term.
MODEL_A_LAGS = 0

# Piecewise-linear trend. The drift is the structurally interesting component
# and does not exist under growth='off'.
MODEL_A_GROWTH = 'linear'

# Changepoints are placed at quantiles of the observed timestamps rather than
# uniformly along the axis: this window contains outages of 40, 42, 17 and 103
# days, and a changepoint inside one is constrained by no data.
MODEL_A_CHANGEPOINTS = 12

# Yearly seasonality is fitted both ways and kept only if it improves held-out
# error. The window spans 3.2 annual cycles with a 103-day hole in the last one,
# which is not obviously enough to identify an annual term.
MODEL_A_YEARLY_CANDIDATES = (False, True)

MODEL_A_EPOCHS = 30
MODEL_A_QUANTILES = (0.05, 0.95)
MODEL_A_SEED = 0

# Training origin: the model is fitted on everything before this instant and
# judged on everything after it. Chronological, never random.
MODEL_A_TRAIN_END = '2025-09-01'

# Minimum segment length, in slots. A segment shorter than a day cannot inform
# a daily seasonality, and contributes noise to the trend.
MODEL_A_MIN_SEGMENT = 72
```

- [ ] **Step 2: Add the step 4 Markdown cell**

```markdown
# %% [markdown]
# ## 4 · What the record is made of
#
# The level is decomposed into a trend, a daily cycle, the response to the two
# environmental channels measured beside it, and a remainder. The level is used
# here and nowhere else in this study: its lag-one autocorrelation is 0.998, so
# an error metric computed against it would measure the sampling interval rather
# than the model. What it is good for is the trend, which no differenced target
# can recover — the mean of the gap-safe change implies −65.7 mdeg/yr at twenty
# minutes and +0.97 mdeg/yr at one hour on this same record, because segment
# endpoints do not sample the diurnal cycle uniformly.
#
# Nothing is interpolated. Rows missing any required channel are dropped, the
# remainder is split into contiguous segments, and each segment is handed to
# NeuralProphet under its own identifier, so no fitted window ever spans a gap.
# Imputation is disabled explicitly: with `impute_missing=True`, its default,
# NeuralProphet silently fabricates gaps of at least thirty hours.
#
# ### Parameter Tuning Guidance
#
# **`MODEL_A_LAGS`** — autoregressive lags; must stay `0`. Any positive value
# transfers the diurnal cycle from the seasonal component into the
# autoregressive one and makes the decomposition unreadable as physics.
#
# **`MODEL_A_GROWTH`** — `'linear'` or `'off'`; default `'linear'`. Under
# `'off'` the trend is a constant and the drift disappears.
#
# **`MODEL_A_CHANGEPOINTS`** — number of trend changepoints, placed on covered
# time; default `12`, roughly one per quarter of the window. More changepoints
# track shorter movements at the cost of absorbing signal that belongs to the
# seasonal or regressor terms.
#
# **`MODEL_A_YEARLY_CANDIDATES`** — whether an annual term is fitted; both are
# tried and the comparison is reported in `NP_06`.
#
# **`MODEL_A_TRAIN_END`** — the frozen training origin. Everything after it is
# out of sample. Moving it later buys training data and costs evaluation data.
#
# **`MODEL_A_MIN_SEGMENT`** — shortest usable run, in slots; default `72`, one
# day at twenty minutes.
```

- [ ] **Step 3: Add the step 4 code cell**

```python
# %%
frame_a = (window.rename(columns={TARGET_COLUMN: 'y'})
           .rename(columns={f'{name}_str': name for name in PREDICTOR_COLUMNS})
           .loc[:, ['y'] + list(PREDICTOR_COLUMNS)])

segmented_a = prediction.contiguous_segments(
    frame_a, required=['y'] + list(PREDICTOR_COLUMNS),
    min_length=MODEL_A_MIN_SEGMENT, freq=MODEL_FREQ_A)

train_a = segmented_a.loc[:MODEL_A_TRAIN_END]
test_a = segmented_a.loc[MODEL_A_TRAIN_END:]
changepoints_a = prediction.covered_changepoints(
    train_a.index, MODEL_A_CHANGEPOINTS)

print(f'Model A: {len(train_a):,} training rows in '
      f'{train_a["segment_id"].nunique()} segments, '
      f'{len(test_a):,} evaluation rows')
```

```python
# %%
fits = {}
for yearly in MODEL_A_YEARLY_CANDIDATES:
    model, predictions = prediction.neuralprophet_backtest(
        train_a, test_a, regressors=PREDICTOR_COLUMNS, task='nowcast',
        n_lags=MODEL_A_LAGS, epochs=MODEL_A_EPOCHS, yearly=yearly,
        quantiles=MODEL_A_QUANTILES, seed=MODEL_A_SEED,
        growth=MODEL_A_GROWTH, changepoints=changepoints_a,
        freq=MODEL_FREQ_A)
    fits[yearly] = (model, predictions)
    scores = prediction.score_predictions(predictions, [])
    print(f'yearly={yearly}: out-of-sample MAE {scores["mae"].iloc[0]:.3f} mdeg')

MODEL_A_YEARLY = min(
    fits, key=lambda flag: prediction.score_predictions(
        fits[flag][1], [])['mae'].iloc[0])
model_a, predictions_a = fits[MODEL_A_YEARLY]
print(f'Chosen: yearly={MODEL_A_YEARLY}')
```

- [ ] **Step 4: Add the step 5 Markdown cell**

```markdown
# %% [markdown]
# ## 5 · The components, and whether they agree with Study 03
#
# The fitted air-temperature contribution is the one number in this study that
# can be checked against an independent measurement. Study 03 screened the same
# response against the same channel by a completely different method — a lag and
# gain scan on the diurnal band — and measured **−2.79 mdeg/°C** with
# `r = −0.957`, a value that survived substitution of the ground station
# (−2.23) and ERA5 (−2.04) for the on-structure sensor. If this decomposition
# reproduces it, two unrelated methods agree on a physical constant. If it does
# not, that disagreement is the study's finding and the work stops here rather
# than proceeding to build an anomaly detector on a model that does not describe
# the wall.
#
# The gain is fitted on the compensated channel, which this study takes as its
# raw data. Whether Study 01's compensation is correctly sized is Study 01's
# question, and it is not reopened here: the number below is what the wall does
# after that correction, which is the only quantity a monitoring system ever
# sees.
```

- [ ] **Step 5: Add the step 5 code cell**

```python
# %%
components_a = prediction.decompose_components(
    model_a, segmented_a, regressors=PREDICTOR_COLUMNS)
residual_a = components_a['residual']

shares = prediction.component_variance_shares(components_a)
diagnostics = prediction.residual_diagnostics(residual_a, lags=(1, 72, 216))
display(shares)
display(diagnostics)

shares.to_csv(OUTPUT_DIR / 'NP_05_component_shares.csv', index=False)
diagnostics.to_csv(OUTPUT_DIR / 'NP_08_residual_diagnostics.csv', index=False)

figures.plot_decomposition_stack(
    components_a.loc[MODEL_A_TRAIN_END:],
    title='What the inclination record is made of',
    save_path=str(OUTPUT_DIR), filename='NP_F05_decomposition_stack')
plt.show()
```

```python
# %%
# The learned thermal gain, set against the two independent statements of it.
paired = pd.concat([components_a['future_regressor_tair'],
                    segmented_a['tair']], axis=1).dropna()
paired.columns = ['contribution', 'tair']
learned_gain = np.polyfit(paired['tair'], paired['contribution'], 1)[0]

gains = pd.DataFrame([
    {'source': 'Model A, compensated channel', 'gain_mdeg_per_degC': learned_gain,
     'method': 'NeuralProphet future regressor', 'n': len(paired)},
    {'source': 'Study 03, diurnal band', 'gain_mdeg_per_degC': -2.79,
     'method': 'lag and gain scan', 'n': np.nan},
    {'source': 'Study 03, ground station', 'gain_mdeg_per_degC': -2.23,
     'method': 'lag and gain scan', 'n': np.nan},
    {'source': 'Study 03, ERA5', 'gain_mdeg_per_degC': -2.04,
     'method': 'lag and gain scan', 'n': np.nan},
])
display(gains)
gains.to_csv(OUTPUT_DIR / 'NP_06_learned_gains.csv', index=False)
```

`NP_06` carries the fitted gain and Study 03's three independent measurements of it, and nothing
else. No raw-channel refit and no documented-coefficient row: D10 places compensation outside this
study, and the Study 03 confrontation does not depend on it.

- [ ] **Step 6: Execute and inspect**

```bash
cd studies/04_neuralprophet_inclination_prediction
jupytext --sync neuralprophet_inclination_prediction_study.py
jupyter nbconvert --to notebook --execute --inplace \
  neuralprophet_inclination_prediction_study.ipynb --ExecutePreprocessor.timeout=7200
```

- [ ] **Step 7: Evaluate the checkpoint condition**

```bash
python - <<'PY'
import pandas as pd
out = 'studies/04_neuralprophet_inclination_prediction/outputs'
gains = pd.read_csv(f'{out}/NP_06_learned_gains.csv')
learned = float(gains.loc[gains.source == 'Model A, compensated channel',
                          'gain_mdeg_per_degC'].iloc[0])
print('learned', round(learned, 3), 'study 03', -2.79,
      'ratio', round(learned / -2.79, 3))
print(pd.read_csv(f'{out}/NP_05_component_shares.csv').head())
print(pd.read_csv(f'{out}/NP_08_residual_diagnostics.csv'))
PY
```

Pass condition: the learned gain is negative and within roughly ±30 % of −2.79 mdeg/°C, and the
Ljung–Box p-values either exceed 0.01 or the surviving structure is named. A positive gain, or one
differing by more than a factor of two, **stops the plan here** — see Step 9.

- [ ] **Step 8: If the checkpoint fails, stop and write up the disagreement**

Do not tune the model to reach the expected number. Record in the notebook which of the two
methods the record supports, and bring the disagreement to the user before any further task. Two
methods disagreeing about a physical constant is a result; a model adjusted until it agrees is not.

- [ ] **Step 9: Commit**

```bash
git add studies/04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.py \
        studies/04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.ipynb
git commit -m "$(cat <<'EOF'
feat(study04): decompose the level and confront the fitted gain with study 03

Model A carries no autoregressive term, so its seasonal component is the wall's
daily thermal cycle rather than the periodicity left over after autoregression,
and its piecewise-linear trend on covered changepoints is the only unbiased
estimate of drift available on a record whose segments do not sample the diurnal
cycle uniformly.

The fitted air-temperature contribution is compared against study 03's
independently measured -2.79 mdeg/degC, which is this study's one external
check. The compensation Study 01 applied is taken as given and is not examined
here.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

> **CHECKPOINT 4 — show the user:** `NP_05`, `NP_06`, `NP_08`, `NP_F05`, and the Step 7 output.
> The learned gain must agree with Study 03 within uncertainty, residuals must be white or the
> missing component named, and the interval must be near its nominal coverage. Wait for approval
> before Task 14.

---

### Task 14: The expectation and its calibration

**Files:**
- Modify: the notebook (append step 6)
- Produces: `outputs/NP_07_nowcast_metrics.csv`, `NP_16_component_stability.csv`,
  `NP_F06_daily_cycle_and_response.{png,svg}`, `NP_F07_observed_vs_expected.{png,svg}`

**Interfaces:**
- Consumes: `predictions_a` and `components_a` from Task 13; `prediction.score_predictions`,
  `figures.plot_prediction_band`, `figures.plot_decomposition_stack`.
- Produces: notebook variable `naive_scale_a`, consumed by Task 17 so that MASE is computed
  against one scale throughout.

- [ ] **Step 1: Add the step 6 Markdown cell**

```markdown
# %% [markdown]
# ## 6 · Is this reading the one that was expected?
#
# The model is asked a conditional question, not a predictive one: given the air
# temperature and relative humidity recorded in the same acquisition record as
# this inclination, what inclination did the wall owe? Consuming those two
# channels at the same instant is therefore not leakage — they arrive together,
# twenty minutes at a time, and a monitoring system has them in hand at the
# moment it must judge. The forecast model in step 8 is held to the opposite
# rule and may consume only past predictor values; the two are never scored on
# the same table.
#
# An interval that is drawn but never checked is decoration, so the 90 %
# interval is scored on three counts: how often it actually contains the
# observation, how wide it had to be to manage that, and the pinball loss of
# each quantile on its own terms. Coverage alone can always be bought by
# widening.
#
# ### Parameter Tuning Guidance
#
# **`MODEL_A_QUANTILES`** — the interval's quantiles; default `(0.05, 0.95)`,
# a nominal 90 % interval. Narrower quantiles alarm more often at a fixed
# threshold and shorten detection delay at the cost of false alarms.
#
# **`NOWCAST_INTERVAL_ALPHA`** — nominal miss rate used by the Winkler interval
# score; default `0.10`, matching the quantiles above. Change both together or
# the score is no longer comparable with the coverage beside it.
```

- [ ] **Step 2: Add `NOWCAST_INTERVAL_ALPHA` to the parameter cell**

```python
NOWCAST_INTERVAL_ALPHA = 0.10
```

- [ ] **Step 3: Add the step 6 code cell**

```python
# %%
# The naive scale for MASE is the in-sample mean absolute one-step change. It is
# computed on the training rows only: a scale taken from the evaluation rows
# would make the metric self-referential.
naive_scale_a = float(
    prediction.hourly_change(train_a['y'], freq=MODEL_FREQ_A).abs().mean())

nowcast_scores = prediction.score_predictions(
    predictions_a, [], naive_scale=naive_scale_a,
    alpha=NOWCAST_INTERVAL_ALPHA)
nowcast_scores.insert(0, 'model', 'Model A · tair + rh')
display(nowcast_scores)
nowcast_scores.to_csv(OUTPUT_DIR / 'NP_07_nowcast_metrics.csv', index=False)

print(f'Nominal coverage {1 - NOWCAST_INTERVAL_ALPHA:.0%}, '
      f'observed {nowcast_scores["coverage_q05_q95"].iloc[0]:.1%}, '
      f'median width {nowcast_scores["width_q05_q95"].iloc[0]:.2f} mdeg')
```

```python
# %%
# One month of the evaluation period, drawn at readable density.
view = predictions_a.set_index('ds').sort_index().loc['2025-11-01':'2025-12-01']
figures.plot_prediction_band(
    view['y'], view['yhat'], view['q05'], view['q95'],
    title='Observed inclination against what the measured environment predicted',
    save_path=str(OUTPUT_DIR), filename='NP_F07_observed_vs_expected')

# The two interpretable components on their own axes: the daily cycle the model
# fitted, and the response it attributes to air temperature.
figures.plot_decomposition_stack(
    components_a.loc['2025-11-01':'2025-11-15'],
    columns=['season_daily', 'future_regressor_tair', 'future_regressor_rh'],
    title='The daily cycle and the response to the measured environment',
    save_path=str(OUTPUT_DIR), filename='NP_F06_daily_cycle_and_response')
plt.show()
```

- [ ] **Step 4: Add the component-stability cell**

A component estimated once is an assertion; one that survives being re-estimated on disjoint
stretches of the record is a finding. The model is refitted on each chronological third of the
training period and the two interpretable quantities — the trend slope and the air-temperature
gain — are compared across them. A gain that changes sign between thirds, or a slope that varies
by more than its own magnitude, is reported as instability rather than averaged away.

```python
# %%
stability_rows = []
thirds = np.array_split(train_a.index.unique(), 3)
for number, block in enumerate(thirds, 1):
    part = train_a.loc[block[0]:block[-1]]
    if part['segment_id'].nunique() < 2:
        continue
    model_part, _ = prediction.neuralprophet_backtest(
        part, part, regressors=PREDICTOR_COLUMNS, task='nowcast',
        n_lags=MODEL_A_LAGS, epochs=MODEL_A_EPOCHS, yearly=MODEL_A_YEARLY,
        quantiles=(), seed=MODEL_A_SEED, growth=MODEL_A_GROWTH,
        changepoints=prediction.covered_changepoints(
            part.index, max(2, MODEL_A_CHANGEPOINTS // 3)),
        freq=MODEL_FREQ_A)
    parts = prediction.decompose_components(
        model_part, part, regressors=PREDICTOR_COLUMNS)
    paired_part = pd.concat(
        [parts['future_regressor_tair'], part['tair']], axis=1).dropna()
    paired_part.columns = ['contribution', 'tair']
    elapsed_days = ((parts.index - parts.index[0])
                    / pd.Timedelta(days=1)).to_numpy()
    stability_rows.append({
        'block': f'third {number}',
        'start': part.index.min(),
        'end': part.index.max(),
        'tair_gain_mdeg_per_degC': float(
            np.polyfit(paired_part['tair'], paired_part['contribution'], 1)[0]),
        'trend_mdeg_per_year': float(
            np.polyfit(elapsed_days, parts['trend'].to_numpy(), 1)[0] * 365.0),
    })

stability = pd.DataFrame(stability_rows)
display(stability)
stability.to_csv(OUTPUT_DIR / 'NP_16_component_stability.csv', index=False)

signs = np.sign(stability['tair_gain_mdeg_per_degC'])
print('Thermal gain keeps its sign across thirds:', bool(signs.nunique() == 1))
```

- [ ] **Step 5: Execute and verify**

```bash
cd studies/04_neuralprophet_inclination_prediction
jupytext --sync neuralprophet_inclination_prediction_study.py
jupyter nbconvert --to notebook --execute --inplace \
  neuralprophet_inclination_prediction_study.ipynb --ExecutePreprocessor.timeout=7200
python - <<'PY'
import pandas as pd
print(pd.read_csv('outputs/NP_16_component_stability.csv').to_string(index=False))
scores = pd.read_csv('outputs/NP_07_nowcast_metrics.csv')
print(scores[['mae', 'rmse', 'bias', 'mase', 'coverage_q05_q95',
              'width_q05_q95', 'interval_score']].to_string(index=False))
PY
```

Expected: `coverage_q05_q95` near 0.90. Coverage far below nominal means the interval is
overconfident and the anomaly detector built on it in Task 15 would alarm constantly; coverage far
above means it is uninformative. Record the observed value either way — it is reported in the
study, not corrected away.

- [ ] **Step 6: Commit**

```bash
git add studies/04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.py \
        studies/04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.ipynb
git commit -m "$(cat <<'EOF'
feat(study04): score the expectation and check that its interval means what it says

Point accuracy, scale-free error against an in-sample naive scale, and three
separate statements about the interval: how often it contains the observation,
how wide it had to be, and the pinball loss of each quantile.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5 · Judging a departure

### Task 15: Control charts tuned to a stated false-alarm budget

**Files:**
- Modify: the notebook (append step 7)
- Produces: `outputs/NP_09_alarm_episodes.csv`, `NP_F08_control_chart.{png,svg}`

**Interfaces:**
- Consumes: `residual_a` from Task 13; `monitoring.reference_stats`, `monitoring.ewma_chart`,
  `monitoring.cusum_chart`, `monitoring.joint_alarm`, `monitoring.alarm_episodes`,
  `monitoring.average_run_length`, `figures.plot_control_chart`.
- Produces: notebook variables `reference_a`, `alarm_a`, consumed by Task 16.

- [ ] **Step 1: Add the monitoring parameters to the parameter cell**

```python
# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
# The reference window. Every limit drawn is a multiple of the centre and scale
# estimated here, so a window containing a departure would calibrate the
# detector against the very thing it is meant to find. This one ends well before
# the stretch Study 01 flagged in summer 2026 - excluded as a precaution, since
# a reference window must be in control. This study makes no detection claim
# about that stretch (spec section 11.3).
REFERENCE_START = '2023-06-21'
REFERENCE_END = '2025-06-01'

# EWMA smoothing and limit width. Lambda smaller reacts more slowly and finds
# smaller sustained shifts; L wider means fewer false alarms and later
# detection. Both are swept in step 7 and the pair meeting the false-alarm
# budget is the one reported.
EWMA_LAMBDA = 0.05
EWMA_L = 3.0

# CUSUM slack and decision interval, in standard deviations.
CUSUM_K = 0.5
CUSUM_H = 5.0

# Coincidence window for the joint alarm.
JOINT_WINDOW = '6h'

# No event window is parameterised. The summer-2026 stretch is kept out of the
# reference window above, but it is not a test and no result is stated from it
# (D12).

# The false-alarm budget the charts are tuned to, in days between false alarms
# on the in-control reference stretch. Every detection figure in this study is
# only comparable at a stated run length, and this is it.
TARGET_ARL_DAYS = 90.0

```

- [ ] **Step 2: Add the step 7 Markdown cell**

```markdown
# %% [markdown]
# ## 7 · Judging a departure
#
# The residual of step 6 is what remains after the trend, the daily cycle and
# the measured environment have been accounted for. A departure is a stretch
# where that remainder stops behaving as it did over the reference window.
#
# Two charts run together because they fail in different ways. The exponentially
# weighted average answers "has the level moved and stayed moved", which is the
# shape a structural departure takes; the cumulative sum accumulates evidence
# and finds a shift too small to breach a limit on any single sample, provided
# it persists. An alarm is raised only where both agree within a coincidence
# window, which trades a little sensitivity for a large reduction in isolated
# false alarms.
#
# A detector can be made to look perfect by never alarming, so the settings are
# fixed the other way round: the limit width is chosen to meet a stated
# false-alarm budget on the in-control stretch, and every detection figure that
# follows is quoted at that budget.
#
# ### Parameter Tuning Guidance
#
# **`REFERENCE_START`, `REFERENCE_END`** — the in-control window. Must exclude
# any stretch suspected of carrying a departure; on this record it ends before
# the one Study 01 flagged in summer 2026. That exclusion is a precaution about
# calibration, not a claim that the detector finds it.
#
# **`TARGET_ARL_DAYS`** — days of watched time per false alarm; default `90`.
# Lower it and the system finds smaller movements sooner while crying wolf more
# often. This is the operator's dial, and it is the axis every other detection
# number is quoted against.
#
# **`EWMA_LAMBDA`** — smoothing weight; default `0.05`, tuned for sustained
# shifts rather than spikes. **`EWMA_L`** is swept to meet the budget rather
# than set by hand.
#
# **`CUSUM_K`, `CUSUM_H`** — slack and decision interval, in standard
# deviations; defaults `0.5` and `5.0`, the textbook pair for detecting a
# one-sigma shift quickly.
#
# **`JOINT_WINDOW`** — how close in time the two charts must agree; default
# `'6h'`. Wider admits more coincidences and raises the false-alarm rate.
```

- [ ] **Step 3: Add the step 7 code cell**

```python
# %%
reference_a = monitoring.reference_stats(
    residual_a, start=REFERENCE_START, end=REFERENCE_END, robust=True)
print(f'Reference: mu={reference_a["mu"]:.3f} mdeg, '
      f'sigma={reference_a["sigma"]:.3f} mdeg, n={reference_a["n"]:,}')

# Choose the limit width that meets the false-alarm budget on the in-control
# stretch, rather than accepting a conventional value and reporting whatever
# rate it happens to give.
in_control = residual_a.loc[REFERENCE_START:REFERENCE_END]
budget = []
for candidate_L in np.arange(2.0, 5.01, 0.25):
    ewma = monitoring.ewma_chart(in_control, reference_a['mu'],
                                 reference_a['sigma'],
                                 lam=EWMA_LAMBDA, L=candidate_L)
    cusum = monitoring.cusum_chart(in_control, reference_a['mu'],
                                   reference_a['sigma'],
                                   k=CUSUM_K, h=CUSUM_H)
    joint = monitoring.joint_alarm(ewma['alarm'], cusum['alarm'],
                                   window=JOINT_WINDOW)
    arl = monitoring.average_run_length(joint, freq=MODEL_FREQ_A)
    budget.append({'L': candidate_L, **arl})

budget = pd.DataFrame(budget)
display(budget)
EWMA_L = float(budget.loc[budget['arl_days'] >= TARGET_ARL_DAYS, 'L'].min())
print(f'Chosen L = {EWMA_L}, meeting {TARGET_ARL_DAYS:.0f} days per false alarm')
```

```python
# %%
ewma_a = monitoring.ewma_chart(residual_a, reference_a['mu'],
                               reference_a['sigma'],
                               lam=EWMA_LAMBDA, L=EWMA_L)
cusum_a = monitoring.cusum_chart(residual_a, reference_a['mu'],
                                 reference_a['sigma'], k=CUSUM_K, h=CUSUM_H)
alarm_a = monitoring.joint_alarm(ewma_a['alarm'], cusum_a['alarm'],
                                 window=JOINT_WINDOW)
episodes_a = monitoring.alarm_episodes(alarm_a, ewma_a['z'])
display(episodes_a)

episodes_a.to_csv(OUTPUT_DIR / 'NP_09_alarm_episodes.csv', index=False)
figures.plot_control_chart(
    ewma_a, statistic='ewma', episodes=episodes_a,
    title='Residual control chart, with alarming episodes shaded',
    save_path=str(OUTPUT_DIR), filename='NP_F08_control_chart')
plt.show()
```

- [ ] **Step 4: Execute and verify the budget was met**

```bash
cd studies/04_neuralprophet_inclination_prediction
jupytext --sync neuralprophet_inclination_prediction_study.py
jupyter nbconvert --to notebook --execute --inplace \
  neuralprophet_inclination_prediction_study.ipynb --ExecutePreprocessor.timeout=7200
python - <<'PY'
import pandas as pd
episodes = pd.read_csv('outputs/NP_09_alarm_episodes.csv', parse_dates=['start', 'end'])
print(f'{len(episodes)} episodes')
print(episodes.head(10).to_string(index=False))
PY
```

Expected: a run length at or above `TARGET_ARL_DAYS` on the reference stretch. If no candidate `L`
meets the budget, the residual is not in control over the reference window — return to Checkpoint 4
rather than widening the limit until the alarms stop. The episode table is reported as it comes;
**no episode is counted against the summer-2026 stretch and no verdict is stated about it** (D12),
because a detector's sensitivity is established in Task 16 by injection, not by one large event.

- [ ] **Step 5: Commit**

```bash
git add studies/04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.py \
        studies/04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.ipynb
git commit -m "$(cat <<'EOF'
feat(study04): tune the residual charts to a stated false-alarm budget

The limit width is chosen to deliver a target number of days between false
alarms on the in-control stretch, rather than taken from convention and reported
with whatever rate it happens to produce. Every detection figure that follows is
quoted at that budget.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 16: Detectability against three damage mechanisms

The sweep runs over perturbations whose shape corresponds to a damage mechanism in a three-leaf
stone wall (D12). The summer-2026 event is not tested: it is too large to demonstrate sensitivity,
and finding it would prove only that the detector is not broken.

**Files:**
- Modify: the notebook (append step 7b)
- Produces: `outputs/NP_10_detectability.csv`, `NP_F09_detectability.{png,svg}`

**Interfaces:**
- Consumes: `residual_a`, `reference_a`, `components_a`; `monitoring.detectability_curve`,
  `monitoring.phase_shift_amplitude`, `figures.plot_detectability`. `alarm_a` is no longer consumed.
- Produces: nothing consumed later; this task's output is a study result.

- [ ] **Step 1: Add the sweep parameters to the parameter cell**

```python
# Injected departures, in millidegrees and hours. The magnitudes bracket the
# residual's own scale so that the curve crosses from undetectable to certain
# inside the swept range; the durations span a working day to a fortnight.
DETECT_MAGNITUDES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
DETECT_DURATIONS = ('6h', '24h', '72h', '168h', '336h')

# The three mechanisms swept, each a shape a three-leaf wall can produce:
#   'amplitude' - the daily swing grows while its timing and mean hold, which is
#                 what loss of composite action between the leaves looks like;
#   'phase'     - the response arrives earlier or later against the same
#                 forcing, which is a change in the thermal path rather than in
#                 stiffness, such as water in the core;
#   'drift'     - a slow monotone accumulation, the shape of mortar creep,
#                 thermal ratcheting or settlement.
DETECT_KINDS = ('amplitude', 'phase', 'drift')

# Timing shifts probed for the phase mechanism, in hours. Each is converted into
# the residual amplitude it implies through the fitted daily amplitude, so the
# result is quoted as the shift an engineer would picture rather than as a
# millidegree figure with no mechanism attached.
DETECT_PHASE_SHIFTS_H = (0.25, 0.5, 1.0, 2.0)

# Drift rates probed, in millidegrees per year.
DETECT_DRIFT_RATES = (1.0, 2.0, 5.0, 10.0, 20.0)

# How long after a departure ends an alarm still counts as having found it.
DETECT_RESPONSE_WINDOW = '24h'
```

- [ ] **Step 2: Add the step 7b Markdown cell**

```markdown
# %% [markdown]
# ### 7b · Which damage signatures would be found, and how late
#
# One large event cannot state a detector's sensitivity. Departures of known
# size, length and *shape* are injected into the residual instead, the charts
# are re-run with the reference statistics estimated on the uncontaminated
# record, and an alarm counts only where the uncontaminated run is silent. What
# comes out is the study's headline operational number: which movements this
# system finds, and how long each has to persist before it does.
#
# The three shapes are not arbitrary. A three-leaf stone wall — two masonry
# leaves either side of a weaker rubble-and-mortar core — fails in ways that
# leave distinguishable marks on a thermally driven inclination record:
#
# * **Amplitude growth.** The leaves stop acting together: delamination at the
#   core interface, or loss of through-stones. The section bends further under
#   the same daily heating, so the diurnal swing grows while its timing and its
#   mean stay put.
# * **Phase change.** The thermal path changes rather than the stiffness — water
#   entering the core raises its heat capacity, or a crack re-routes conduction.
#   The wall answers the same forcing later or earlier, which appears in the
#   residual as a harmonic in quadrature with the daily cycle. It is quoted as
#   the timing shift in hours, converted through the daily amplitude this
#   decomposition already measured.
# * **Drift.** Creep of the lime mortar under sustained load, thermal ratcheting
#   of the outer leaf, or foundation settlement: slow, monotone, invisible in any
#   single day, and quoted in millidegrees per year.
#
# The summer-2026 stretch Study 01 flagged is kept out of the reference window,
# but no claim is made about whether this detector finds it. Its size makes it
# uninformative about sensitivity, which is what this sweep exists to measure.
```

- [ ] **Step 3: Add the step 7b code cell**

```python
# %%
# The daily amplitude this decomposition fitted, which converts a timing shift
# into the residual amplitude it implies.
daily_amplitude = float(
    shares.set_index('component').loc['season_daily', 'peak_to_peak'] / 2.0)
phase_magnitudes = tuple(
    monitoring.phase_shift_amplitude(daily_amplitude, hours)
    for hours in DETECT_PHASE_SHIFTS_H)
print(f'Daily amplitude {daily_amplitude:.2f} mdeg; a timing shift of '
      f'{DETECT_PHASE_SHIFTS_H[0]} h to {DETECT_PHASE_SHIFTS_H[-1]} h implies '
      f'{phase_magnitudes[0]:.2f} to {phase_magnitudes[-1]:.2f} mdeg')

in_control = residual_a.loc[REFERENCE_START:REFERENCE_END]
sweeps = {
    'amplitude': DETECT_MAGNITUDES,
    'phase': phase_magnitudes,
    'drift': DETECT_DRIFT_RATES,
}

curves = []
for kind in DETECT_KINDS:
    curve = monitoring.detectability_curve(
        in_control, reference_a['mu'], reference_a['sigma'],
        magnitudes=sweeps[kind], durations=DETECT_DURATIONS, kind=kind,
        freq=MODEL_FREQ_A, lam=EWMA_LAMBDA, L=EWMA_L, k=CUSUM_K, h=CUSUM_H,
        response_window=DETECT_RESPONSE_WINDOW)
    curves.append(curve.assign(kind=kind))

detectability = pd.concat(curves, ignore_index=True)
display(detectability)
detectability.to_csv(OUTPUT_DIR / 'NP_10_detectability.csv', index=False)
```

```python
# %%
for kind in DETECT_KINDS:
    figures.plot_detectability(
        detectability[detectability['kind'] == kind],
        title=f'Detectability of a {kind} departure',
        save_path=str(OUTPUT_DIR), filename=f'NP_F09_detectability_{kind}')
    plt.show()

smallest = (detectability[detectability['detected']]
            .groupby(['kind', 'duration_h'])['magnitude'].min())
print('Smallest departure found, by mechanism and persistence:')
print(smallest.to_string())
```

- [ ] **Step 4: Execute and verify**

```bash
cd studies/04_neuralprophet_inclination_prediction
jupytext --sync neuralprophet_inclination_prediction_study.py
jupyter nbconvert --to notebook --execute --inplace \
  neuralprophet_inclination_prediction_study.ipynb --ExecutePreprocessor.timeout=7200
python - <<'PY'
import pandas as pd
curve = pd.read_csv('outputs/NP_10_detectability.csv')
for kind, block in curve.groupby('kind'):
    print(kind)
    print(block.pivot(index='magnitude', columns='duration_h',
                      values='detected').to_string())
PY
```

Expected: a monotone field within each mechanism — larger and longer departures detected, smaller
and shorter not. A non-monotone field means the injection point interacts with a real feature of the
residual; move the injection or sweep several points and report the spread. The three mechanisms are
**not** expected to agree with one another: a drift of a few millidegrees per year is a far smaller
instantaneous departure than an amplitude growth of the same figure, and the difference between them
is the result.

- [ ] **Step 5: Commit**

```bash
git add studies/04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.py \
        studies/04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.ipynb
git commit -m "$(cat <<'EOF'
feat(study04): measure detectability against three damage mechanisms

Injected departures of known magnitude, duration and shape turn the detector's
sensitivity into a curve per mechanism: growth of the daily swing, which is what
loss of composite action between the leaves produces; a timing shift, which is a
change in the thermal path rather than in stiffness; and a slow drift, the shape
of mortar creep or settlement.

One large event cannot state a sensitivity, so the stretch study 01 flagged in
summer 2026 is kept out of the reference window and no result is claimed from
it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

> **CHECKPOINT 5 — show the user:** the chosen `L` and its run length in days, `NP_09`, `NP_10`,
> `NP_F08`, the three `NP_F09_detectability_*` figures, and the smallest departure found per
> mechanism and persistence. A mechanism that stays undetected across the whole swept range is
> reported as such, not tuned away — it is a statement about what this instrument and this model
> cannot see. Wait for approval before Task 17.

---

## Phase 6 · Model B — how far ahead is prediction worth anything

### Task 17: The ablation ladder against three baselines

**Files:**
- Modify: the notebook (append step 8)
- Produces: `outputs/NP_11_forecast_metrics.csv`, `NP_12_skill_vs_baseline.csv`,
  `NP_13_ablation.csv`, `NP_F10_skill_vs_horizon.{png,svg}`, `NP_F11_ablation.{png,svg}`

**Interfaces:**
- Consumes: `window`, `naive_scale_a`; `prediction.hourly_change`,
  `prediction.contiguous_segments`, `prediction.expanding_segment_folds`,
  `prediction.execution_folds`, `prediction.backtest_specifications`,
  `prediction.baseline_predictions`, `prediction.score_predictions`,
  `prediction.paired_mae_skill`, `figures.plot_metric_vs_horizon`.
- Produces: notebook variable `predictions_b`, consumed by Task 18.

- [ ] **Step 1: Add the Model B parameters to the parameter cell**

```python
# ---------------------------------------------------------------------------
# Model B - forecast
# ---------------------------------------------------------------------------
# Hourly, because the twenty-minute first difference is noise-dominated:
# measured on this record its lag-one autocorrelation is -0.058, the signature
# of additive noise on the level, while the hourly difference retains +0.349.
# Step 3 exports the comparison as NP_04.
MODEL_B_TARGET = 'change'

# Autoregressive window, chosen from NP_03 rather than by habit: it is the
# longest window that leaves enough training windows on this record's
# contiguity. Read the table before changing it.
MODEL_B_LAGS = 24

# Forecast length and the horizons kept from it.
MODEL_B_FORECASTS = 48
MODEL_B_HORIZONS = (1, 3, 6, 12, 24, 48)

# Predictor history. Past values only: a forecast that consumed a future
# observed air temperature would be answering a different question.
MODEL_B_REGRESSOR_LAGS = 12

# The ablation ladder. Each rung adds one thing, so the increment it buys is
# attributable. The battery control is not a silent channel - study 03 measured
# r = -0.673 for it in the diurnal band - so it marks a conservative floor
# rather than a zero, and a driver is credited only when it clears that floor.
MODEL_B_SPECIFICATIONS = {
    'AR only': (),
    'AR + tair': ('tair',),
    'AR + tair + rh': ('tair', 'rh'),
    'AR + batt (control)': ('batt',),
}

MODEL_B_EPOCHS = 30
MODEL_B_QUANTILES = (0.05, 0.95)
MODEL_B_SEED = 0
MODEL_B_INITIAL_SEGMENTS = 20
MODEL_B_FOLDS = 5
MODEL_B_MIN_SEGMENT = 96
MODEL_B_REFIT_EACH_FOLD = False

# Block bootstrap for the paired skill comparison. Residuals are autocorrelated,
# so an unpaired comparison would overstate significance; the block length is
# adapted to the horizon inside paired_mae_skill.
BOOTSTRAP_BLOCK_HOURS = 24
BOOTSTRAP_REPETITIONS = 2000
```

- [ ] **Step 2: Add the step 8 Markdown cell**

```markdown
# %% [markdown]
# ## 8 · How far ahead is prediction worth anything?
#
# The target changes here, and so does the grid. The response is the gap-safe
# one-hour change: formed only between adjacent accepted observations inside one
# instrument era, so it never bridges a gap or the 2025 changeover. The level is
# not forecast, because its lag-one autocorrelation of 0.998 makes any error
# metric computed against it a measurement of the sampling interval.
#
# An absolute error at a given horizon answers nothing on its own, so every
# model is scored against three baselines a monitoring system could run for
# free: predicting no change at all, persisting the last change, and repeating
# the change from the same hour yesterday. Skill is the fractional reduction in
# mean absolute error against each, computed **paired** and with a block
# bootstrap, because residuals are autocorrelated and an unpaired comparison
# would report significance that is not there. A horizon counts as skilful only
# where the bootstrap interval excludes zero.
#
# The ladder exists because a model carrying both autoregressive memory and air
# temperature cannot say which of the two earned its skill. Each rung adds one
# thing; the increment is what that thing bought.
#
# ### Parameter Tuning Guidance
#
# **`MODEL_B_LAGS`** — autoregressive window in hours; default `24`, read off
# `NP_03`. Raising it costs training windows disproportionately on a fragmented
# record, because a segment shorter than `lags + forecasts` contributes nothing.
#
# **`MODEL_B_HORIZONS`** — horizons kept from the model output; default
# `(1, 3, 6, 12, 24, 48)`. Each must not exceed `MODEL_B_FORECASTS`.
#
# **`MODEL_B_REGRESSOR_LAGS`** — hours of predictor history; default `12`,
# inside study 03's admissible range for an external forcing. Study 03 found no
# level-band time constant that was not an artefact of the scan boundary, so no
# longer memory is justified.
#
# **`MODEL_B_SPECIFICATIONS`** — the ladder. Keep `'AR only'` first: without it
# nothing in this study is attributable.
#
# **`BOOTSTRAP_REPETITIONS`** — bootstrap draws; default `2000`. Lower it only
# for a smoke run; the reported intervals need the full count.
```

- [ ] **Step 3: Add the step 8 code cell — build the hourly change target**

```python
# %%
hourly = window.resample(MODEL_FREQ_B).mean(numeric_only=True)
hourly['era'] = window['era'].resample(MODEL_FREQ_B).first() \
    if 'era' in window.columns else None

frame_b = pd.DataFrame({
    'y': prediction.hourly_change(
        hourly[TARGET_COLUMN], era=None, freq=MODEL_FREQ_B),
    'tair': hourly['tair_str'],
    'rh': hourly['rh_str'],
    'batt': hourly['batt_str'],
})

segmented_b = prediction.contiguous_segments(
    frame_b, required=['y', 'tair', 'rh'],
    min_length=MODEL_B_MIN_SEGMENT, freq=MODEL_FREQ_B)
folds_b = prediction.expanding_segment_folds(
    segmented_b['segment_id'], MODEL_B_INITIAL_SEGMENTS, MODEL_B_FOLDS)
execution_b = prediction.execution_folds(
    folds_b, refit_each_fold=MODEL_B_REFIT_EACH_FOLD)

print(f'Model B: {len(segmented_b):,} rows in '
      f'{segmented_b["segment_id"].nunique()} segments, '
      f'{len(folds_b)} folds')
```

- [ ] **Step 4: Add the backtest and baseline cell**

```python
# %%
predictions_b = prediction.backtest_specifications(
    segmented_b, execution_b, MODEL_B_SPECIFICATIONS,
    epochs=MODEL_B_EPOCHS, quantiles=MODEL_B_QUANTILES, seed=MODEL_B_SEED)
baselines_b = prediction.baseline_predictions(
    segmented_b, execution_b, MODEL_B_HORIZONS, task='forecast')
all_b = pd.concat([predictions_b, baselines_b], ignore_index=True)

naive_scale_b = float(segmented_b['y'].abs().mean())
metrics_b = prediction.score_predictions(
    all_b, ['model', 'horizon_h'], naive_scale=naive_scale_b,
    alpha=NOWCAST_INTERVAL_ALPHA)
display(metrics_b)

metrics_b.to_csv(OUTPUT_DIR / 'NP_11_forecast_metrics.csv', index=False)
figures.plot_metric_vs_horizon(
    metrics_b, metric='mae', by='model',
    title='Forecast error against horizon, by model and baseline',
    save_path=str(OUTPUT_DIR), filename='NP_F10_skill_vs_horizon')
plt.show()
```

- [ ] **Step 5: Add the paired-skill cell**

```python
# %%
skill_rows = []
for baseline in ('zero', 'persistence', 'seasonal_naive'):
    for model in MODEL_B_SPECIFICATIONS:
        for horizon in MODEL_B_HORIZONS:
            parent = all_b[(all_b['model'] == baseline)
                           & (all_b['horizon_h'] == horizon)]
            child = all_b[(all_b['model'] == model)
                          & (all_b['horizon_h'] == horizon)]
            if parent.empty or child.empty:
                continue
            result = prediction.paired_mae_skill(
                parent, child, block_hours=BOOTSTRAP_BLOCK_HOURS,
                repetitions=BOOTSTRAP_REPETITIONS, seed=MODEL_B_SEED,
                horizon_hours=horizon)
            skill_rows.append({'baseline': baseline, 'model': model,
                               'horizon_h': horizon, **result})

skill = pd.DataFrame(skill_rows)
display(skill)
skill.to_csv(OUTPUT_DIR / 'NP_12_skill_vs_baseline.csv', index=False)
```

- [ ] **Step 6: Add the ablation cell**

```python
# %%
# What each rung of the ladder bought, as the increment over the rung below it.
ladder = list(MODEL_B_SPECIFICATIONS)
ablation_rows = []
for lower, upper in zip(ladder[:-1], ladder[1:]):
    if upper.endswith('(control)'):
        continue
    for horizon in MODEL_B_HORIZONS:
        parent = all_b[(all_b['model'] == lower)
                       & (all_b['horizon_h'] == horizon)]
        child = all_b[(all_b['model'] == upper)
                      & (all_b['horizon_h'] == horizon)]
        if parent.empty or child.empty:
            continue
        result = prediction.paired_mae_skill(
            parent, child, block_hours=BOOTSTRAP_BLOCK_HOURS,
            repetitions=BOOTSTRAP_REPETITIONS, seed=MODEL_B_SEED,
            horizon_hours=horizon)
        ablation_rows.append({'added_over': lower, 'model': upper,
                              'horizon_h': horizon, **result})

ablation = pd.DataFrame(ablation_rows)
display(ablation)
ablation.to_csv(OUTPUT_DIR / 'NP_13_ablation.csv', index=False)

figures.plot_metric_vs_horizon(
    ablation.rename(columns={'model': 'rung'}), metric='skill', by='rung',
    title='Skill increment bought by each predictor, over the rung below it',
    save_path=str(OUTPUT_DIR), filename='NP_F11_ablation')
plt.show()
```

If `paired_mae_skill` names its output column something other than `skill`, use that name in the
`metric=` argument above; read `prediction.py:287` before running this cell.

- [ ] **Step 7: Execute and read off the horizon limit**

```bash
cd studies/04_neuralprophet_inclination_prediction
jupytext --sync neuralprophet_inclination_prediction_study.py
jupyter nbconvert --to notebook --execute --inplace \
  neuralprophet_inclination_prediction_study.ipynb --ExecutePreprocessor.timeout=14400
python - <<'PY'
import pandas as pd
skill = pd.read_csv('outputs/NP_12_skill_vs_baseline.csv')
best = skill[(skill.baseline == 'seasonal_naive') & (skill.model == 'AR + tair')]
print(best.to_string(index=False))
PY
```

Read off the largest horizon whose bootstrap interval excludes zero. That number, with its
interval, is the study's answer to the third question — whatever it turns out to be. A small or
absent increment for `tair` over `AR only` is a legitimate result: Study 03 measured a real
coupling, and a coupling that is real is not obliged to be *predictively* useful once the response's
own recent history is already in the model.

- [ ] **Step 8: Commit**

```bash
git add studies/04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.py \
        studies/04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.ipynb
git commit -m "$(cat <<'EOF'
feat(study04): forecast the hourly change and attribute the skill

Every model is scored against three baselines a monitoring system could run for
free, with paired block-bootstrap intervals, because autocorrelated residuals
make an unpaired comparison overstate significance. The ladder from AR-only
upward is what makes any of the skill attributable to a predictor rather than to
the response's own memory.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

> **CHECKPOINT 6 — show the user:** `NP_11`–`NP_13`, `NP_F10`, `NP_F11`, the horizon at which skill
> stops excluding zero, and the increment attributable to `tair` beyond autoregressive memory.
> Wait for approval before Task 18.

---

## Phase 7 · Can the model fill gaps?

### Task 18: The gap-closure verdict

**Files:**
- Modify: the notebook (append step 9)
- Produces: `outputs/NP_14_gap_closure.csv`, `NP_F12_gap_closure.{png,svg}`

**Interfaces:**
- Consumes: `segmented_b`, `predictions_b`, `hourly`; `prediction.gap_closure_summary`,
  `figures.plot_prediction_band`.
- Produces: an explicit verdict string reported in the notebook and the report.

- [ ] **Step 1: Add the step 9 Markdown cell**

```markdown
# %% [markdown]
# ## 9 · Can the model fill the gaps it was trained around?
#
# The operational temptation, once a model predicts a change, is to accumulate
# its predictions across a gap and call the result a reconstructed level. That
# is a stronger claim than anything measured so far: a per-step error that is
# small and unbiased still accumulates, and a reconstruction that arrives at the
# wrong level on the far side of a gap is worse than an admitted absence,
# because it looks like a measurement.
#
# The check is arithmetical rather than statistical. Where a gap is bracketed by
# accepted observations on both sides, the true change across it is known; the
# predicted changes are summed and compared. A reconstruction is safe only if it
# closes. Terminal gaps, and gaps crossing the instrument changeover, are
# reported with their status rather than scored.
```

- [ ] **Step 2: Add the step 9 code cell**

```python
# %%
prior_only = predictions_b[
    (predictions_b['model'] == 'AR + tair')
    & (predictions_b['horizon_h'] == 1)]

closure = prediction.gap_closure_summary(
    hourly[TARGET_COLUMN], prior_only.set_index('ds')['yhat'],
    era=None, freq=MODEL_FREQ_B)
display(closure)
closure.to_csv(OUTPUT_DIR / 'NP_14_gap_closure.csv', index=False)

closed = closure[closure['status'] == 'closed'] if 'status' in closure else closure
verdict = ('safe to accumulate' if len(closed) and
           closed['error'].abs().median() < 1.0 else 'NOT safe to accumulate')
print(f'Gap-closure verdict: reconstruction is {verdict}')
```

```python
# %%
# One bracketed gap drawn whole: the observed level either side, and the level
# the accumulated predictions arrive at across it. Whether the reconstruction
# closes is visible rather than only tabulated.
worst = closed.reindex(closed['error'].abs().sort_values(ascending=False).index)
example = worst.iloc[0]
span = slice(example['start'] - pd.Timedelta('48h'),
             example['end'] + pd.Timedelta('48h'))
observed_level = hourly[TARGET_COLUMN].loc[span]
reconstructed = (observed_level.ffill().iloc[0]
                 + prior_only.set_index('ds')['yhat'].reindex(
                     observed_level.index).fillna(0.0).cumsum())

figures.plot_prediction_band(
    observed_level, reconstructed, reconstructed, reconstructed,
    title='Accumulated prediction across the worst-closing bracketed gap',
    highlight=[(example['start'], example['end'])],
    save_path=str(OUTPUT_DIR), filename='NP_F12_gap_closure')
plt.show()
```

Read `prediction.py:748` before running this cell and use the column names
`gap_closure_summary` actually returns. The literal `1.0` above is replaced by the study's own
stated criterion in the next step.

- [ ] **Step 3: Add the closure criterion to the parameter cell**

```python
# The reconstruction is accepted only if the median absolute closure error
# across bracketed gaps stays below this many millidegrees. It is set against
# the minimum detectable step measured in step 7b: a reconstruction whose error
# exceeds what the monitor can detect would manufacture alarms.
GAP_CLOSURE_TOLERANCE_MDEG = 1.0
```

and replace the literal `1.0` in Step 2's cell with `GAP_CLOSURE_TOLERANCE_MDEG`.

- [ ] **Step 4: Execute and record the verdict**

```bash
cd studies/04_neuralprophet_inclination_prediction
jupytext --sync neuralprophet_inclination_prediction_study.py
jupyter nbconvert --to notebook --execute --inplace \
  neuralprophet_inclination_prediction_study.ipynb --ExecutePreprocessor.timeout=14400
```

- [ ] **Step 5: Commit**

```bash
git add studies/04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.py \
        studies/04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.ipynb
git commit -m "$(cat <<'EOF'
feat(study04): check whether predicted changes close the gaps they span

A small unbiased per-step error still accumulates, and a reconstruction that
arrives at the wrong level is worse than an admitted absence because it looks
like a measurement. The check is arithmetical: where a gap is bracketed, the
true change is known.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

> **CHECKPOINT 7 — show the user:** `NP_14` and an explicit safe/unsafe statement, with the
> tolerance it was judged against. Wait for approval before Task 19.

---

## Phase 8 · The report

### Task 19: Run metadata and the LaTeX table bodies

**Files:**
- Modify: the notebook (append step 10)
- Produces: `outputs/NP_15_run_metadata.csv` and one `.tex` table body per table the report quotes

**Interfaces:**
- Consumes: every parameter in the parameter cell; `tables.write_table`, `tables.to_rows`.
- Produces: `outputs/NP_*_body.tex` files, read by the report with `\input`.

- [ ] **Step 1: Add the step 10 code cell**

```python
# %%
import neuralprophet

metadata = pd.DataFrame([
    {'parameter': 'segment_start', 'value': SEGMENT_START},
    {'parameter': 'native_freq', 'value': NATIVE_FREQ},
    {'parameter': 'model_a_freq', 'value': MODEL_FREQ_A},
    {'parameter': 'model_b_freq', 'value': MODEL_FREQ_B},
    {'parameter': 'target_column', 'value': TARGET_COLUMN},
    {'parameter': 'predictors', 'value': ', '.join(PREDICTOR_COLUMNS)},
    {'parameter': 'model_a_lags', 'value': MODEL_A_LAGS},
    {'parameter': 'model_a_growth', 'value': MODEL_A_GROWTH},
    {'parameter': 'model_a_changepoints', 'value': MODEL_A_CHANGEPOINTS},
    {'parameter': 'model_a_yearly', 'value': MODEL_A_YEARLY},
    {'parameter': 'model_a_train_end', 'value': MODEL_A_TRAIN_END},
    {'parameter': 'model_a_epochs', 'value': MODEL_A_EPOCHS},
    {'parameter': 'model_b_lags', 'value': MODEL_B_LAGS},
    {'parameter': 'model_b_forecasts', 'value': MODEL_B_FORECASTS},
    {'parameter': 'model_b_regressor_lags', 'value': MODEL_B_REGRESSOR_LAGS},
    {'parameter': 'model_b_folds', 'value': MODEL_B_FOLDS},
    {'parameter': 'model_b_refit_each_fold', 'value': MODEL_B_REFIT_EACH_FOLD},
    {'parameter': 'model_b_epochs', 'value': MODEL_B_EPOCHS},
    {'parameter': 'seed', 'value': MODEL_A_SEED},
    {'parameter': 'quantiles', 'value': str(MODEL_A_QUANTILES)},
    {'parameter': 'reference_window',
     'value': f'{REFERENCE_START} to {REFERENCE_END}'},
    {'parameter': 'ewma_lambda', 'value': EWMA_LAMBDA},
    {'parameter': 'ewma_L', 'value': EWMA_L},
    {'parameter': 'cusum_k', 'value': CUSUM_K},
    {'parameter': 'cusum_h', 'value': CUSUM_H},
    {'parameter': 'joint_window', 'value': JOINT_WINDOW},
    {'parameter': 'target_arl_days', 'value': TARGET_ARL_DAYS},
    {'parameter': 'gap_closure_tolerance_mdeg',
     'value': GAP_CLOSURE_TOLERANCE_MDEG},
    {'parameter': 'neuralprophet_version', 'value': neuralprophet.__version__},
    {'parameter': 'pandas_version', 'value': pd.__version__},
    {'parameter': 'numpy_version', 'value': np.__version__},
])
metadata.to_csv(OUTPUT_DIR / 'NP_15_run_metadata.csv', index=False)
display(metadata)
```

- [ ] **Step 2: Add the table-body export cell**

```python
# %%
# LaTeX bodies for the tables the report quotes. The report reads these with
# \input, so a re-run of the notebook updates the report's numbers without any
# hand editing - which is what keeps every number in the prose traceable.
tables.write_table(
    coverage.reset_index(), str(OUTPUT_DIR / 'NP_01_body.tex'),
    columns=['channel', 'accepted', 'coverage'])
tables.write_table(
    gaps.groupby('gap_class', as_index=False)
        .agg(gaps=('n_slots', 'size'), hours=('duration_h', 'sum')),
    str(OUTPUT_DIR / 'NP_02_body.tex'),
    columns=['gap_class', 'gaps', 'hours'])
tables.write_table(
    survival, str(OUTPUT_DIR / 'NP_03_body.tex'),
    columns=['lag_hours', 'forecast_hours', 'n_segments', 'n_surviving',
             'n_windows'])
tables.write_table(
    cadence, str(OUTPUT_DIR / 'NP_04_body.tex'),
    columns=['cadence', 'level_autocorr1', 'change_autocorr1', 'change_std',
             'corr_change', 'drift_per_year'])
tables.write_table(
    shares, str(OUTPUT_DIR / 'NP_05_body.tex'),
    columns=['component', 'variance', 'share', 'peak_to_peak'])
tables.write_table(
    gains, str(OUTPUT_DIR / 'NP_06_body.tex'),
    columns=['source', 'gain_mdeg_per_degC', 'method'])
tables.write_table(
    stability, str(OUTPUT_DIR / 'NP_16_body.tex'),
    columns=['block', 'tair_gain_mdeg_per_degC', 'trend_mdeg_per_year'])
tables.write_table(
    nowcast_scores, str(OUTPUT_DIR / 'NP_07_body.tex'),
    columns=['model', 'n', 'mae', 'rmse', 'bias', 'mase',
             'coverage_q05_q95', 'width_q05_q95'])
tables.write_table(
    metrics_b, str(OUTPUT_DIR / 'NP_11_body.tex'),
    columns=['model', 'horizon_h', 'n', 'mae', 'rmse', 'mase',
             'coverage_q05_q95'])
tables.write_table(
    skill, str(OUTPUT_DIR / 'NP_12_body.tex'),
    columns=['baseline', 'model', 'horizon_h', 'skill', 'ci_low', 'ci_high'])
tables.write_table(
    metadata, str(OUTPUT_DIR / 'NP_15_body.tex'),
    columns=['parameter', 'value'])
```

Check `NP_12`'s real column names against `prediction.paired_mae_skill` before running.

- [ ] **Step 3: Execute, then commit**

```bash
cd studies/04_neuralprophet_inclination_prediction
jupytext --sync neuralprophet_inclination_prediction_study.py
jupyter nbconvert --to notebook --execute --inplace \
  neuralprophet_inclination_prediction_study.ipynb --ExecutePreprocessor.timeout=14400
ls outputs/
git add neuralprophet_inclination_prediction_study.py neuralprophet_inclination_prediction_study.ipynb
git commit -m "$(cat <<'EOF'
feat(study04): export run metadata and the report's table bodies

The report reads its numbers from these files rather than carrying them in
prose, so re-running the notebook updates the report and no figure in the text
can drift away from the artefact behind it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 20: Write the report

**Files:**
- Modify: `studies/04_neuralprophet_inclination_prediction/report/neuralprophet_inclination_prediction_report.tex`

**Interfaces:**
- Consumes: every artefact in `outputs/`.
- Produces: the study's deliverable.

**This task is never delegated.** It is the one place where the study says what it found, and it is
written in full academic prose — the caveman register does not apply to a single word of it.

- [ ] **Step 1: Keep the standing front matter and introduction**

Lines 1–85 of the report stand as written: the preamble, the title, the introduction's account of
what Study 01 established and why the window begins in June 2023. Only the deleted paragraph is
gone.

- [ ] **Step 2: Add section 2 — the record this study works on**

Reuse the existing `NP_F01` figure and its caption. Add one paragraph stating the window's extent
and the coverage of each channel, with `\input{../outputs/NP_01_body.tex}` inside a `tabular`.

- [ ] **Step 3: Add section 3 — the anatomy of what is missing**

State the finding measured in Task 6: 1,647 gaps over 6,789 missing hours, of which four gaps
longer than seven days carry 72.8 % of the missing time while 1,282 gaps of an hour or less carry
8.6 % of it and 78 % of the gap count. Explain why the second number matters more than the first —
the short gaps are what break contiguity, and contiguity is what an autoregressive window costs.
Include `NP_F02`, `NP_F03`, and the bodies `NP_02_body.tex` and `NP_03_body.tex`. Note explicitly
that `tair` and `rh` cost no coverage at all because their missingness is nested inside the
inclination's.

- [ ] **Step 4: Add section 4 — method, one subsection per decision**

Seven subsections, each stating the decision, the evidence, and what would have gone wrong
otherwise:

4.1 Why the level is decomposed but never scored — the 0.998 autocorrelation.
4.2 Why forecasting is done on the hourly change — −0.058 against +0.349, and −0.778 against −0.893.
4.3 Why the anomaly test is not differenced, and why contemporaneous regressors are not leakage.
4.4 Why two models — what autoregression does to the seasonal component.
4.5 How gaps are honoured — segment identifiers, no imputation, and the measured fact that
NeuralProphet's defaults silently fabricate gaps of at least thirty hours.
4.6 Changepoints on covered time; no era offset.
4.7 The metrics, and what each is for — with the note that coverage alone can be bought by
widening, which is why the interval score and pinball loss appear beside it.

- [ ] **Step 5: Add section 5 — what the record is made of**

The component shares from `NP_05_body.tex`, the decomposition stack `NP_F05`, the daily cycle and
regressor response `NP_F06`, and the confrontation with Study 03 from `NP_06_body.tex`. State
plainly whether the two methods agree, and by how much. Add the stability table
`NP_16_body.tex` beside it: a gain that keeps its sign and magnitude across three disjoint
stretches of the record is a finding, and one that does not is reported as instability. State that
the gain is fitted on the compensated channel and is therefore what the wall does after Study 01's
correction, which is the only quantity a monitoring system sees; do not argue about the correction
itself (D10).

- [ ] **Step 6: Add section 6 — is this reading expected?**

`NP_07_body.tex` and `NP_F07`. Report the observed interval coverage against its nominal 90 %
without adjustment.

- [ ] **Step 7: Add section 7 — judging a departure**

The reference window, and why it stops before the stretch Study 01 flagged in summer 2026 — a
calibration precaution, since a reference window must be in control, and explicitly not a test. The
chosen limit width and the run length it delivers, in days. `NP_F08` and the episode table.

Then the sensitivity statement, which is where this section's weight sits: the three
`NP_F09_detectability_*` figures and the smallest departure found for each mechanism at each
persistence. Say what each mechanism stands for in a three-leaf wall — amplitude growth for loss of
composite action between the leaves, a timing shift for a changed thermal path such as water in the
core, drift for mortar creep or settlement — and quote each in the unit that makes it physical:
millidegrees for the swing, hours of shift for the timing, millidegrees per year for the drift. A
mechanism undetected across the whole swept range is reported as a limit of this instrument and this
model, which is a result about the method's reach and not a failure to be hidden.

**No claim whatsoever is made about the summer-2026 stretch** (D12): it is too large to
demonstrate sensitivity, and a detector that finds it has shown only that it is not broken.

- [ ] **Step 8: Add section 8 — how far ahead is prediction worth anything?**

`NP_11_body.tex`, `NP_12_body.tex`, `NP_F10`, `NP_F11`. The horizon at which skill stops excluding
zero, with its interval. The increment attributable to air temperature beyond autoregressive
memory, whatever its size, and the battery control's floor beside it.

- [ ] **Step 9: Add section 9 — can the model fill gaps?**

The closure verdict, the tolerance it was judged against, and why an unsafe reconstruction is worse
than an admitted absence.

- [ ] **Step 10: Add section 10 — verdict**

One paragraph per question, each carrying its qualification.

- [ ] **Step 11: Add section 11 — limitations**

The compensation is not independent of the thermal term being fitted. Wall temperature and solar
radiation were excluded and cover 17 % of the window. The battery control is not a silent channel.
Three annual cycles with a 103-day hole cannot identify a yearly term. One sensor, one station, one
site. The known event is a single case and cannot establish a detection rate.

- [ ] **Step 12: Build the report twice and check every reference resolves**

```bash
cd studies/04_neuralprophet_inclination_prediction/report
pdflatex neuralprophet_inclination_prediction_report.tex
pdflatex neuralprophet_inclination_prediction_report.tex
grep -c "undefined" neuralprophet_inclination_prediction_report.log || true
```
Expected: no undefined references, no missing files.

- [ ] **Step 13: Update the README to describe the finished study**

Change `Status: in progress` to `Status: complete`, add the outputs table listing `NP_01`–`NP_15`
and `NP_F01`–`NP_F12`, and replace the three questions with the three answers in one sentence each.
Update the study 04 row of `studies/README.md` to `Complete` with the widened question.

- [ ] **Step 14: Run every test one final time**

```bash
cd studies
for f in shmlib/tests/test_shmlib.py shmlib/tests/test_monitoring.py \
         04_neuralprophet_inclination_prediction/tests/test_prediction.py \
         04_neuralprophet_inclination_prediction/tests/test_gaps.py \
         04_neuralprophet_inclination_prediction/tests/test_decomposition.py \
         04_neuralprophet_inclination_prediction/tests/test_folder_honesty.py \
         03_thermomechanical_response/tests/test_shmlib_study03.py; do
  echo "== $f"; python "$f" 2>&1 | tail -3
done
```
Expected: `OK` from every file. `test_folder_honesty.py` now also proves that every artefact the
README and the report name exists on disk.

- [ ] **Step 15: Commit**

```bash
git add studies/04_neuralprophet_inclination_prediction/report \
        studies/04_neuralprophet_inclination_prediction/README.md \
        studies/README.md
git commit -m "$(cat <<'EOF'
docs(study04): write the report

Eleven sections, each decision stated where it is made and before the result
that depends on it. Every number in the prose is read from a table body in
outputs/, so no claim can drift away from the artefact behind it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

> **CHECKPOINT 8 — show the user:** the built PDF, the passing test sweep, and a spot check that
> every number in the report's verdict section traces to a named file in `outputs/`.

---

## Notes for the executor

**Delegation.** Tasks 2–5, 7–12, 17 and 18 are mechanical once specified and suit a Sonnet subagent
under the written instruction above; each spawn prompt must include "respond in caveman ultra
mode". Tasks 1, 6, 13, 14, 15, 16 and 19 involve judgement about what the record supports and stay
with the orchestrator. **Task 20 is never delegated.**

**Runtime.** Model A fits on roughly 60,000 twenty-minute rows and Model B runs four specifications
across five folds. Expect tens of minutes per full notebook execution on Apple Silicon. Use
`MODEL_A_EPOCHS=1`, `MODEL_B_EPOCHS=1` and a two-element `MODEL_B_HORIZONS` for a path check before
any full run, and record in `NP_15` which setting produced the committed artefacts.

**Google Drive.** The tree lives inside a sync daemon's reach. Pause syncing before any long
notebook execution or heavy git operation; concurrent writes into `.git` are a known source of
corruption.

**When a checkpoint fails.** Stop and report. Do not adjust a model until it produces the expected
number — the disagreement is the result. This applies with full force at Checkpoint 4, where the
fitted thermal gain is compared with Study 03's independent measurement.
