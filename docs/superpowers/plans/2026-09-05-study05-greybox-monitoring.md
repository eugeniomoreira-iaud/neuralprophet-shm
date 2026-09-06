# Study 05 · Grey-box expectation and monitoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Study 05 — the additive grey-box NeuralProphet decomposition, rolling expectation, three-scale monitor and outage bridges on the whole station 02 record — with the report written section by section as each phase's artefacts land, so progress is visible in the PDF at every checkpoint.

**Architecture:** Every function lives in `studies/shmlib/`; the notebook `studies/05_greybox_monitoring/greybox_monitoring_study.py` orchestrates with parameters declared in its parameter cells; tables and figures land in `outputs/` under the `GM_` prefix; the report `report/greybox_monitoring_report.tex` starts as a complete skeleton whose sections carry a `\pending{}` marker that each phase replaces with prose, tables and figures. A folder-honesty test forbids any figure or table body the report names without a file behind it.

**Tech Stack:** Python 3 in conda env `neuralprophet_env` (`/Users/eugenio/anaconda3/envs/neuralprophet_env/bin/python`), NeuralProphet 0.8.0, pandas 2.3, numpy 1.26, scipy, matplotlib/seaborn via `shmlib.viz`, jupytext, nbconvert, pdflatex. Tests are `unittest`, run directly with `python <file>`.

**Spec:** `docs/superpowers/specs/2026-09-05-study05-greybox-monitoring-design.md` (approved 2026-09-05). Read it before any task; every task below cites the decision it implements.

## Progress

Updated 2026-09-06 · 19:04 UTC. This block is the summary a reader needs to follow the
implementation; the checkboxes under each task below are ticked as the work lands, and the
detailed execution ledger (rulings, fix rounds, commits) lives in
`.superpowers/sdd/2026-09-05-study05-greybox-monitoring/progress.md`, which is gitignored.

| Measure | Progress |
|---|---|
| Tasks complete (of 41, Tasks 0.1 to 7.3) | `[████████████░░░░░░░░]` **24 of 41** (58 %) |
| Report sections written (of 14) | `[███████████░░░░░░░░░]` **8 of 14** (57 %) |

| Phase | Tasks | State | Result and commits |
|---|---|---|---|
| 0 · Smoke tests, housekeeping, report skeleton | 0.1–0.4 | ✅ complete | 8a5b32c…bea9ef1. All five NeuralProphet 0.8.0 capabilities confirmed (no design fallback needed). Checkpoint 0 approved. |
| 1 · Data and regressor sets | 1.1–1.4 | ✅ complete | bc76907…5c73f6d (+02a9bf2, c8da472). `GM_01`–`GM_03`, `GM_F01`; report §1–§3. Checkpoint 1 approved. |
| 1b · Harmonic diagnostics | 1b.1–1b.6 | ✅ complete | 12043f6…6176256. `GM_04`, `GM_F02`; `YEARLY_ORDER = 1`, `DAILY_ORDER = 2`, weight curve from the residual's order-two fit; report §5.1. Checkpoint 1b approved. |
| 2 · Model A attribution on three sets | 2.1–2.6, 2.4b, 2.4c | ✅ complete | a4652cf…2853c8a, 16f3cb1, 442f5d6. `GM_04d`, `GM_05`–`GM_08b`, `GM_F03`–`GM_F06b`; `TREND_REG = 0.0`; conditional daily term rejected. **Checkpoint 2 passed:** on-structure air-temperature gain −2.64 mdeg/°C against Study 03's −2.79, inside its interval. Report §4, §5.2, §6. |
| 2b · Current-era ladder | 2b.1–2b.2 | ✅ complete | ef0a67a, 6882563, 74cf7dc; 0d63e51, e5e07ba. `GM_16`, `GM_F14` on one matched window. Neither the pyranometer nor the probe buys anything for the expectation on the current era. Report §9 written. |
| 3 · Expectation and interval | 3.1–3.3 | ✅ committed · in review | 9f4719f, 4f0278f; §8 committed. `GM_09`, `GM_F08`: pooled coverage 88.4 % (on-structure) against Study 04's 67.7 %; coverage 96 → 82 % and MAE 5.3 → 10.9 mdeg across the refit month. Two runs of about an hour each (the first died on the native conformal plot). |
| 4 · Model B impulse response | 4.1–4.3 | ⬜ pending | `GM_10`, `GM_F07`; report §7. |
| 5 · The monitor | 5.1–5.5 | ⬜ pending | `GM_11`–`GM_13`, `GM_F09`–`GM_F11`, `GM_F13`; report §10. |
| 6 · Outage bridges | 6.1–6.3 | ⬜ pending | `GM_14`, `GM_F12`; report §11. |
| 7 · Closure and the revision pass | 7.1–7.3 | ⬜ pending | `GM_15`; report §12–§14; then Task 7.3, the revision pass from `report05_check.md`. |

Report sections: 1 Introduction ✅ · 2 Record ✅ · 3 Method ✅ · 4 Trend ✅ · 5 Seasonality ✅ · 6 Regressors ✅ · 7 Impulse response ⬜ · 8 Uncertainty ✅ · 9 Ladder ✅ · 10 Monitor ⬜ · 11 Outages ⬜ · 12 Verdict ⬜ · 13 Limitations ⬜ · 14 Run metadata ⬜.

## Global Constraints

- **Caveman ultra, always.** Every subagent prompt starts with "respond in caveman ultra mode". Produced documents (report `.tex`, README, docstrings, comments, commit messages) are full prose. (Spec §0.)
- **Delegate down.** Tasks marked **S** are for a Sonnet subagent; tasks marked **O** (report prose, checkpoint reviews, Phase 2 attribution interpretation) stay with the orchestrator. (Spec §0.)
- **Library-first.** No function is defined in the notebook or the study folder; every function goes in `studies/shmlib/`. A cell longer than about ten lines that defines a function, builds a table row by row or composes a figure axis by axis is a defect. (`instructions-pipeline.md` § Studies.)
- **Additive adaptations only.** Every change to an existing `shmlib` function keeps every existing caller's behaviour, defaults included. After each adaptation run all of: `python shmlib/tests/test_shmlib.py`, `python shmlib/tests/test_monitoring.py`, `python 03_thermomechanical_response/tests/test_shmlib_study03.py`, `python 04_neuralprophet_inclination_prediction/tests/test_prediction.py`, `test_gaps.py`, `test_decomposition.py`, `test_folder_honesty.py`, all from `studies/`. (Spec §5.2.)
- **No imputation of the target.** `impute_missing=False, drop_missing=False` on every NeuralProphet constructor; regressor gaps ≤ 2 h filled and flagged; nothing else filled. (Spec D13.)
- **Compensation as given.** `inc_comp_cleaned` only; no raw-channel fit anywhere. (Spec §2.1.)
- **Figure rules** from `instructions-pipeline.md`: Okabe–Ito channel colours through `viz.channel_style`/`viz.driver_colour`, Cividis for scalars, legends below the axes (`loc='upper center', bbox_to_anchor=(0.5, -0.30), frameon=False`), span highlights black at 5 %, accent `#D55E00` for markers only, every figure through `viz.finish` (PNG + SVG). (Spec §5.4.)
- **Artefact names** exactly as spec §6: `GM_01`…`GM_16`, `GM_F01`…`GM_F14`, LaTeX bodies `GM_NN_body.tex` beside each CSV.
- **Every number in the report traces to a `GM_` artefact.** A claim with no artefact does not enter the report. (Spec §7.)
- **Every image in the report comes from the paired notebook.** Each `\includegraphics{GM_Fxx_…}` in `report/greybox_monitoring_report.tex` names a file that a `figures.*` call in `greybox_monitoring_study.py` writes with `save_path=str(OUTPUT_DIR), filename='GM_Fxx_…'`, and each `\input{../outputs/GM_xx_body.tex}` names a body that a `tables.write_table` call in the same notebook writes. No image reaches the PDF from a test, a helper script, a hand edit or the paper side; a figure that needs changing is changed in `shmlib.figures` and the notebook is re-run. The folder-honesty test enforces the mapping from Task 2.4b onward. (User rule, 2026-09-06.)
- **The notebook tells the code side of the story.** Each movement opens with a Markdown cell stating what it computes, why, and which `GM_` artefacts it writes, and each step inside a movement has its own `###` Markdown cell in the same voice, so that a reader who follows the report can open the notebook at the matching movement and find the calls that produced every figure and table. Code comments do not substitute for these cells; reviewers of every notebook task check for them. (User rule, 2026-09-06.)
- **Report in parts.** Each phase ends with a task that writes that phase's report section(s), removes their `\pending{}` marker, rebuilds the PDF twice, runs the honesty test and commits. The PDF is tracked.
- **Notebook execution.** Edit the `.py`; sync with `jupytext --to ipynb`. Execute to a scratch path with `--output-dir` (or stop `auto_watcher.py` first), verify by reading printed lines back from the executed `.ipynb`, since nbconvert exits 0 on a failing cell. (Study 04 README.)
- **Commits.** One per task, message in full prose, ending with the two trailer lines `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01UarwTTUzLD2P9hff6RJHew`.
- **Paths.** Repo root `neuralprophet-shm/neuralprophet-shm`; run tests and notebooks from `studies/`; the study folder is `studies/05_greybox_monitoring/`.

---

## File structure

| File | Responsibility | Tasks |
|---|---|---|
| `studies/05_greybox_monitoring/tests/test_neuralprophet_capabilities.py` | Phase 0 smoke tests of NeuralProphet 0.8.0 features the design relies on | 0.1 |
| `studies/05_greybox_monitoring/tests/test_folder_honesty.py` | Folder hygiene: no code outside shmlib, every included graphic and table body exists, no `\pending` at the end | 0.4, 7.2 |
| `studies/05_greybox_monitoring/report/greybox_monitoring_report.tex` | The report, all 14 sections from Phase 0, filled phase by phase | 0.4, 1.4, 1b.6, 2.6, 2b.2, 3.3, 4.3, 5.5, 6.3, 7.1 |
| `studies/shmlib/proxies.py` | + `to_native_grid`, `fill_short_gaps` | 1.1, 1.2 |
| `studies/shmlib/tests/test_proxies_grid.py` | Tests for the two proxy helpers | 1.1, 1.2 |
| `studies/shmlib/monitoring.py` | + `daily_harmonic`, `prewhiten`, `channel_coincidence`; `detectability_curve(statistic=…)` | 1b.1, 5.1, 5.2, 5.3 |
| `studies/shmlib/coupling.py` | + `annual_modulation`, `evaluate_modulation`, `cycle_surface_rank` | 1b.2, 1b.3 |
| `studies/shmlib/prediction.py` | backtest/rolling adaptations; + `seasonal_weights`, `trend_parameters`, `seasonal_parameters`, `regressor_gains`, `rolling_conformal`, `lagged_regressor_weights`, `impulse_response_summary`, `outage_bridge` | 1b.4, 2.1, 2.2, 3.1, 4.1, 6.1 |
| `studies/shmlib/figures.py` | + `plot_harmonic_diagnostics`, `plot_fit_metrics`, `plot_trend_parameters`, `plot_seasonal_parameters`, `plot_regressor_gains`, `plot_impulse_response`, `plot_daily_harmonic_chart`, `plot_outage_bridge` | 1b.5, 2.3, 4.1, 5.3, 6.1 |
| `studies/05_greybox_monitoring/tests/test_harmonics.py` | Tests for Phase 1b functions | 1b.1–1b.4 |
| `studies/05_greybox_monitoring/tests/test_model_a.py` | Tests for backtest adaptations and extractors | 2.1, 2.2, 3.1 |
| `studies/05_greybox_monitoring/tests/test_model_b.py` | Tests for lagged-regressor weights and impulse summary | 4.1 |
| `studies/05_greybox_monitoring/tests/test_monitor.py` | Tests for prewhitening, attribution, statistic-aware detectability | 5.1–5.3 |
| `studies/05_greybox_monitoring/tests/test_bridges.py` | Tests for `outage_bridge` | 6.1 |
| `studies/05_greybox_monitoring/greybox_monitoring_study.py` | The notebook: parameter cells exist; analysis cells added per phase under `# %% [markdown]\n# ## Movement N` headings | 1.3, 1b.6, 2.4, 2b.1, 3.2, 4.2, 5.4, 6.2, 7.1 |
| `studies/05_greybox_monitoring/README.md` | Status line and artefact list updated at each checkpoint | 1.4 … 7.2 |
| `studies/02_proxy_forcing_characterization/README.md` | One status line reconciled | 0.3 |
| `studies/README.md` | Study 5 status row: "Design approved" → "In progress" (Phase 1) → "Complete" (Phase 7) | 1.4, 7.2 |

Test-file conventions (copy exactly): module docstring with the run command, `import os, sys, unittest`, `import numpy as np`, `import pandas as pd`, then

```python
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
from shmlib import <module>  # noqa: E402
```

and `if __name__ == '__main__': unittest.main(verbosity=2)`. For `shmlib/tests/` files the path insert is the same (`'..', '..'` from `shmlib/tests/`).

Figure-function template (from `shmlib/figures.py:1902`, `plot_gap_anatomy`): NumPy docstring → compute → `fig, axes = plt.subplots(..., figsize=viz.figsize(viz.FIGURE_WIDTH, h))` → draw → `viz.format_spines(ax)` per axes → `fig.suptitle(title)` if given → `viz.finish(fig, save_path=save_path, filename=filename)` → `return fig`.

Table-writing template (Study 04 notebook `:1574`):

```python
tables.write_table(
    frame, str(OUTPUT_DIR / 'GM_05_body.tex'),
    [('component', tables.texttt), ('variance', ',.1f'), ('share', tables.percent),
     ('peak_to_peak', ',.1f')])
```

---

# Phase 0 · Smoke tests, housekeeping, report skeleton

### Task 0.1 (S): NeuralProphet 0.8.0 capability tests

**Files:**
- Create: `studies/05_greybox_monitoring/tests/test_neuralprophet_capabilities.py`

**Interfaces:**
- Produces: a green test file that later tasks rely on as proof that lagged regressors work with `n_lags=0`, that `conformal_predict` returns `yhat1 - qhat1` / `yhat1 + qhat1`, that `predict_trend` and `predict_seasonal_components` exist, that float condition columns are accepted by `add_seasonality`, and that the matplotlib backend still returns a Figure.

- [x] **Step 1: Write the tests**

```python
"""
Smoke tests for the NeuralProphet 0.8.0 capabilities Study 05 relies on.

Run from studies/:  python 05_greybox_monitoring/tests/test_neuralprophet_capabilities.py

Each test states one capability the design assumes (spec D6-D9, D14). A
failure here means the design's fallback for that capability applies.
"""
import logging
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

logging.getLogger('NP').setLevel(logging.ERROR)


def _frame(n=24 * 40, freq='1h', seed=0):
    rng = np.random.default_rng(seed)
    ds = pd.date_range('2024-01-01', periods=n, freq=freq)
    hours = np.arange(n)
    x = 10.0 + 5.0 * np.sin(2 * np.pi * hours / 24.0)
    y = 100.0 - 2.5 * np.roll(x, 3) + rng.normal(0, 0.3, n)
    doy = ds.dayofyear.to_numpy()
    summer_w = 0.5 * (1 - np.cos(2 * np.pi * (doy - 15) / 365.0))
    return pd.DataFrame({'ds': ds, 'y': y, 'x': x,
                         'summer_w': summer_w, 'winter_w': 1 - summer_w})


def _model(**kwargs):
    from neuralprophet import NeuralProphet
    base = dict(n_lags=0, n_forecasts=1, yearly_seasonality=False,
                weekly_seasonality=False, daily_seasonality=True,
                epochs=5, learning_rate=0.05, impute_missing=False,
                drop_missing=False, quantiles=[0.05, 0.95])
    base.update(kwargs)
    return NeuralProphet(**base)


class TestLaggedRegressorWithoutAutoregression(unittest.TestCase):
    """Spec D9: Model B carries lagged regressors with n_lags=0 on the target."""

    def test_fit_and_predict_with_lagged_regressor_only(self):
        df = _frame()
        m = _model()
        m.add_lagged_regressor('x', n_lags=6)
        m.fit(df, freq='1h', progress='none', minimal=True)
        out = m.predict(df, decompose=True)
        self.assertIn('yhat1', out.columns)
        lagged = [c for c in out.columns if c.startswith('lagged_regressor')]
        self.assertTrue(lagged, 'no lagged_regressor component column')


class TestConformalPredict(unittest.TestCase):
    """Spec D8: split conformal prediction with the cqr method."""

    def test_conformal_columns(self):
        df = _frame()
        train, cal, test = df.iloc[:600], df.iloc[600:800], df.iloc[800:]
        m = _model()
        m.add_future_regressor('x')
        m.fit(train, freq='1h', progress='none', minimal=True)
        out = m.conformal_predict(test, calibration_df=cal, alpha=0.1,
                                  method='cqr')
        self.assertIn('yhat1 - qhat1', out.columns)
        self.assertIn('yhat1 + qhat1', out.columns)


class TestParameterExtractors(unittest.TestCase):
    """Spec D14: redraws read public methods, not figures."""

    def test_predict_trend_and_seasonal_components(self):
        df = _frame()
        m = _model()
        m.fit(df, freq='1h', progress='none', minimal=True)
        trend = m.predict_trend(df)
        seasonal = m.predict_seasonal_components(df)
        self.assertEqual(len(trend), len(df))
        self.assertIn('daily', seasonal.columns)

    def test_matplotlib_backend_returns_figure(self):
        import matplotlib
        matplotlib.use('Agg')
        df = _frame()
        m = _model()
        m.fit(df, freq='1h', progress='none', minimal=True)
        fig = m.plot_parameters(plotting_backend='matplotlib')
        self.assertTrue(hasattr(fig, 'savefig'))


class TestConditionalSeasonalityWithFloatWeights(unittest.TestCase):
    """Spec D7: two daily series blended by float weights in 0..1."""

    def test_float_conditions_are_accepted_and_decomposed(self):
        df = _frame()
        m = _model(daily_seasonality=False)
        m.add_seasonality(name='daily_summer', period=1, fourier_order=3,
                          condition_name='summer_w')
        m.add_seasonality(name='daily_winter', period=1, fourier_order=3,
                          condition_name='winter_w')
        m.fit(df, freq='1h', progress='none', minimal=True)
        out = m.predict(df, decompose=True)
        self.assertIn('daily_summer', out.columns)
        self.assertIn('daily_winter', out.columns)


if __name__ == '__main__':
    unittest.main(verbosity=2)
```

- [x] **Step 2: Run the tests**

Run from `studies/`: `python 05_greybox_monitoring/tests/test_neuralprophet_capabilities.py`
Expected: all five PASS. If `TestLaggedRegressorWithoutAutoregression` fails, record it in the task report: the spec's fallback (Model B hourly with `n_lags=12`) applies and Task 4.2 uses `MODEL_B_FALLBACK_FREQ`/`MODEL_B_FALLBACK_LAGS`. If `test_float_conditions…` fails, the fallback is four boolean seasons (spec D7).

- [x] **Step 3: Commit**

```bash
git add studies/05_greybox_monitoring/tests/test_neuralprophet_capabilities.py
git commit -m "test(study05): pin the NeuralProphet 0.8.0 capabilities the design relies on"
```

### Task 0.2 (S): Bring the studies knowledge graph up to date

**Files:** none edited by hand; `studies/graphify-out/` regenerated.

- [x] **Step 1: Update the graph**

Run from `studies/`: `graphify update .` (the `--update` flow: re-extracts only changed files). Expected: the summary reports new nodes for `shmlib/monitoring.py` and `05_greybox_monitoring/`.

- [x] **Step 2: Verify**

Run: `graphify query "monitoring control charts detectability" --budget 800` from `studies/`. Expected: nodes from `shmlib/monitoring.py` appear (`ewma_chart`, `detectability_curve`).

- [x] **Step 3: Commit**

```bash
git add studies/graphify-out
git commit -m "chore(studies): refresh the knowledge graph to include study 04 and the study 05 scaffold"
```

### Task 0.3 (S): Reconcile Study 02's status line

**Files:**
- Modify: `studies/02_proxy_forcing_characterization/README.md` (the line beginning `**Status: skeleton.**`)

- [x] **Step 1: Edit the line**

Replace the paragraph starting `**Status: skeleton.**` with:

```markdown
**Status: in progress.** The notebook runs and writes `PF_01` to `PF_14` and `PF_F01` to
`PF_F18` into `outputs/`; the report's source characterisation sections are written, and its
comparison, compatibility and conclusion sections are still to be written from those artefacts.
```

- [x] **Step 2: Verify the index agrees**

Run: `grep -n "02_proxy_forcing" studies/README.md`. Expected: the row's status reads `In progress` (unchanged).

- [x] **Step 3: Commit**

```bash
git add studies/02_proxy_forcing_characterization/README.md
git commit -m "docs(study02): state the study's real status, in progress with artefacts on disk"
```

### Task 0.4 (S writes skeleton and test; O reviews): Report skeleton with pending sections, and the folder-honesty test

**Files:**
- Create: `studies/05_greybox_monitoring/report/greybox_monitoring_report.tex`
- Create: `studies/05_greybox_monitoring/tests/test_folder_honesty.py`
- Delete: `studies/05_greybox_monitoring/report/.gitkeep`, `studies/05_greybox_monitoring/tests/.gitkeep`

**Interfaces:**
- Produces: `\pending{<phase>}` macro; 14 `\section` headings with fixed `\label`s (`sec:intro`, `sec:record`, `sec:method`, `sec:trend`, `sec:seasonality`, `sec:regressors`, `sec:impulse`, `sec:uncertainty`, `sec:ladder`, `sec:monitor`, `sec:outages`, `sec:verdict`, `sec:limitations`, `sec:metadata`). Later report tasks replace the `\pending{}` line inside a section and nothing else.

- [x] **Step 1: Write the honesty test**

```python
"""
Guards that the study folder never claims a result it does not hold.

Run from studies/:  python 05_greybox_monitoring/tests/test_folder_honesty.py

Repository hygiene rather than numerics: no code outside shmlib, every
graphic and table body the report includes exists on disk, and, once the
study is complete, no section is still marked pending.
"""
import re
import unittest
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
REPORT = STUDY / 'report' / 'greybox_monitoring_report.tex'
README = STUDY / 'README.md'
OUTPUTS = STUDY / 'outputs'

# Flipped to True in the final task, when every artefact the README names
# must exist and no section may remain pending.
STUDY_COMPLETE = False


class TestNoCodeOutsideShmlib(unittest.TestCase):

    def test_only_the_notebook_source_is_a_python_file_in_the_study_root(self):
        found = sorted(p.name for p in STUDY.glob('*.py'))
        self.assertEqual(found, ['greybox_monitoring_study.py'])


class TestReportClaimsAreSupported(unittest.TestCase):

    def test_every_included_graphic_exists(self):
        text = REPORT.read_text(encoding='utf-8')
        for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text):
            stem = Path(name).stem
            self.assertTrue(list(OUTPUTS.glob(stem + '.*')),
                            f'report includes {name}; outputs/ has no such file')

    def test_every_input_table_body_exists(self):
        text = REPORT.read_text(encoding='utf-8')
        for name in re.findall(r'\\input\{\.\./outputs/([^}]+)\}', text):
            self.assertTrue((OUTPUTS / name).exists(),
                            f'report inputs {name}; outputs/ has no such file')

    def test_no_section_is_pending_once_complete(self):
        if not STUDY_COMPLETE:
            self.skipTest('study in progress; pending sections allowed')
        text = REPORT.read_text(encoding='utf-8')
        self.assertNotIn(r'\pending{', text)


class TestReadmeMatchesTheFolder(unittest.TestCase):

    def test_readme_names_only_artefacts_that_exist(self):
        if not STUDY_COMPLETE:
            self.skipTest('study in progress; README lists planned artefacts')
        text = README.read_text(encoding='utf-8')
        for name in re.findall(r'`(GM_F?\d\d[A-Za-z0-9_]*)', text):
            self.assertTrue(list(OUTPUTS.glob(name + '*')),
                            f'README names {name}; outputs/ does not contain it')


if __name__ == '__main__':
    unittest.main(verbosity=2)
```

- [x] **Step 2: Run it; expected: the two guarded tests skip, the others pass** (the report does not exist yet, so `test_every_included_graphic_exists` will ERROR — that is the failing state before Step 3).

Run from `studies/`: `python 05_greybox_monitoring/tests/test_folder_honesty.py`

- [x] **Step 3: Write the report skeleton**

```latex
% ---------------------------------------------------------------------------
% Study report · Grey-box expectation and monitoring of the station 02 inclination
%
% Build: pdflatex greybox_monitoring_report.tex (twice, for the ToC)
% Figures and table bodies are read from ../outputs/ and are produced by the
% paired notebook, which must be run first. Sections are written phase by
% phase; a section still to be written carries a \pending{} note.
% ---------------------------------------------------------------------------
\documentclass[11pt,a4paper]{article}

\newcommand{\runninghead}{Grey-box expectation and monitoring at Gubbio}

\input{../../_shared/reportstyle.tex}
\graphicspath{{../outputs/}}

\newcommand{\pending}[1]{\begin{center}\textcolor{rulegrey}{\itshape
This section is written after #1. Nothing here is a result yet.}\end{center}}

\title{\vspace{-1.2cm}\textbf{Is this reading the one the wall was expected to show?}\\[2mm]
\large A grey-box NeuralProphet decomposition, rolling expectation and
three-scale monitor of the station 02 inclination, 2018 to 2026}
\author{Study report · \texttt{studies/05\_greybox\_monitoring/}}
\date{}

\begin{document}
\maketitle
\thispagestyle{fancy}

\tableofcontents

% ===========================================================================
\section{Introduction}
\label{sec:intro}
\pending{Phase 1}

% ===========================================================================
\section{The record and the three regressor sets}
\label{sec:record}
\pending{Phase 1}

% ===========================================================================
\section{Method}
\label{sec:method}
\pending{Phase 1}

% ===========================================================================
\section{Trend}
\label{sec:trend}
\pending{Phase 2}

% ===========================================================================
\section{Seasonality}
\label{sec:seasonality}
\pending{Phase 1b, completed after Phase 2}

% ===========================================================================
\section{Regressors and gains}
\label{sec:regressors}
\pending{Phase 2}

% ===========================================================================
\section{Impulse response}
\label{sec:impulse}
\pending{Phase 4}

% ===========================================================================
\section{Uncertainty and validation}
\label{sec:uncertainty}
\pending{Phase 3}

% ===========================================================================
\section{What the wall temperature and the pyranometer buy}
\label{sec:ladder}
\pending{Phase 2b}

% ===========================================================================
\section{The monitor}
\label{sec:monitor}
\pending{Phase 5}

% ===========================================================================
\section{Outages as hypotheses}
\label{sec:outages}
\pending{Phase 6}

% ===========================================================================
\section{Verdict}
\label{sec:verdict}
\pending{Phase 7}

% ===========================================================================
\section{Limitations}
\label{sec:limitations}
\pending{Phase 7}

% ===========================================================================
\section{Run metadata}
\label{sec:metadata}
\pending{Phase 7}

\end{document}
```

- [x] **Step 4: Build twice and run the honesty test**

Run from `studies/05_greybox_monitoring/report/`: `pdflatex -interaction=nonstopmode greybox_monitoring_report.tex && pdflatex -interaction=nonstopmode greybox_monitoring_report.tex`. Expected: `greybox_monitoring_report.pdf` with 14 pending sections and a table of contents; no `!` error lines in the `.log`.
Run from `studies/`: `python 05_greybox_monitoring/tests/test_folder_honesty.py`. Expected: 3 pass, 2 skipped.

- [x] **Step 5: Commit** (remove the two `.gitkeep` files first)

```bash
git rm -q studies/05_greybox_monitoring/report/.gitkeep studies/05_greybox_monitoring/tests/.gitkeep
git add studies/05_greybox_monitoring/report/greybox_monitoring_report.tex studies/05_greybox_monitoring/report/greybox_monitoring_report.pdf studies/05_greybox_monitoring/tests/test_folder_honesty.py
git commit -m "docs(study05): report skeleton with every section pending, and the folder-honesty test"
```

> **Checkpoint 0 (O):** show the user the five capability results (or fallbacks), the graph refresh summary, and the built PDF. Approval opens Phase 1.

---

# Phase 1 · Data and the three regressor sets (spec D1, D2, D3, D13)

### Task 1.1 (S): `proxies.to_native_grid`

**Files:**
- Modify: `studies/shmlib/proxies.py` (append after `harmonise`)
- Create: `studies/shmlib/tests/test_proxies_grid.py`

**Interfaces:**
- Produces: `proxies.to_native_grid(frame, freq='20min', accumulations=('sr',)) -> pd.DataFrame` on a regular `freq` grid spanning the input, UTC index named `'datetime'`. Each column is interpolated linearly in time between consecutive native observations that are at most one native step apart (median index spacing per column, tolerance ×1.5); nothing is bridged across a longer gap. A column whose quantity prefix (`column.split('_')[0]`) is in `accumulations` is first shifted back by half its native step, because an hourly accumulation reported at H describes (H−1, H].

- [x] **Step 1: Write the failing tests**

```python
"""
Tests for the proxy grid helpers Study 05 adds to shmlib.proxies.

Run from studies/:  python shmlib/tests/test_proxies_grid.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import proxies  # noqa: E402


class TestToNativeGrid(unittest.TestCase):

    def test_hourly_ramp_is_interpolated_to_twenty_minutes(self):
        index = pd.date_range('2024-01-01', periods=4, freq='1h', tz='UTC')
        frame = pd.DataFrame({'tair_era5': [0.0, 3.0, 6.0, 9.0]}, index=index)
        out = proxies.to_native_grid(frame, freq='20min')
        self.assertEqual(out.index.freqstr, '20min')
        self.assertAlmostEqual(out.loc['2024-01-01 00:20', 'tair_era5'], 1.0)
        self.assertAlmostEqual(out.loc['2024-01-01 02:40', 'tair_era5'], 8.0)

    def test_a_gap_longer_than_one_native_step_is_not_bridged(self):
        index = pd.DatetimeIndex(['2024-01-01 00:00', '2024-01-01 01:00',
                                  '2024-01-01 04:00'], tz='UTC')
        frame = pd.DataFrame({'tair_gs': [0.0, 1.0, 4.0]}, index=index)
        out = proxies.to_native_grid(frame, freq='20min')
        self.assertTrue(np.isnan(out.loc['2024-01-01 02:00', 'tair_gs']))
        self.assertAlmostEqual(out.loc['2024-01-01 00:40', 'tair_gs'], 2.0 / 3.0)

    def test_an_accumulation_is_centred_half_a_step_earlier(self):
        index = pd.date_range('2024-06-01', periods=3, freq='1h', tz='UTC')
        frame = pd.DataFrame({'sr_era5': [0.0, 100.0, 200.0]}, index=index)
        out = proxies.to_native_grid(frame, freq='20min', accumulations=('sr',))
        # value 100 belongs to 00:30, so 00:40 sits a third of the way to 200
        self.assertAlmostEqual(out.loc['2024-06-01 00:40', 'sr_era5'], 100.0 + 100.0 / 3.0, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
```

- [x] **Step 2: Run to verify failure**

Run from `studies/`: `python shmlib/tests/test_proxies_grid.py`
Expected: `AttributeError: module 'shmlib.proxies' has no attribute 'to_native_grid'`.

- [x] **Step 3: Implement**

```python
def to_native_grid(frame, freq='20min', accumulations=('sr',)):
    """
    Bring hourly or half-hourly proxies onto the sensor's native grid.

    Each column is interpolated linearly in time between consecutive native
    observations, but only where those two observations are at most one
    native step apart, so a proxy outage is never bridged. A column whose
    quantity is an accumulation over the preceding native step (ERA5
    radiation) is first shifted back by half a step, since a value reported
    at H describes the interval (H-1, H] and belongs at its centre.

    Parameters
    ----------
    frame : pd.DataFrame
        Datetime-indexed proxies, UTC, one column per channel.
    freq : str, optional
        Target grid. Default ``'20min'``.
    accumulations : sequence of str, optional
        Quantity prefixes (``column.split('_')[0]``) reported as accumulations.
        Default ``('sr',)``.

    Returns
    -------
    pd.DataFrame
        On ``pd.date_range(start.ceil(freq), end.floor(freq), freq=freq)``,
        index named ``'datetime'``.
    """
    step = pd.Timedelta(freq)
    grid = pd.date_range(frame.index.min().ceil(freq),
                         frame.index.max().floor(freq), freq=freq,
                         tz=frame.index.tz)
    out = pd.DataFrame(index=grid)
    out.index.name = 'datetime'
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors='coerce').dropna()
        if series.size < 2:
            out[column] = np.nan
            continue
        native = pd.Series(series.index).diff().median()
        if column.split('_')[0] in accumulations:
            series.index = series.index - native / 2
        union = series.index.union(grid)
        dense = series.reindex(union).interpolate(method='time',
                                                  limit_area='inside')
        stamps = pd.Series(series.index, index=series.index)
        previous = stamps.reindex(union).ffill()
        following = stamps.reindex(union).bfill()
        bridged = (following - previous) > native * 1.5
        out[column] = dense.where(~bridged).reindex(grid)
    return out
```

- [x] **Step 4: Run the tests; expected: 3 PASS.** Then run the whole existing suite (Global Constraints) to confirm nothing else changed.

- [x] **Step 5: Commit**

```bash
git add studies/shmlib/proxies.py studies/shmlib/tests/test_proxies_grid.py
git commit -m "feat(shmlib): bring proxies onto the native grid without bridging their gaps"
```

### Task 1.2 (S): `proxies.fill_short_gaps`

**Files:**
- Modify: `studies/shmlib/proxies.py` (append after `to_native_grid`)
- Modify: `studies/shmlib/tests/test_proxies_grid.py` (add a class)

**Interfaces:**
- Produces: `proxies.fill_short_gaps(frame, columns, max_gap='2h', flag=True) -> pd.DataFrame`: copy of `frame` where, in each named column, a run of missing values whose bracketing observations are at most `max_gap` apart is filled by linear-in-time interpolation; with `flag=True` a boolean column `<column>_filled` marks the filled slots. Longer runs stay missing.

- [x] **Step 1: Write the failing tests** (append to `test_proxies_grid.py` before the `__main__` block)

```python
class TestFillShortGaps(unittest.TestCase):

    def _frame(self):
        index = pd.date_range('2024-01-01', periods=12, freq='20min', tz='UTC')
        values = np.arange(12, dtype=float)
        values[2] = np.nan            # one slot: 20 min gap
        values[5:10] = np.nan         # five slots: 1h40 between neighbours
        return pd.DataFrame({'tair_gs': values}, index=index)

    def test_gaps_within_the_limit_are_filled_and_flagged(self):
        out = proxies.fill_short_gaps(self._frame(), ['tair_gs'], max_gap='2h')
        self.assertAlmostEqual(out['tair_gs'].iloc[2], 2.0)
        self.assertAlmostEqual(out['tair_gs'].iloc[7], 7.0)
        self.assertTrue(out['tair_gs_filled'].iloc[2])
        self.assertFalse(out['tair_gs_filled'].iloc[3])

    def test_a_gap_beyond_the_limit_stays_missing(self):
        out = proxies.fill_short_gaps(self._frame(), ['tair_gs'], max_gap='1h')
        self.assertAlmostEqual(out['tair_gs'].iloc[2], 2.0)
        self.assertTrue(out['tair_gs'].iloc[5:10].isna().all())
        self.assertFalse(out['tair_gs_filled'].iloc[5:10].any())

    def test_no_flag_column_when_flag_is_false(self):
        out = proxies.fill_short_gaps(self._frame(), ['tair_gs'], flag=False)
        self.assertNotIn('tair_gs_filled', out.columns)
```

- [x] **Step 2: Run; expected AttributeError on `fill_short_gaps`.**

- [x] **Step 3: Implement**

```python
def fill_short_gaps(frame, columns, max_gap='2h', flag=True):
    """
    Fill short regressor dropouts by linear interpolation, and say where.

    A regressor is a measured, smooth driver rather than the quantity being
    judged, so a dropout of a few slots may be bridged from its neighbours
    without inventing structure. The target is never filled by this or any
    other function. Runs whose bracketing observations are more than
    ``max_gap`` apart are left missing.

    Parameters
    ----------
    frame : pd.DataFrame
        Datetime-indexed frame on a regular grid.
    columns : sequence of str
        Columns to fill.
    max_gap : str or pd.Timedelta, optional
        Longest bracket (last observation before to first after) that may be
        bridged. Default ``'2h'``.
    flag : bool, optional
        Add ``<column>_filled`` booleans marking filled slots. Default ``True``.

    Returns
    -------
    pd.DataFrame
        A copy of ``frame`` with the fills applied.
    """
    limit = pd.Timedelta(max_gap)
    out = frame.copy()
    for column in columns:
        series = pd.to_numeric(out[column], errors='coerce')
        observed = series.dropna()
        stamps = pd.Series(observed.index, index=observed.index)
        previous = stamps.reindex(series.index).ffill()
        following = stamps.reindex(series.index).bfill()
        short = series.isna() & ((following - previous) <= limit)
        filled = series.interpolate(method='time', limit_area='inside')
        out[column] = series.where(~short, filled)
        if flag:
            out[column + '_filled'] = short.fillna(False).astype(bool)
    return out
```

- [x] **Step 4: Run; expected 6 PASS in the file. Run the full existing suite.**

- [x] **Step 5: Commit**

```bash
git add studies/shmlib/proxies.py studies/shmlib/tests/test_proxies_grid.py
git commit -m "feat(shmlib): fill and flag regressor dropouts up to a stated length"
```

### Task 1.3 (S implements, O reviews the printed summary): Movement 1 in the notebook — load, check clocks, build the three sets, export `GM_01`–`GM_03` and `GM_F01`

**Files:**
- Modify: `studies/05_greybox_monitoring/greybox_monitoring_study.py` — (a) the Regressor sets parameter group; (b) new cells appended after the final "## Movements" markdown cell.
- Modify: `studies/shmlib/figures.py` — add `plot_regressor_sets`.
- Modify: `studies/shmlib/tests/test_shmlib.py` — add one test for `plot_regressor_sets`.

**Interfaces:**
- Consumes: `proxies.load_response`, `load_sensor_forcings`, `join_eras`, `load_ground_station`, `load_era5`, `harmonise`, `to_native_grid`, `fill_short_gaps`; `quality.clock_check`; `coupling.thermal_operator`; `prediction.gap_inventory`; `tables.write_table`.
- Produces: in the notebook namespace, `target` (Series, 20 min, UTC), `sets` (dict `set_name -> DataFrame` with columns `tair`, `rh`, `sr` on the 20-min grid, radiation already delayed), `frame` (DataFrame: `y` plus every set's columns suffixed `_<set>`); files `GM_01_window_coverage.csv/.tex`, `GM_02_gap_inventory.csv/.tex`, `GM_03_clock_check.csv/.tex`, `GM_F01_regressor_sets.png/.svg`.

- [x] **Step 1: Parameter-cell edits.** In the "Parameters · Regressor sets" group: copy `STR_MAP_CURRENT` and `STR_MAP_LEGACY` verbatim from `04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.py` lines 253–268 (the block-to-quantity maps for the current and legacy eras) and add a guidance bullet for each. Change the `'str'` entry of `REGRESSOR_SETS` to `{'tair': 'tair_str', 'rh': 'rh_str', 'sr': 'sr_gs'}`, since `join_eras` returns `_str`-suffixed columns.

- [x] **Step 2: Add `figures.plot_regressor_sets`** (append to `shmlib/figures.py`)

```python
def plot_regressor_sets(frame, target, sets, title='', tick_years=1,
                        save_path=None, filename=None):
    """
    The record and the three regressor sets on one clock, gaps as gaps.

    Four panels: the target, then one panel per role with every set's
    version of it overlaid. Colour is the role's identity colour; the sets
    are told apart by line style, since colour is already spent on identity.

    Parameters
    ----------
    frame : pd.DataFrame
        Datetime-indexed frame holding ``target`` and every set's columns.
    target : str
        Column of the response.
    sets : dict
        ``{set_name: {role: column}}``; roles are ``'tair'``, ``'rh'``, ``'sr'``.
    title : str, optional
    tick_years : int, optional
        Years between x ticks. Default ``1``.
    save_path, filename : str or None, optional
        Passed to ``viz.finish``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    roles = ['tair', 'rh', 'sr']
    styles = {'str': '-', 'gs': '--', 'era5': ':'}
    fig, axes = plt.subplots(1 + len(roles), 1, sharex=True,
                             figsize=viz.figsize(viz.FIGURE_WIDTH,
                                                 1.15 * (1 + len(roles))))
    colour, width = viz.channel_style(target, 0.6)
    axes[0].plot(frame.index, frame[target], color=colour, linewidth=width)
    axes[0].set_ylabel(viz.channel_unit(target))
    for ax, role in zip(axes[1:], roles):
        handles = []
        for name, mapping in sets.items():
            column = mapping[role]
            colour, width = viz.channel_style(role, 0.6)
            line, = ax.plot(frame.index, frame[column], color=colour,
                            linewidth=width, linestyle=styles.get(name, '-'),
                            label=name)
            handles.append(line)
        ax.set_ylabel(viz.channel_unit(role))
    for ax in axes:
        viz.format_spines(ax)
    axes[-1].legend(handles=handles, fontsize='small', ncol=len(sets),
                    loc='upper center', bbox_to_anchor=(0.5, -0.45),
                    frameon=False)
    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig
```

Test to add to `shmlib/tests/test_shmlib.py` (in the `TestViz`-style class, or a new `TestFiguresStudy05` class):

```python
    def test_plot_regressor_sets_draws_four_panels(self):
        index = pd.date_range('2024-01-01', periods=48, freq='1h', tz='UTC')
        frame = pd.DataFrame({
            'y': np.sin(np.arange(48) / 4.0), 'tair_str': 10.0, 'rh_str': 50.0,
            'sr_gs': 100.0, 'tair_gs': 11.0, 'rh_gs': 51.0, 'tair_era5': 9.0,
            'rh_era5': 49.0, 'sr_era5': 90.0}, index=index)
        sets = {'str': {'tair': 'tair_str', 'rh': 'rh_str', 'sr': 'sr_gs'},
                'gs': {'tair': 'tair_gs', 'rh': 'rh_gs', 'sr': 'sr_gs'},
                'era5': {'tair': 'tair_era5', 'rh': 'rh_era5', 'sr': 'sr_era5'}}
        fig = figures.plot_regressor_sets(frame, 'y', sets)
        self.assertEqual(len(fig.axes), 4)
        plt.close(fig)
```

Run `python shmlib/tests/test_shmlib.py`; expected PASS.

- [x] **Step 3: Append Movement 1 cells to the notebook** (after the "## Movements" cell)

```python
# %% [markdown]
# ## Movement 1 · The record and the three regressor sets
#
# Study 1's product is read at its native cadence, the two external sources
# are brought onto the same grid, each source's clock is checked before
# anything is joined, and the three regressor sets of the design (D2) are
# assembled with Study 3's operator applied to radiation (D3). Nothing here
# fills the target; regressor dropouts up to `REGRESSOR_FILL_MAX_GAP` are
# filled and flagged (D13).

# %%
target, target_provenance = proxies.load_response(
    ARCHIVE_CSV, column=TARGET_COLUMN, spike_column=SPIKE_COLUMN,
    honour_spike=True, freq=NATIVE_FREQ, tz=site.SITE_TZ, min_count=1)
target = target.loc[WINDOW_START:WINDOW_END]

sensor_current, _ = proxies.load_sensor_forcings(
    ARCHIVE_CSV, column_map=STR_MAP_CURRENT, freq=NATIVE_FREQ,
    tz=site.SITE_TZ, honour_suspect=True, min_count=1)
sensor_legacy, _ = proxies.load_sensor_forcings(
    ARCHIVE_CSV, column_map=STR_MAP_LEGACY, freq=NATIVE_FREQ,
    tz=site.SITE_TZ, honour_suspect=True, min_count=1)
sensor = proxies.join_eras([sensor_current, sensor_legacy])

station_hourly, _ = proxies.load_ground_station(STATION_CSV)
era5_hourly, _ = proxies.load_era5(ERA5_CSV)
print(f'target {target.notna().sum():,} accepted slots of {len(target):,}; '
      f'station {len(station_hourly):,} h; ERA5 {len(era5_hourly):,} h')

# %% [markdown]
# ### The clock of each source
#
# Study 2's two-sided test: radiation against computed solar noon, and each
# source against ERA5 by cross-correlation. Run on the hourly grid, where
# the test was designed, before any upsampling.

# %%
hourly = proxies.harmonise(
    [proxies.load_sensor_forcings(ARCHIVE_CSV, column_map=STR_MAP_CURRENT,
                                  honour_suspect=True)[0],
     station_hourly, era5_hourly], freq=site.ANALYSIS_FREQ)
clock = quality.clock_check(hourly, 'sr')
display(clock)
clock.to_csv(OUTPUT_DIR / 'GM_03_clock_check.csv', index=False)
tables.write_table(clock, str(OUTPUT_DIR / 'GM_03_body.tex'),
                   [(c, tables.texttt if clock[c].dtype == object else '.2f')
                    for c in clock.columns])

# %% [markdown]
# ### Onto the native grid, and the regressor sets

# %%
station = proxies.to_native_grid(station_hourly, freq=NATIVE_FREQ,
                                 accumulations=())
era5 = proxies.to_native_grid(
    era5_hourly, freq=NATIVE_FREQ,
    accumulations=('sr',) if ERA5_SR_IS_ACCUMULATION else ())
record = proxies.harmonise([sensor, station, era5, target.to_frame('y')],
                           freq=NATIVE_FREQ).loc[WINDOW_START:WINDOW_END]

sets = {}
for name, mapping in REGRESSOR_SETS.items():
    block = record[[mapping['tair'], mapping['rh'], mapping['sr']]].copy()
    block.columns = ['tair', 'rh', 'sr']
    block = proxies.fill_short_gaps(block, ['tair', 'rh', 'sr'],
                                    max_gap=REGRESSOR_FILL_MAX_GAP, flag=True)
    # Study 3's operator: radiation delayed by RADIATION_DELAY_H, no inertia.
    # thermal_operator counts the delay in samples of dt_hours; confirm the
    # docstring before trusting the conversion below and assert it here.
    slots = int(round(RADIATION_DELAY_H / (pd.Timedelta(NATIVE_FREQ)
                                            / pd.Timedelta(hours=1))))
    delayed = coupling.thermal_operator(block['sr'], delay=slots, tau=0.0,
                                        dt_hours=pd.Timedelta(NATIVE_FREQ)
                                        / pd.Timedelta(hours=1))
    assert delayed.dropna().iloc[:5].equals(
        block['sr'].shift(slots).dropna().iloc[:5]), 'delay unit mismatch'
    block['sr'] = delayed
    sets[name] = block

frame = record[['y']].copy()
for name, block in sets.items():
    for column in ['tair', 'rh', 'sr']:
        frame[f'{column}_{name}'] = block[column]
        frame[f'{column}_{name}_filled'] = block[f'{column}_filled']

# %% [markdown]
# ### Coverage and the anatomy of the target's gaps

# %%
rows = []
for name, block in sets.items():
    for column in ['tair', 'rh', 'sr']:
        rows.append({'set': name, 'role': column,
                     'accepted': int(block[column].notna().sum()),
                     'filled': int(block[f'{column}_filled'].sum()),
                     'coverage': float(block[column].notna().mean())})
rows.append({'set': 'target', 'role': TARGET_COLUMN,
             'accepted': int(frame['y'].notna().sum()), 'filled': 0,
             'coverage': float(frame['y'].notna().mean())})
coverage = pd.DataFrame(rows)
display(coverage)
coverage.to_csv(OUTPUT_DIR / 'GM_01_window_coverage.csv', index=False)
tables.write_table(coverage, str(OUTPUT_DIR / 'GM_01_body.tex'),
                   [('set', tables.texttt), ('role', tables.texttt),
                    ('accepted', ',d'), ('filled', ',d'),
                    ('coverage', tables.percent)])

gaps = prediction.gap_inventory(frame['y'], freq=NATIVE_FREQ)
gap_classes = (gaps.groupby('gap_class', sort=False)
               .agg(gaps=('n_slots', 'size'), hours=('duration_h', 'sum'))
               .reset_index())
display(gap_classes)
gaps.to_csv(OUTPUT_DIR / 'GM_02_gap_inventory.csv', index=False)
tables.write_table(gap_classes, str(OUTPUT_DIR / 'GM_02_body.tex'),
                   [('gap_class', tables.texttt), ('gaps', ',d'),
                    ('hours', ',.0f')])

# %%
figures.plot_regressor_sets(
    frame, 'y', {n: {r: f'{r}_{n}' for r in ['tair', 'rh', 'sr']}
                 for n in sets},
    title='The record and the three regressor sets',
    save_path=str(OUTPUT_DIR), filename='GM_F01_regressor_sets')
```

- [x] **Step 4: Sync and execute**

From `studies/05_greybox_monitoring/`: `jupytext --to ipynb greybox_monitoring_study.py`, then `jupyter nbconvert --to notebook --execute greybox_monitoring_study.ipynb --output-dir /tmp/gm05 --ExecutePreprocessor.timeout=3600`. Read back the printed lines from `/tmp/gm05/greybox_monitoring_study.ipynb` (grep `"text"` cells for `target`, `Traceback`). Expected: no Traceback; `outputs/` holds `GM_01`–`GM_03` CSV and body files and `GM_F01` PNG/SVG. Copy the executed notebook back over `greybox_monitoring_study.ipynb`.

- [x] **Step 5: Honesty test and commit**

Run from `studies/`: `python 05_greybox_monitoring/tests/test_folder_honesty.py` (still 3 pass, 2 skip).

```bash
git add studies/05_greybox_monitoring/greybox_monitoring_study.py studies/05_greybox_monitoring/greybox_monitoring_study.ipynb studies/shmlib/figures.py studies/shmlib/tests/test_shmlib.py
git commit -m "feat(study05): load the record, check the clocks and build the three regressor sets"
```

### Task 1.4 (O): Report sections 1–3 from the Phase 1 artefacts; status to "In progress"

**Files:**
- Modify: `studies/05_greybox_monitoring/report/greybox_monitoring_report.tex` — replace the `\pending{}` lines of `sec:intro`, `sec:record`, `sec:method`.
- Modify: `studies/05_greybox_monitoring/README.md` — status line; `studies/README.md` — study 5 row status.

- [x] **Step 1: Write §1 Introduction** (full prose, from spec §1 and §2): the operational question; what is inherited from Studies 1–4 and not re-litigated; compensation as given; forecasting out of scope; the five questions.

- [x] **Step 2: Write §2 The record and the three regressor sets**: window and grid; Table with `\input{../outputs/GM_01_body.tex}` (columns set, role, accepted, filled, coverage); the gap anatomy with `\input{../outputs/GM_02_body.tex}`; the clock check with `\input{../outputs/GM_03_body.tex}`; `\includegraphics{GM_F01_regressor_sets.png}` with a caption stating that gaps are gaps and radiation in the on-structure set is borrowed from the station. Every number quoted comes from those three CSVs.

- [x] **Step 3: Write §3 Method**: one subsection per spec decision D1–D14, in the spec's order, each stating the decision, the evidence and the alternative rejected. Subsections for decisions whose results come later (D6 onward) state the decision only; their results are written in their own sections.

- [x] **Step 4: Build twice, run the honesty test, update statuses**

README status line → `Status: **in progress — Phase 1 complete (record, clocks, regressor sets).**`; `studies/README.md` study 5 row status → `In progress`.

- [x] **Step 5: Commit**

```bash
git add studies/05_greybox_monitoring/report studies/05_greybox_monitoring/README.md studies/README.md
git commit -m "docs(study05): write the introduction, the record and the method sections of the report"
```

> **Checkpoint 1 (O):** coverage per set (`GM_01`), clock check clean (`GM_03`), the PDF with three sections written. Approval opens Phase 1b.

---

# Phase 1b · Harmonic diagnostics before modelling (spec D6)

### Task 1b.1 (S): `monitoring.daily_harmonic`

**Files:**
- Modify: `studies/shmlib/monitoring.py` (append)
- Create: `studies/05_greybox_monitoring/tests/test_harmonics.py`

**Interfaces:**
- Produces: `monitoring.daily_harmonic(series, min_slots=60, period_hours=24.0) -> pd.DataFrame` indexed by calendar day (UTC midnight), columns `amplitude`, `phase_h` (hour of the 24-h harmonic's maximum, in [0, 24)), `amplitude_12h`, `n`, `r2`. A day with fewer than `min_slots` finite samples is absent. Least-squares fit of `a0 + a1 cos ωt + b1 sin ωt + a2 cos 2ωt + b2 sin 2ωt`, ω = 2π/period, t in hours since midnight.

- [x] **Step 1: Write the failing test**

```python
"""
Tests for the harmonic diagnostics of Study 05 (spec D6).

Run from studies/:  python 05_greybox_monitoring/tests/test_harmonics.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import coupling, monitoring, prediction  # noqa: E402


def _days(n_days, amplitude=10.0, peak_hour=14.0, freq='20min', seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range('2024-01-01', periods=n_days * 72, freq=freq, tz='UTC')
    hours = index.hour + index.minute / 60.0
    values = amplitude * np.cos(2 * np.pi * (hours - peak_hour) / 24.0)
    return pd.Series(values + rng.normal(0, 0.1, len(index)), index=index)


class TestDailyHarmonic(unittest.TestCase):

    def test_amplitude_and_peak_hour_are_recovered(self):
        out = monitoring.daily_harmonic(_days(5), min_slots=60)
        self.assertEqual(len(out), 5)
        self.assertTrue(np.allclose(out['amplitude'], 10.0, atol=0.2))
        self.assertTrue(np.allclose(out['phase_h'], 14.0, atol=0.2))

    def test_a_short_day_is_dropped(self):
        series = _days(3)
        series.iloc[72:72 + 20] = np.nan   # second day keeps 52 of 72 slots
        out = monitoring.daily_harmonic(series, min_slots=60)
        self.assertEqual(len(out), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
```

- [x] **Step 2: Run; expected AttributeError on `daily_harmonic`.**

- [x] **Step 3: Implement**

```python
def daily_harmonic(series, min_slots=60, period_hours=24.0):
    """
    Amplitude and phase of the daily cycle, one row per calendar day.

    A 24-hour harmonic and its 12-hour companion are fitted to each day by
    least squares. The amplitude of the 24-hour term is the size of the
    daily swing; its phase is the hour at which that term peaks. On a
    residual these two series are the daily chart's statistics (spec D10);
    on the diurnal band of the target they are the measured daily cycle the
    harmonic diagnostic fits against day of year (spec D6).

    Parameters
    ----------
    series : pd.Series
        Signal indexed by UTC timestamp on a regular sub-daily grid.
    min_slots : int, optional
        Fewest finite samples a day needs to be fitted. Default ``60``.
    period_hours : float, optional
        Period of the fundamental. Default ``24.0``.

    Returns
    -------
    pd.DataFrame
        Indexed by day (UTC midnight): ``amplitude``, ``phase_h`` in
        ``[0, period_hours)``, ``amplitude_12h``, ``n``, ``r2``.
    """
    values = pd.to_numeric(series, errors='coerce')
    index = pd.DatetimeIndex(values.index)
    omega = 2.0 * np.pi / float(period_hours)
    rows = []
    for day, chunk in values.groupby(index.floor('D')):
        chunk = chunk.dropna()
        if len(chunk) < int(min_slots):
            continue
        t = (chunk.index.hour + chunk.index.minute / 60.0
             + chunk.index.second / 3600.0).to_numpy(dtype=float)
        design = np.column_stack([
            np.ones_like(t), np.cos(omega * t), np.sin(omega * t),
            np.cos(2 * omega * t), np.sin(2 * omega * t)])
        y = chunk.to_numpy(dtype=float)
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ coef
        total = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float(((y - fitted) ** 2).sum()) / total if total > 0 else np.nan
        a1, b1, a2, b2 = coef[1], coef[2], coef[3], coef[4]
        rows.append({
            'day': day,
            'amplitude': float(np.hypot(a1, b1)),
            'phase_h': float((np.arctan2(b1, a1) / omega) % period_hours),
            'amplitude_12h': float(np.hypot(a2, b2)),
            'n': int(len(chunk)),
            'r2': r2,
        })
    out = pd.DataFrame(rows, columns=['day', 'amplitude', 'phase_h',
                                      'amplitude_12h', 'n', 'r2'])
    return out.set_index('day')
```

- [x] **Step 4: Run; expected 2 PASS. Run `python shmlib/tests/test_monitoring.py`; expected unchanged.**

- [x] **Step 5: Commit**

```bash
git add studies/shmlib/monitoring.py studies/05_greybox_monitoring/tests/test_harmonics.py
git commit -m "feat(shmlib): fit the daily cycle's amplitude and phase day by day"
```

### Task 1b.2 (S): `coupling.annual_modulation` and `coupling.evaluate_modulation`

**Files:**
- Modify: `studies/shmlib/coupling.py` (append)
- Modify: `studies/05_greybox_monitoring/tests/test_harmonics.py`

**Interfaces:**
- Produces: `coupling.annual_modulation(daily_series, harmonics=(1, 2), holdout='year') -> (table, fit)`. `table`: DataFrame with columns `order`, `holdout_mse`, `chosen` (bool), one row per candidate order; `fit`: dict `{'order': K, 'coef': np.ndarray of length 1 + 2K, 'period_days': 365.25, 'n': int}` for the chosen order (lowest leave-one-year-out MSE). `coupling.evaluate_modulation(fit, index) -> pd.Series` evaluates the fitted annual Fourier series at each timestamp's day of year.

- [x] **Step 1: Write the failing test** (append class to `test_harmonics.py`)

```python
class TestAnnualModulation(unittest.TestCase):

    def _daily(self, years=4, seed=1):
        rng = np.random.default_rng(seed)
        days = pd.date_range('2019-01-01', periods=365 * years, freq='D', tz='UTC')
        doy = days.dayofyear.to_numpy()
        truth = 20.0 + 8.0 * np.cos(2 * np.pi * (doy - 200) / 365.25)
        return pd.Series(truth + rng.normal(0, 1.0, len(days)), index=days)

    def test_one_harmonic_is_chosen_and_the_peak_day_is_recovered(self):
        table, fit = coupling.annual_modulation(self._daily(), harmonics=(1, 2))
        self.assertEqual(fit['order'], 1)
        self.assertTrue(table.loc[table['chosen'], 'order'].item() == 1)
        year = pd.date_range('2023-01-01', periods=365, freq='D', tz='UTC')
        curve = coupling.evaluate_modulation(fit, year)
        self.assertAlmostEqual(curve.idxmax().dayofyear, 200, delta=6)
        self.assertAlmostEqual(curve.max() - curve.min(), 16.0, delta=1.0)
```

- [x] **Step 2: Run; expected AttributeError.**

- [x] **Step 3: Implement**

```python
def _annual_design(index, order, period_days=365.25):
    doy = pd.DatetimeIndex(index).dayofyear.to_numpy(dtype=float)
    columns = [np.ones_like(doy)]
    for k in range(1, int(order) + 1):
        angle = 2.0 * np.pi * k * doy / period_days
        columns += [np.cos(angle), np.sin(angle)]
    return np.column_stack(columns)


def annual_modulation(daily_series, harmonics=(1, 2), holdout='year'):
    """
    How a daily statistic moves through the year, as a low-order Fourier fit.

    Fits ``a0 + sum_k (a_k cos + b_k sin)(2 pi k doy / 365.25)`` to a daily
    series for each candidate order, scores each by leave-one-year-out mean
    squared error, and keeps the lowest. The chosen fit is what the smooth
    conditional daily term consumes as its weight curve (spec D6, D7).

    Parameters
    ----------
    daily_series : pd.Series
        One value per day, indexed by day.
    harmonics : sequence of int, optional
        Candidate Fourier orders. Default ``(1, 2)``.
    holdout : {'year'}, optional
        Hold-out unit for the order choice. Default ``'year'``.

    Returns
    -------
    (pd.DataFrame, dict)
        ``table`` with ``order``, ``holdout_mse``, ``chosen``; ``fit`` with
        ``order``, ``coef``, ``period_days`` and ``n`` for the chosen order.
    """
    values = pd.to_numeric(daily_series, errors='coerce').dropna()
    index = pd.DatetimeIndex(values.index)
    years = index.year
    rows, coefs = [], {}
    for order in harmonics:
        errors = []
        for held in np.unique(years):
            train, test = years != held, years == held
            if train.sum() < 3 * (2 * order + 1) or test.sum() == 0:
                continue
            coef, _, _, _ = np.linalg.lstsq(
                _annual_design(index[train], order), values.to_numpy()[train],
                rcond=None)
            predicted = _annual_design(index[test], order) @ coef
            errors.append(float(((values.to_numpy()[test] - predicted) ** 2).mean()))
        coef, _, _, _ = np.linalg.lstsq(_annual_design(index, order),
                                        values.to_numpy(), rcond=None)
        coefs[order] = coef
        rows.append({'order': int(order),
                     'holdout_mse': float(np.mean(errors)) if errors else np.nan})
    table = pd.DataFrame(rows)
    chosen = int(table.loc[table['holdout_mse'].idxmin(), 'order'])
    table['chosen'] = table['order'] == chosen
    fit = {'order': chosen, 'coef': coefs[chosen], 'period_days': 365.25,
           'n': int(values.size)}
    return table, fit


def evaluate_modulation(fit, index):
    """
    The fitted annual modulation at each timestamp's day of year.

    Parameters
    ----------
    fit : dict
        Second return value of :func:`annual_modulation`.
    index : pd.DatetimeIndex

    Returns
    -------
    pd.Series
        Indexed by ``index``.
    """
    design = _annual_design(index, fit['order'], fit['period_days'])
    return pd.Series(design @ fit['coef'], index=pd.DatetimeIndex(index))
```

- [x] **Step 4: Run; expected PASS. Run `python 03_thermomechanical_response/tests/test_shmlib_study03.py`; unchanged.**

- [x] **Step 5: Commit**

```bash
git add studies/shmlib/coupling.py studies/05_greybox_monitoring/tests/test_harmonics.py
git commit -m "feat(shmlib): fit and evaluate the annual modulation of a daily statistic"
```

### Task 1b.3 (S): `coupling.cycle_surface_rank`

**Files:**
- Modify: `studies/shmlib/coupling.py` (append)
- Modify: `studies/05_greybox_monitoring/tests/test_harmonics.py`

**Interfaces:**
- Produces: `coupling.cycle_surface_rank(series, doy_bins=52, freq='20min') -> dict` with `'table'` (DataFrame: `component`, `singular_value`, `variance_share`, `cumulative_share`), `'daily_shapes'` (DataFrame, index = slot-of-day as hour float, one column per component), `'annual_weights'` (DataFrame, index = bin centre day of year, one column per component). Built from the mean of `series` in each (day-of-year bin, slot-of-day) cell; missing cells take the slot's mean before the SVD.

- [x] **Step 1: Write the failing test**

```python
class TestCycleSurfaceRank(unittest.TestCase):

    def test_a_single_modulated_shape_has_rank_one(self):
        index = pd.date_range('2019-01-01', periods=72 * 365 * 3, freq='20min', tz='UTC')
        hours = index.hour + index.minute / 60.0
        doy = index.dayofyear.to_numpy()
        envelope = 1.0 + 0.5 * np.cos(2 * np.pi * (doy - 200) / 365.25)
        series = pd.Series(envelope * np.cos(2 * np.pi * (hours - 14) / 24.0), index=index)
        out = coupling.cycle_surface_rank(series, doy_bins=52)
        self.assertGreater(out['table']['variance_share'].iloc[0], 0.98)
        self.assertEqual(out['daily_shapes'].shape[0], 72)
        self.assertEqual(out['annual_weights'].shape[0], 52)
```

- [x] **Step 2: Run; expected AttributeError.**

- [x] **Step 3: Implement**

```python
def cycle_surface_rank(series, doy_bins=52, freq='20min'):
    """
    How many weighted daily shapes the daily-by-annual surface needs.

    The series is averaged into a surface of day-of-year bin by slot of day
    and decomposed by singular values. The share of variance the leading
    components carry says whether one daily shape with a scalar envelope, or
    two blended shapes, or more, reproduce how the daily cycle changes
    through the year (spec D6).

    Parameters
    ----------
    series : pd.Series
        Signal on a regular sub-daily grid, UTC. Pass the diurnal band.
    doy_bins : int, optional
        Number of day-of-year bins. Default ``52``.
    freq : str, optional
        Grid spacing, used to count slots per day. Default ``'20min'``.

    Returns
    -------
    dict
        ``'table'``, ``'daily_shapes'``, ``'annual_weights'`` as documented in
        the plan.
    """
    values = pd.to_numeric(series, errors='coerce')
    index = pd.DatetimeIndex(values.index)
    slots_per_day = int(pd.Timedelta('1D') / pd.Timedelta(freq))
    slot = ((index.hour * 60 + index.minute) // (24 * 60 // slots_per_day)).to_numpy()
    doy_bin = np.minimum((index.dayofyear.to_numpy() - 1) * doy_bins // 366,
                         doy_bins - 1)
    surface = (pd.DataFrame({'v': values.to_numpy(), 'bin': doy_bin, 'slot': slot})
               .groupby(['bin', 'slot'])['v'].mean().unstack('slot')
               .reindex(index=range(doy_bins), columns=range(slots_per_day)))
    surface = surface.apply(lambda col: col.fillna(col.mean()), axis=0).fillna(0.0)
    matrix = surface.to_numpy(dtype=float)
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    share = s ** 2 / float((s ** 2).sum())
    table = pd.DataFrame({'component': np.arange(1, len(s) + 1),
                          'singular_value': s, 'variance_share': share,
                          'cumulative_share': np.cumsum(share)})
    hours = np.arange(slots_per_day) * 24.0 / slots_per_day
    centres = (np.arange(doy_bins) + 0.5) * 366.0 / doy_bins
    keep = min(3, len(s))
    shapes = pd.DataFrame(vt[:keep].T, index=hours,
                          columns=[f'component_{k + 1}' for k in range(keep)])
    weights = pd.DataFrame(u[:, :keep] * s[:keep], index=centres,
                           columns=[f'component_{k + 1}' for k in range(keep)])
    return {'table': table, 'daily_shapes': shapes, 'annual_weights': weights}
```

- [x] **Step 4: Run; expected PASS.** (The synthetic test has no gaps, so the fill branch is exercised only by the notebook.)

- [x] **Step 5: Commit**

```bash
git add studies/shmlib/coupling.py studies/05_greybox_monitoring/tests/test_harmonics.py
git commit -m "feat(shmlib): rank the daily-by-annual surface of a cycle by singular values"
```

### Task 1b.4 (S): `prediction.seasonal_weights`

**Files:**
- Modify: `studies/shmlib/prediction.py` (append)
- Modify: `studies/05_greybox_monitoring/tests/test_harmonics.py`

**Interfaces:**
- Produces: `prediction.seasonal_weights(index, modulation=None, peak_doy=196) -> pd.DataFrame` with columns `summer_w`, `winter_w` in [0, 1] summing to one. With `modulation` (the `fit` dict from `coupling.annual_modulation`) the summer weight is the fitted curve min-max normalised over one calendar year of days of year; with `modulation=None` it is the cosine fallback `½(1 − cos(2π(doy − (peak_doy − 182))/365.25))`, i.e. maximum at `peak_doy` (196 = 15 July).

- [x] **Step 1: Write the failing test**

```python
class TestSeasonalWeights(unittest.TestCase):

    def test_weights_sum_to_one_and_peak_in_july_by_default(self):
        index = pd.date_range('2024-01-01', periods=366, freq='D', tz='UTC')
        out = prediction.seasonal_weights(index)
        self.assertTrue(np.allclose(out['summer_w'] + out['winter_w'], 1.0))
        self.assertEqual(out['summer_w'].idxmax().month, 7)
        self.assertGreaterEqual(out['summer_w'].min(), 0.0)
        self.assertLessEqual(out['summer_w'].max(), 1.0)

    def test_a_measured_modulation_is_normalised_to_the_unit_interval(self):
        fit = {'order': 1, 'coef': np.array([20.0, -8.0, 0.0]),
               'period_days': 365.25, 'n': 1000}
        index = pd.date_range('2024-01-01', periods=366, freq='D', tz='UTC')
        out = prediction.seasonal_weights(index, modulation=fit)
        self.assertAlmostEqual(out['summer_w'].max(), 1.0, places=6)
        self.assertAlmostEqual(out['summer_w'].min(), 0.0, places=6)
```

- [x] **Step 2: Run; expected AttributeError.**

- [x] **Step 3: Implement**

```python
def seasonal_weights(index, modulation=None, peak_doy=196):
    """
    Condition columns for the smoothly weighted daily seasonality (spec D7).

    Two weights that sum to one at every timestamp and vary only with the
    calendar. With a measured annual modulation the summer weight is that
    curve min-max normalised over one year of days; without one it is a
    cosine peaking at ``peak_doy``.

    Parameters
    ----------
    index : pd.DatetimeIndex
    modulation : dict or None, optional
        ``fit`` from :func:`shmlib.coupling.annual_modulation`. Default
        ``None`` (cosine fallback).
    peak_doy : int, optional
        Day of year of the fallback cosine's maximum. Default ``196``.

    Returns
    -------
    pd.DataFrame
        ``summer_w`` and ``winter_w`` indexed by ``index``.
    """
    from shmlib import coupling as _coupling

    index = pd.DatetimeIndex(index)
    if modulation is None:
        doy = index.dayofyear.to_numpy(dtype=float)
        summer = 0.5 * (1.0 - np.cos(2.0 * np.pi * (doy - (peak_doy - 182.625))
                                     / 365.25))
    else:
        year = pd.date_range('2001-01-01', periods=365, freq='D')
        curve = _coupling.evaluate_modulation(modulation, year)
        low, high = float(curve.min()), float(curve.max())
        raw = _coupling.evaluate_modulation(modulation, index).to_numpy()
        summer = (raw - low) / (high - low) if high > low else np.full_like(raw, 0.5)
    summer = np.clip(summer, 0.0, 1.0)
    return pd.DataFrame({'summer_w': summer, 'winter_w': 1.0 - summer},
                        index=index)
```

- [x] **Step 4: Run; expected PASS. Run Study 04's `test_prediction.py`, `test_gaps.py`, `test_decomposition.py`; unchanged.**

- [x] **Step 5: Commit**

```bash
git add studies/shmlib/prediction.py studies/05_greybox_monitoring/tests/test_harmonics.py
git commit -m "feat(shmlib): calendar weights for the smoothly conditioned daily seasonality"
```

### Task 1b.5 (S): `figures.plot_harmonic_diagnostics`

**Files:**
- Modify: `studies/shmlib/figures.py` (append)
- Modify: `studies/shmlib/tests/test_shmlib.py` (one test)

**Interfaces:**
- Consumes: `period_scan` tables (two: target and residual), `daily_harmonic` frames (two), `annual_modulation` fits (amplitude and phase), `cycle_surface_rank` output.
- Produces: `figures.plot_harmonic_diagnostics(scans, dailies, fits, surface, title='', save_path=None, filename=None) -> Figure` with four panels: (1) certified periods per series as vertical markers on a log-period axis with their power; (2) daily amplitude by day of year (points, light) with the fitted annual curve for each series; (3) daily phase by day of year with its fit; (4) singular-value shares of the surface as bars. `scans`, `dailies`, `fits` are dicts keyed by series name (`'target'`, `'residual'`); `fits[name]` is `{'amplitude': fit, 'phase': fit}`.

- [x] **Step 1: Write the test** (append to `test_shmlib.py`)

```python
    def test_plot_harmonic_diagnostics_draws_four_panels(self):
        days = pd.date_range('2019-01-01', periods=730, freq='D', tz='UTC')
        daily = pd.DataFrame({'amplitude': 10 + 5 * np.cos(2 * np.pi * days.dayofyear / 365.25),
                              'phase_h': 14.0 + np.zeros(len(days))}, index=days)
        scan = pd.DataFrame({'rank': [1, 2], 'period_days': [365.0, 1.0],
                             'power': [1.0, 0.6]})
        _, fit_a = coupling.annual_modulation(daily['amplitude'])
        _, fit_p = coupling.annual_modulation(daily['phase_h'])
        surface = {'table': pd.DataFrame({'component': [1, 2, 3],
                                          'singular_value': [3.0, 1.0, 0.1],
                                          'variance_share': [0.89, 0.10, 0.01],
                                          'cumulative_share': [0.89, 0.99, 1.0]})}
        fig = figures.plot_harmonic_diagnostics(
            {'target': scan, 'residual': scan}, {'target': daily, 'residual': daily},
            {'target': {'amplitude': fit_a, 'phase': fit_p},
             'residual': {'amplitude': fit_a, 'phase': fit_p}}, surface)
        self.assertEqual(len(fig.axes), 4)
        plt.close(fig)
```

- [x] **Step 2: Implement**

```python
def plot_harmonic_diagnostics(scans, dailies, fits, surface, title='',
                              save_path=None, filename=None):
    """
    The harmonic diagnostic of spec D6, on one page.

    Four panels: the periods the spectral scan certifies for each series;
    the daily amplitude by day of year with its annual fit; the daily phase
    likewise; and the singular-value shares of the daily-by-annual surface.

    Parameters
    ----------
    scans, dailies : dict of pd.DataFrame
        Keyed by series name; outputs of ``prediction.period_scan`` and
        ``monitoring.daily_harmonic``.
    fits : dict of dict
        ``fits[name]['amplitude']`` and ``fits[name]['phase']`` from
        ``coupling.annual_modulation``.
    surface : dict
        Output of ``coupling.cycle_surface_rank``.
    title : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    from shmlib import coupling as _coupling

    fig, axes = plt.subplots(2, 2, figsize=viz.figsize(viz.FIGURE_WIDTH, 5.2))
    styles = {'target': '-', 'residual': '--'}
    colour = viz.INC_COLOUR
    year = pd.date_range('2001-01-01', periods=365, freq='D')

    ax = axes[0, 0]
    for name, scan in scans.items():
        ax.stem(scan['period_days'], scan['power'], linefmt=colour,
                markerfmt=' ', basefmt=' ', label=name)
    ax.set_xscale('log')
    ax.set_xlabel('Period [days]')
    ax.set_ylabel('Normalised power')
    for period in (0.5, 1.0, 182.6, 365.25):
        ax.axvline(period, color=viz.MARK_COLOUR, linewidth=0.6, alpha=0.6)

    for ax, key, label in ((axes[0, 1], 'amplitude', 'Daily amplitude [mdeg]'),
                           (axes[1, 0], 'phase_h', 'Hour of daily maximum')):
        for name, daily in dailies.items():
            doy = daily.index.dayofyear
            ax.scatter(doy, daily[key], s=3, color=colour, alpha=0.15)
            fit = fits[name]['amplitude' if key == 'amplitude' else 'phase']
            curve = _coupling.evaluate_modulation(fit, year)
            ax.plot(year.dayofyear, curve, color=colour, linewidth=1.6,
                    linestyle=styles.get(name, '-'), label=name)
        ax.set_xlabel('Day of year')
        ax.set_ylabel(label)

    ax = axes[1, 1]
    table = surface['table'].head(6)
    ax.bar(table['component'], table['variance_share'], color=colour, width=0.7)
    ax.set_xlabel('Component')
    ax.set_ylabel('Share of variance')

    for ax in axes.ravel():
        viz.format_spines(ax)
    axes[0, 1].legend(fontsize='small', ncol=2, loc='upper center',
                      bbox_to_anchor=(0.5, -0.30), frameon=False)
    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig
```

- [x] **Step 3: Run `python shmlib/tests/test_shmlib.py`; expected PASS. Commit**

```bash
git add studies/shmlib/figures.py studies/shmlib/tests/test_shmlib.py
git commit -m "feat(shmlib): draw the harmonic diagnostic on one page"
```

### Task 1b.6 (S runs, O interprets and writes): Movement 1b in the notebook, `GM_04`, `GM_F02`, then §5 first half of the report

**Files:**
- Modify: notebook (append Movement 1b cells); parameter cell values `YEARLY_ORDER`, `DAILY_ORDER`, `WEIGHT_CURVE`.
- Modify: report `sec:seasonality` (first half); README status.

- [x] **Step 1: Append Movement 1b cells**

```python
# %% [markdown]
# ## Movement 1b · Harmonic diagnostics before modelling
#
# Two series: the target, and the residual of a plain least-squares
# regression of the target on the on-structure set's drivers. The spectral
# scan fixes the seasonal orders; the daily cycle's amplitude and phase,
# fitted against day of year, give the weight curve; the surface rank says
# how many weighted daily shapes the conditional term needs (D6).

# %%
drivers = sets['str'][['tair', 'rh', 'sr']]
paired = pd.concat([frame['y'], drivers], axis=1).dropna()
design = np.column_stack([np.ones(len(paired)), paired[['tair', 'rh', 'sr']].to_numpy()])
coef, _, _, _ = np.linalg.lstsq(design, paired['y'].to_numpy(), rcond=None)
residual_ols = pd.Series(paired['y'].to_numpy() - design @ coef, index=paired.index)
series_for_scan = {'target': frame['y'], 'residual': residual_ols}
print('OLS gains on the on-structure set:', dict(zip(['tair', 'rh', 'sr'], coef[1:].round(4))))

# %%
scans, dailies, fits = {}, {}, {}
for name, series in series_for_scan.items():
    scans[name] = prediction.period_scan(
        series, min_days=PERIOD_SCAN_MIN_DAYS, max_days=PERIOD_SCAN_MAX_DAYS,
        n_periods=8000, top=8).assign(series=name)
    band = coupling.diurnal_band(series, window=72)
    dailies[name] = monitoring.daily_harmonic(band, min_slots=DAILY_HARMONIC_MIN_SLOTS)
    table_a, fit_a = coupling.annual_modulation(dailies[name]['amplitude'],
                                                harmonics=ANNUAL_MODULATION_HARMONICS)
    table_p, fit_p = coupling.annual_modulation(dailies[name]['phase_h'],
                                                harmonics=ANNUAL_MODULATION_HARMONICS)
    fits[name] = {'amplitude': fit_a, 'phase': fit_p,
                  'amplitude_table': table_a.assign(series=name, statistic='amplitude'),
                  'phase_table': table_p.assign(series=name, statistic='phase')}
surface = coupling.cycle_surface_rank(coupling.diurnal_band(residual_ols, window=72),
                                      doy_bins=SURFACE_DOY_BINS, freq=NATIVE_FREQ)

harmonic = pd.concat(
    [pd.concat(scans.values()).assign(block='period_scan')]
    + [fits[n][k].assign(block='annual_modulation') for n in fits
       for k in ('amplitude_table', 'phase_table')]
    + [surface['table'].assign(block='surface_rank', series='residual')],
    ignore_index=True)
display(harmonic)
harmonic.to_csv(OUTPUT_DIR / 'GM_04_harmonic_diagnostics.csv', index=False)
tables.write_table(
    harmonic[harmonic['block'] == 'period_scan'], str(OUTPUT_DIR / 'GM_04_body.tex'),
    [('series', tables.texttt), ('rank', 'd'), ('period_days', ',.2f'), ('power', '.3f')])
tables.write_table(
    harmonic[harmonic['block'] == 'annual_modulation'],
    str(OUTPUT_DIR / 'GM_04b_body.tex'),
    [('series', tables.texttt), ('statistic', tables.texttt), ('order', 'd'),
     ('holdout_mse', ',.3f'), ('chosen', tables.yes_no)])
tables.write_table(
    harmonic[harmonic['block'] == 'surface_rank'].head(6),
    str(OUTPUT_DIR / 'GM_04c_body.tex'),
    [('component', 'd'), ('variance_share', tables.percent),
     ('cumulative_share', tables.percent)])

figures.plot_harmonic_diagnostics(
    scans, dailies, fits, surface, title='Harmonic diagnostics',
    save_path=str(OUTPUT_DIR), filename='GM_F02_harmonic_diagnostics')

# %%
# What the diagnostic decides, printed so the checkpoint can read it back.
certified = scans['residual']
has_semi_annual = ((certified['period_days'] - 182.6).abs() < certified['resolution_days']).any()
has_twelve_hour = ((certified['period_days'] - 0.5).abs() < 0.05).any()
print('semi-annual peak on the residual:', bool(has_semi_annual))
print('12-hour peak on the residual:', bool(has_twelve_hour))
print('annual modulation order for the daily amplitude:', fits['residual']['amplitude']['order'])
print('first two surface components carry',
      f"{surface['table']['cumulative_share'].iloc[1]:.1%}")
weights_measured = prediction.seasonal_weights(frame.index, modulation=fits['residual']['amplitude'])
print('measured summer weight peaks on day', int(weights_measured['summer_w'].idxmax().dayofyear))
```

- [x] **Step 2: Sync, execute to the scratch path, read back the printed decisions, copy the executed notebook back.**

- [x] **Step 3 (O): Fix the parameter cell from the measurement.** Set `YEARLY_ORDER = 2` if the semi-annual peak is certified else `1`; `DAILY_ORDER = 2` if the 12-hour peak is certified else `1`; `WEIGHT_CURVE = {'order': …, 'coef': […], 'period_days': 365.25}` copied from `fits['residual']['amplitude']` (print `fits['residual']['amplitude']` and paste the numbers), with a guidance bullet naming `GM_04`. If the first two surface components carry under 90 %, record it and keep `CONDITIONAL_DAILY = True` with a note that Phase 2's test decides.

- [x] **Step 4 (O): Write §5 Seasonality, first half** — the diagnostic: what was scanned, which periods were certified on each series (from `GM_04_body.tex`), the annual modulation of amplitude and phase with the held-out order (`GM_04b_body.tex`), the surface rank (`GM_04c_body.tex`), `\includegraphics{GM_F02_harmonic_diagnostics.png}`, and the orders and weight curve that follow. Leave a line `\pending{Phase 2 — the fitted yearly and daily terms}` at the end of the section for the second half.

- [x] **Step 5: Build twice, honesty test, README status "Phase 1b complete", commit**

```bash
git add studies/05_greybox_monitoring
git commit -m "feat(study05): measure the seasonal orders and the daily-cycle weight curve before modelling"
```

> **Checkpoint 1b (O):** `YEARLY_ORDER`, `DAILY_ORDER`, weight curve, surface rank, each traced to `GM_04`; the PDF's §5 first half. Approval opens Phase 2.

---

# Phase 2 · Model A attribution on three sets (spec D5, D7)

### Task 2.1 (S): Additive parameters on `prediction.neuralprophet_backtest`

**Files:**
- Modify: `studies/shmlib/prediction.py:895-1030` (`neuralprophet_backtest`)
- Create: `studies/05_greybox_monitoring/tests/test_model_a.py`

**Interfaces:**
- Produces: `neuralprophet_backtest(train, test, regressors=(), task='forecast', n_lags=24, n_forecasts=24, regressor_lags=12, horizons=None, epochs=30, yearly=False, quantiles=(0.05, 0.95), seed=0, growth='off', changepoints=None, n_changepoints=10, freq=None, decompose=False, trend_reg=0.0, changepoints_range=None, learning_rate=0.01, yearly_order=None, daily_order=None, conditional_seasonality=None, lagged_regressors=(), lagged_n_lags=None, lagged_regularization=None, validation=None)`. New behaviour, each only when the new argument is given: `trend_reg` and `learning_rate` reach the constructor; `changepoints_range` reaches it when not `None`; `yearly_order`/`daily_order` replace the boolean seasonality flags by integer Fourier orders; `conditional_seasonality={'daily_summer': 'summer_w', 'daily_winter': 'winter_w'}` turns `daily_seasonality` off and registers one `add_seasonality(name, period=1, fourier_order=daily_order or 6, condition_name=column)` per entry, passing the condition columns through to the model frames; `lagged_regressors` are registered with `add_lagged_regressor(name, n_lags=lagged_n_lags or regressor_lags, regularization=lagged_regularization)` in **either** task and their columns pass through; `validation` (a frame like `train`) makes the fit collect metrics (`collect_metrics=True`, `validation_df=`, `minimal=False`) and stores the returned metrics frame on the model as `model.fit_metrics_`. Every default reproduces today's behaviour exactly.

- [x] **Step 1: Write the failing tests**

```python
"""
Tests for the Model A additions Study 05 makes to shmlib.prediction.

Run from studies/:  python 05_greybox_monitoring/tests/test_model_a.py
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


def _frame(n=24 * 60, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC')
    hours = np.arange(n)
    tair = 10.0 + 6.0 * np.sin(2 * np.pi * hours / 24.0) + rng.normal(0, 0.2, n)
    y = 50.0 - 2.5 * tair + 0.01 * hours + rng.normal(0, 0.3, n)
    weights = prediction.seasonal_weights(index)
    return pd.DataFrame({'y': y, 'tair': tair, 'summer_w': weights['summer_w'],
                         'winter_w': weights['winter_w']}, index=index)


class TestBacktestAdditions(unittest.TestCase):

    def test_defaults_are_unchanged(self):
        frame = _frame()
        model, out = prediction.neuralprophet_backtest(
            frame.iloc[:1000], frame.iloc[1000:], regressors=('tair',),
            task='nowcast', epochs=3, freq='1h')
        self.assertEqual(model.config_trend.trend_reg, 0)
        self.assertEqual(len(out), len(frame) - 1000)

    def test_orders_regularisation_and_validation_reach_the_model(self):
        frame = _frame()
        model, _ = prediction.neuralprophet_backtest(
            frame.iloc[:1000], frame.iloc[1000:], regressors=('tair',),
            task='nowcast', epochs=3, freq='1h', trend_reg=1.5,
            yearly_order=2, daily_order=3, learning_rate=0.02,
            validation=frame.iloc[1000:])
        self.assertEqual(model.config_trend.trend_reg, 1.5)
        self.assertTrue(hasattr(model, 'fit_metrics_'))
        self.assertIn('MAE_val', model.fit_metrics_.columns)

    def test_conditional_daily_seasonality_is_decomposed(self):
        frame = _frame()
        model, out = prediction.neuralprophet_backtest(
            frame, frame, regressors=('tair',), task='nowcast', epochs=3,
            freq='1h', daily_order=3,
            conditional_seasonality={'daily_summer': 'summer_w',
                                     'daily_winter': 'winter_w'},
            decompose=True)
        components = prediction.decompose_components(model, frame, regressors=('tair',))
        self.assertIn('daily_summer', components.columns)
        self.assertIn('daily_winter', components.columns)

    def test_lagged_regressor_in_a_nowcast_model(self):
        frame = _frame()
        model, _ = prediction.neuralprophet_backtest(
            frame, frame, regressors=(), task='nowcast', epochs=3, freq='1h',
            lagged_regressors=('tair',), lagged_n_lags=6)
        weights = model.model.get_covar_weights()
        self.assertIn('tair', weights)


if __name__ == '__main__':
    unittest.main(verbosity=2)
```

- [x] **Step 2: Run; expected `TypeError: unexpected keyword argument 'trend_reg'` on the second test and failures on the third and fourth.**

- [x] **Step 3: Implement.** In `neuralprophet_backtest`: extend the signature with the nine new keyword arguments after `decompose=False`; extend the docstring's Parameters accordingly. Replace the constructor call and the regressor loop with:

```python
    lagged_regressors = tuple(lagged_regressors or ())
    conditions = dict(conditional_seasonality or {})
    daily_setting = (True if daily_order is None else int(daily_order))
    if conditions:
        daily_setting = False
    constructor = dict(
        growth=growth,
        changepoints=(list(pd.DatetimeIndex(changepoints))
                      if changepoints is not None else None),
        n_changepoints=int(n_changepoints),
        trend_reg=float(trend_reg),
        n_lags=int(n_lags),
        n_forecasts=int(n_forecasts),
        daily_seasonality=daily_setting,
        weekly_seasonality=False,
        yearly_seasonality=(yearly if yearly_order is None else int(yearly_order)),
        normalize='standardize',
        global_normalization=True,
        global_time_normalization=True,
        unknown_data_normalization=True,
        impute_missing=False,
        drop_missing=False,
        loss_func='SmoothL1Loss',
        learning_rate=float(learning_rate),
        epochs=int(epochs),
        quantiles=list(quantiles or ()),
        collect_metrics=validation is not None,
    )
    if changepoints_range is not None:
        constructor['changepoints_range'] = float(changepoints_range)
    model = NeuralProphet(**constructor)
    for name, column in conditions.items():
        model.add_seasonality(name=name, period=1,
                              fourier_order=int(daily_order or 6),
                              condition_name=column)
    for regressor in regressors:
        if task == 'nowcast':
            model.add_future_regressor(regressor)
        else:
            model.add_lagged_regressor(regressor, n_lags=int(regressor_lags))
    for regressor in lagged_regressors:
        model.add_lagged_regressor(
            regressor, n_lags=int(lagged_n_lags or regressor_lags),
            regularization=lagged_regularization)

    passthrough = tuple(regressors) + lagged_regressors + tuple(conditions.values())
    train_df = _model_frame(train, passthrough)
    test_df = _model_frame(test, passthrough)
    fit_freq = freq if freq is not None else _analysis_freq(train.index)
    train_df = _drop_singleton_segments(train_df, 'neuralprophet_backtest',
                                        'fitting')
    if validation is not None:
        validation_df = _drop_singleton_segments(
            _model_frame(validation, passthrough), 'neuralprophet_backtest',
            'validation')
        model.fit_metrics_ = model.fit(train_df, freq=fit_freq,
                                       validation_df=validation_df,
                                       progress='none', minimal=False)
    else:
        model.fit(train_df, freq=fit_freq, progress='none', minimal=True)
```

Keep the rest (`segmented_test`, `predict_df`, `wide`, `_long_predictions`) unchanged, except `max(int(n_lags), int(regressor_lags))` becomes `max(int(n_lags), int(regressor_lags), int(lagged_n_lags or 0))`. Check `_model_frame`'s signature first (grep `def _model_frame` in `prediction.py`); if its second argument is named differently or it filters columns by a `regressors` list only, pass `passthrough` there — the intent is that every named column survives into the `ds`/`y` frame.

- [x] **Step 4: Run the new tests (4 PASS) and every existing test file listed in Global Constraints (all unchanged).**

- [x] **Step 5: Commit**

```bash
git add studies/shmlib/prediction.py studies/05_greybox_monitoring/tests/test_model_a.py
git commit -m "feat(shmlib): seasonal orders, trend regularisation, conditional seasonality, lagged regressors and validation metrics in the NeuralProphet wrapper"
```

### Task 2.2 (S): Parameter extractors — `trend_parameters`, `seasonal_parameters`, `regressor_gains`

**Files:**
- Modify: `studies/shmlib/prediction.py` (append)
- Modify: `studies/05_greybox_monitoring/tests/test_model_a.py`

**Interfaces:**
- `prediction.trend_parameters(model, frame, changepoints, regressors=()) -> (trend, rates)`: `trend` is a `pd.Series` (the model's `predict_trend` on `frame`'s rows, indexed like `frame`); `rates` a DataFrame with `start`, `end`, `rate_mdeg_per_year` for each segment between consecutive entries of `changepoints` (plus the ends of the record), from the trend's value at the segment ends.
- `prediction.seasonal_parameters(model, dates, freq='20min', conditions=None, regressors=()) -> pd.DataFrame` long: `date`, `hour`, `component`, `value`, from `predict_seasonal_components` on one synthetic day per date (with `y=0.0`, condition columns from `seasonal_weights` when `conditions` maps names to columns, and any regressor columns set to 0.0) plus one synthetic year at daily resolution for the `yearly` component (`date` = the year's days, `hour` = 0).
- `prediction.regressor_gains(components, frame, regressors) -> pd.DataFrame` with `regressor`, `gain`, `r2`: slope of `components['future_regressor_<name>']` on `frame[name]`.

- [x] **Step 1: Write the failing tests** (append)

```python
class TestExtractors(unittest.TestCase):

    def test_regressor_gain_is_the_slope_of_the_component(self):
        index = pd.date_range('2024-01-01', periods=100, freq='1h', tz='UTC')
        tair = pd.Series(np.linspace(0, 10, 100), index=index)
        components = pd.DataFrame({'future_regressor_tair': -2.5 * tair + 3.0}, index=index)
        frame = pd.DataFrame({'tair': tair}, index=index)
        gains = prediction.regressor_gains(components, frame, ('tair',))
        self.assertAlmostEqual(gains.loc[0, 'gain'], -2.5, places=6)
        self.assertAlmostEqual(gains.loc[0, 'r2'], 1.0, places=6)

    def test_trend_rates_recover_a_linear_drift(self):
        frame = _frame(n=24 * 120)
        changepoints = prediction.covered_changepoints(frame.index, 2)
        model, _ = prediction.neuralprophet_backtest(
            frame, frame, regressors=('tair',), task='nowcast', epochs=8,
            freq='1h', growth='linear', changepoints=changepoints,
            n_changepoints=2, quantiles=())
        trend, rates = prediction.trend_parameters(model, frame, changepoints,
                                                   regressors=('tair',))
        self.assertEqual(len(trend), len(frame))
        self.assertGreater(len(rates), 0)
        # 0.01 per hour is 87.6 per year; NeuralProphet's fit is stochastic
        self.assertAlmostEqual(rates['rate_mdeg_per_year'].mean(), 87.6, delta=30.0)

    def test_seasonal_parameters_return_one_curve_per_date(self):
        frame = _frame()
        model, _ = prediction.neuralprophet_backtest(
            frame, frame, regressors=('tair',), task='nowcast', epochs=3,
            freq='1h', daily_order=3, quantiles=())
        curves = prediction.seasonal_parameters(
            model, ['2024-06-21', '2024-12-21'], freq='1h', regressors=('tair',))
        daily = curves[curves['component'] == 'daily']
        self.assertEqual(daily['date'].nunique(), 2)
        self.assertEqual(daily.groupby('date').size().iloc[0], 24)
```

- [x] **Step 2: Run; expected AttributeError on the three functions.**

- [x] **Step 3: Implement**

```python
def regressor_gains(components, frame, regressors):
    """
    The gain each future regressor learned, read from its component.

    NeuralProphet's additive future regressor is linear in the regressor, so
    the slope of the component against the regressor value is the learned
    coefficient in the series' own units per unit of the driver.

    Parameters
    ----------
    components : pd.DataFrame
        Output of :func:`decompose_components`.
    frame : pd.DataFrame
        The frame the components were computed on, holding the regressors.
    regressors : sequence of str

    Returns
    -------
    pd.DataFrame
        ``regressor``, ``gain``, ``r2``.
    """
    rows = []
    for name in regressors:
        paired = pd.concat([components[f'future_regressor_{name}'], frame[name]],
                           axis=1).dropna()
        paired.columns = ['contribution', 'driver']
        slope, intercept = np.polyfit(paired['driver'], paired['contribution'], 1)
        fitted = slope * paired['driver'] + intercept
        total = float(((paired['contribution'] - paired['contribution'].mean()) ** 2).sum())
        r2 = 1.0 - float(((paired['contribution'] - fitted) ** 2).sum()) / total if total else np.nan
        rows.append({'regressor': name, 'gain': float(slope), 'r2': r2})
    return pd.DataFrame(rows, columns=['regressor', 'gain', 'r2'])


def trend_parameters(model, frame, changepoints, regressors=()):
    """
    The fitted trend on the record, and its rate on each segment.

    Read through the public ``predict_trend``; the rate of each segment
    between consecutive changepoints is the trend's change across that
    segment per year, which is what a rate-change plot shows without
    touching the model's internal deltas.

    Returns
    -------
    (pd.Series, pd.DataFrame)
        ``trend`` indexed like ``frame``; ``rates`` with ``start``, ``end``,
        ``rate_mdeg_per_year``.
    """
    df = _model_frame(frame, tuple(regressors))
    predicted = model.predict_trend(df)
    trend = pd.Series(predicted['trend'].to_numpy(dtype=float),
                      index=pd.DatetimeIndex(frame.index[:len(predicted)]))
    edges = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(changepoints))
                                    | {trend.index.min(), trend.index.max()}))
    rows = []
    for start, end in zip(edges[:-1], edges[1:]):
        segment = trend.loc[start:end].dropna()
        if len(segment) < 2:
            continue
        days = (segment.index[-1] - segment.index[0]) / pd.Timedelta(days=1)
        rows.append({'start': segment.index[0], 'end': segment.index[-1],
                     'rate_mdeg_per_year': float(
                         (segment.iloc[-1] - segment.iloc[0]) / days * 365.25)})
    return trend, pd.DataFrame(rows, columns=['start', 'end', 'rate_mdeg_per_year'])


def seasonal_parameters(model, dates, freq='20min', conditions=None,
                        regressors=()):
    """
    The fitted seasonal curves, evaluated on synthetic days and one year.

    For each date one day is built on the grid, with condition columns from
    :func:`seasonal_weights` where the model was fitted with them, and every
    regressor set to zero; ``predict_seasonal_components`` then gives each
    seasonal term over that day. The yearly term is evaluated over one
    synthetic year at daily resolution.

    Returns
    -------
    pd.DataFrame
        Long: ``date``, ``hour``, ``component``, ``value``.
    """
    conditions = dict(conditions or {})
    rows = []

    def _evaluate(index, date_label):
        frame = pd.DataFrame({'y': 0.0}, index=index)
        for name in regressors:
            frame[name] = 0.0
        if conditions:
            weights = seasonal_weights(index, modulation=getattr(model, 'weight_curve_', None))
            for column in conditions.values():
                frame[column] = weights[column].to_numpy()
        df = _model_frame(frame, tuple(regressors) + tuple(conditions.values()))
        out = model.predict_seasonal_components(df)
        for component in [c for c in out.columns if c not in ('ds', 'ID')]:
            for stamp, value in zip(index, out[component].to_numpy()):
                rows.append({'date': date_label, 'hour': stamp.hour + stamp.minute / 60.0,
                             'component': component, 'value': float(value)})

    for date in dates:
        day = pd.Timestamp(date, tz='UTC')
        _evaluate(pd.date_range(day, day + pd.Timedelta('1D'), freq=freq,
                                inclusive='left'), pd.Timestamp(date))
    year = pd.date_range('2001-01-01', periods=365, freq='D', tz='UTC')
    frame = pd.DataFrame({'y': 0.0}, index=year)
    for name in regressors:
        frame[name] = 0.0
    if conditions:
        weights = seasonal_weights(year, modulation=getattr(model, 'weight_curve_', None))
        for column in conditions.values():
            frame[column] = weights[column].to_numpy()
    out = model.predict_seasonal_components(
        _model_frame(frame, tuple(regressors) + tuple(conditions.values())))
    if 'yearly' in out.columns:
        for stamp, value in zip(year, out['yearly'].to_numpy()):
            rows.append({'date': stamp, 'hour': 0.0, 'component': 'yearly',
                         'value': float(value)})
    return pd.DataFrame(rows, columns=['date', 'hour', 'component', 'value'])
```

The notebook sets `model.weight_curve_ = WEIGHT_CURVE` after each fit so that `seasonal_parameters` can rebuild the condition columns; document that in the docstring.

- [x] **Step 4: Run; expected PASS (the trend test may need `epochs=8` raised to 15 if the slope is out of tolerance; note the value used). Existing suites unchanged.**

- [x] **Step 5: Commit**

```bash
git add studies/shmlib/prediction.py studies/05_greybox_monitoring/tests/test_model_a.py
git commit -m "feat(shmlib): read the trend rates, seasonal curves and regressor gains from a fitted model"
```

### Task 2.3 (S): Figures — `plot_fit_metrics`, `plot_trend_parameters`, `plot_seasonal_parameters`, `plot_regressor_gains`

**Files:**
- Modify: `studies/shmlib/figures.py` (append four functions)
- Modify: `studies/shmlib/tests/test_shmlib.py` (four axes-count tests)

- [x] **Step 1: Implement the four functions** (each on the template; signatures fixed here)

```python
def plot_fit_metrics(metrics, title='', save_path=None, filename=None):
    """Training and validation MAE by epoch, the tutorial's own curve, redrawn."""
    fig, ax = plt.subplots(figsize=viz.figsize(viz.FIGURE_WIDTH, 2.4))
    ax.plot(metrics.index, metrics['MAE'], color=viz.INC_COLOUR, label='training')
    if 'MAE_val' in metrics.columns:
        ax.plot(metrics.index, metrics['MAE_val'], color=viz.INC_COLOUR,
                linestyle='--', label='validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE [mdeg]')
    viz.format_spines(ax)
    ax.legend(fontsize='small', ncol=2, loc='upper center',
              bbox_to_anchor=(0.5, -0.30), frameon=False)
    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_trend_parameters(trend, rates, changepoints, title='', save_path=None,
                          filename=None):
    """The trend on covered time above, the rate of each segment below; changepoints in the accent colour."""
    fig, axes = plt.subplots(2, 1, sharex=True,
                             figsize=viz.figsize(viz.FIGURE_WIDTH, 3.6))
    axes[0].plot(trend.index, trend, color=viz.INC_COLOUR, linewidth=1.0)
    axes[0].set_ylabel('Trend [mdeg]')
    for stamp in pd.DatetimeIndex(changepoints):
        for ax in axes:
            ax.axvline(stamp, color=viz.MARK_COLOUR, linewidth=0.6)
    for _, row in rates.iterrows():
        axes[1].hlines(row['rate_mdeg_per_year'], row['start'], row['end'],
                       color=viz.INC_COLOUR, linewidth=2.0)
    axes[1].axhline(0.0, color='black', linewidth=0.5)
    axes[1].set_ylabel('Rate [mdeg/yr]')
    for ax in axes:
        viz.format_spines(ax)
    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_seasonal_parameters(curves, title='', save_path=None, filename=None):
    """The yearly curve on the left; the daily curve per evaluated date on the right."""
    fig, axes = plt.subplots(1, 2, figsize=viz.figsize(viz.FIGURE_WIDTH, 2.6))
    yearly = curves[curves['component'] == 'yearly']
    axes[0].plot(pd.DatetimeIndex(yearly['date']).dayofyear, yearly['value'],
                 color=viz.INC_COLOUR)
    axes[0].set_xlabel('Day of year')
    axes[0].set_ylabel('Yearly term [mdeg]')
    daily = curves[curves['component'] != 'yearly']
    styles = ['-', '--', ':', '-.']
    for style, (date, group) in zip(styles * 4, daily.groupby('date')):
        total = group.groupby('hour')['value'].sum()
        axes[1].plot(total.index, total, color=viz.INC_COLOUR, linestyle=style,
                     label=pd.Timestamp(date).strftime('%d %b'))
    axes[1].set_xlabel('Hour of day (UTC)')
    axes[1].set_ylabel('Daily term [mdeg]')
    for ax in axes:
        viz.format_spines(ax)
    axes[1].legend(fontsize='small', ncol=4, loc='upper center',
                   bbox_to_anchor=(0.5, -0.30), frameon=False)
    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_regressor_gains(gains, title='', save_path=None, filename=None):
    """Learned gain per driver and set, with Study 03's measurement beside each."""
    fig, ax = plt.subplots(figsize=viz.figsize(viz.FIGURE_WIDTH, 2.6))
    labels = [f"{r['set']} · {r['regressor']}" for _, r in gains.iterrows()]
    positions = np.arange(len(gains))
    ax.bar(positions - 0.2, gains['gain'], width=0.4, color=viz.INC_COLOUR,
           label='learned')
    ax.bar(positions + 0.2, gains['study03_gain'], width=0.4,
           color=viz.INC_COLOUR, alpha=0.4, label='Study 03')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize='small')
    ax.axhline(0.0, color='black', linewidth=0.5)
    ax.set_ylabel('Gain [mdeg per unit]')
    viz.format_spines(ax)
    ax.legend(fontsize='small', ncol=2, loc='upper center',
              bbox_to_anchor=(0.5, -0.45), frameon=False)
    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig
```

- [x] **Step 2: Tests** — one per function asserting `len(fig.axes)` (1, 2, 2, 1) on small synthetic frames built inline as in the earlier figure tests; run `python shmlib/tests/test_shmlib.py`; PASS.

- [x] **Step 3: Commit**

```bash
git add studies/shmlib/figures.py studies/shmlib/tests/test_shmlib.py
git commit -m "feat(shmlib): redraw the NeuralProphet fit, trend, seasonality and regressor plots in the project's language"
```

### Task 2.4 (S runs, O interprets): Movement 2 in the notebook — attribution fits on three sets, `GM_05`–`GM_08`, `GM_F03`–`GM_F06`

**Files:**
- Modify: notebook (append Movement 2); parameter cell value `TREND_REG` after the sweep.

- [x] **Step 1: Append the cells**

```python
# %% [markdown]
# ## Movement 2 · What the record is made of
#
# One specification (D7), fitted once per regressor set on the full training
# window for attribution. The trend regularisation is swept on a held-out
# tail; the conditional daily term is tested against the plain one and kept
# only if it earns its place. Every native NeuralProphet plot runs here as a
# diagnostic; the report's figures are the same content redrawn (D14).

# %%
STUDY03_GAINS = {('str', 'tair'): -2.79, ('gs', 'tair'): -2.23, ('era5', 'tair'): -2.04,
                 ('str', 'sr'): -0.035, ('gs', 'sr'): -0.035, ('era5', 'sr'): -0.026,
                 ('str', 'rh'): np.nan, ('gs', 'rh'): np.nan, ('era5', 'rh'): np.nan}

def_frames = {}
for name in sets:
    columns = {f'{r}_{name}': r for r in ['tair', 'rh', 'sr']}
    block = frame[['y'] + list(columns)].rename(columns=columns)
    weights = prediction.seasonal_weights(block.index, modulation=WEIGHT_CURVE)
    block = pd.concat([block, weights], axis=1).dropna(subset=['y', 'tair', 'rh', 'sr'])
    def_frames[name] = block

split_at = int(len(def_frames['str']) * (1 - VALID_P))
train_str = def_frames['str'].iloc[:split_at]
valid_str = def_frames['str'].iloc[split_at:]
changepoints_str = prediction.covered_changepoints(train_str.index, N_CHANGEPOINTS)

# %%
# Sweep the trend regularisation on the held-out tail of the on-structure set.
sweep = []
for candidate in (0.0, 0.5, 1.0, 2.0, 5.0):
    model, out = prediction.neuralprophet_backtest(
        train_str, valid_str, regressors=('tair', 'rh', 'sr'), task='nowcast',
        epochs=EPOCHS, freq=NATIVE_FREQ, growth='linear',
        changepoints=changepoints_str, n_changepoints=N_CHANGEPOINTS,
        changepoints_range=CHANGEPOINTS_RANGE, trend_reg=candidate,
        yearly_order=YEARLY_ORDER, daily_order=DAILY_ORDER,
        quantiles=QUANTILES, seed=SEED, learning_rate=LEARNING_RATE)
    sweep.append({'trend_reg': candidate,
                  'mae_val': float((out['y'] - out['yhat']).abs().mean())})
sweep = pd.DataFrame(sweep)
display(sweep)
TREND_REG = float(sweep.loc[sweep['mae_val'].idxmin(), 'trend_reg'])
print('TREND_REG chosen:', TREND_REG, '(copy into the parameter cell)')

# %%
# The conditional daily term against the plain one, same held-out tail.
conditional_test = []
for label, conditions in (('plain', None),
                          ('conditional', {'daily_summer': 'summer_w',
                                           'daily_winter': 'winter_w'})):
    model, out = prediction.neuralprophet_backtest(
        train_str, valid_str, regressors=('tair', 'rh', 'sr'), task='nowcast',
        epochs=EPOCHS, freq=NATIVE_FREQ, growth='linear',
        changepoints=changepoints_str, n_changepoints=N_CHANGEPOINTS,
        changepoints_range=CHANGEPOINTS_RANGE, trend_reg=TREND_REG,
        yearly_order=YEARLY_ORDER, daily_order=DAILY_ORDER,
        conditional_seasonality=conditions, quantiles=QUANTILES, seed=SEED,
        learning_rate=LEARNING_RATE)
    components = prediction.decompose_components(model, train_str,
                                                 regressors=('tair', 'rh', 'sr'))
    shares = prediction.component_variance_shares(components)
    daily_share = float(shares.loc[shares['component'].str.startswith('daily')
                                   | shares['component'].eq('season_daily'), 'share'].sum())
    conditional_test.append({'daily_term': label,
                             'mae_val': float((out['y'] - out['yhat']).abs().mean()),
                             'daily_share': daily_share})
conditional_test = pd.DataFrame(conditional_test)
display(conditional_test)
keep_conditional = CONDITIONAL_DAILY and (
    conditional_test.loc[1, 'mae_val'] < conditional_test.loc[0, 'mae_val']
    or conditional_test.loc[1, 'daily_share'] > CONDITIONAL_KEEP_MIN_SHARE)
CONDITIONS = ({'daily_summer': 'summer_w', 'daily_winter': 'winter_w'}
              if keep_conditional else None)
print('conditional daily term kept:', keep_conditional)

# %%
# Attribution fit per set, on the full training window; native plots as diagnostics.
models_a, components_a, shares_rows, gain_rows, diag_rows = {}, {}, [], [], []
for name, block in def_frames.items():
    train = block.iloc[:int(len(block) * (1 - VALID_P))]
    valid = block.iloc[int(len(block) * (1 - VALID_P)):]
    changepoints = prediction.covered_changepoints(train.index, N_CHANGEPOINTS)
    model, _ = prediction.neuralprophet_backtest(
        train, valid, regressors=('tair', 'rh', 'sr'), task='nowcast',
        epochs=EPOCHS, freq=NATIVE_FREQ, growth='linear',
        changepoints=changepoints, n_changepoints=N_CHANGEPOINTS,
        changepoints_range=CHANGEPOINTS_RANGE, trend_reg=TREND_REG,
        yearly_order=YEARLY_ORDER, daily_order=DAILY_ORDER,
        conditional_seasonality=CONDITIONS, quantiles=QUANTILES, seed=SEED,
        learning_rate=LEARNING_RATE, validation=valid)
    model.weight_curve_ = WEIGHT_CURVE
    models_a[name] = (model, train, changepoints)
    components = prediction.decompose_components(model, train, regressors=('tair', 'rh', 'sr'))
    components_a[name] = components
    shares_rows.append(prediction.component_variance_shares(components).assign(set=name))
    gains = prediction.regressor_gains(components, train, ('tair', 'rh', 'sr')).assign(set=name)
    gains['study03_gain'] = [STUDY03_GAINS[(name, r)] for r in gains['regressor']]
    gain_rows.append(gains)
    diag_rows.append(prediction.residual_diagnostics(
        components['residual'], lags=(1, 72, 216)).assign(set=name))
    if name == 'str':
        model.set_plotting_backend('plotly-static')
        forecast_native = model.predict(prediction._model_frame(train, ('tair', 'rh', 'sr') + (tuple(CONDITIONS.values()) if CONDITIONS else ())), decompose=True)
        model.plot(forecast_native)
        model.plot_components(forecast_native)
        model.plot_parameters()

shares = pd.concat(shares_rows, ignore_index=True)
gains = pd.concat(gain_rows, ignore_index=True)
diagnostics = pd.concat(diag_rows, ignore_index=True)
for table, number, name in ((shares, '05', 'component_shares'),
                            (gains, '06', 'learned_gains'),
                            (diagnostics, '08', 'residual_diagnostics')):
    display(table)
    table.to_csv(OUTPUT_DIR / f'GM_{number}_{name}.csv', index=False)
tables.write_table(shares, str(OUTPUT_DIR / 'GM_05_body.tex'),
                   [('set', tables.texttt), ('component', tables.texttt),
                    ('share', tables.percent), ('peak_to_peak', ',.1f')])
tables.write_table(gains, str(OUTPUT_DIR / 'GM_06_body.tex'),
                   [('set', tables.texttt), ('regressor', tables.texttt),
                    ('gain', '.3f'), ('study03_gain', '.3f'), ('r2', '.3f')])
tables.write_table(diagnostics, str(OUTPUT_DIR / 'GM_08_body.tex'),
                   [('set', tables.texttt), ('lag', 'd'), ('lb_pvalue', '.3g'),
                    ('std', '.2f'), ('mad', '.2f')])

# %%
# Fold stability the NeuralProphet way: chronological folds, one refit each.
stability_rows = []
for name, block in def_frames.items():
    model0, _, _ = models_a[name]
    df = prediction._model_frame(block, ('tair', 'rh', 'sr') + (tuple(CONDITIONS.values()) if CONDITIONS else ()))
    folds = model0.crossvalidation_split_df(df, freq=NATIVE_FREQ, k=CV_FOLDS,
                                            fold_pct=CV_FOLD_PCT,
                                            fold_overlap_pct=CV_FOLD_OVERLAP_PCT)
    for k, (fold_train_df, fold_test_df) in enumerate(folds, 1):
        fold_train = block.loc[block.index.isin(pd.DatetimeIndex(fold_train_df['ds']).tz_localize('UTC') if fold_train_df['ds'].dt.tz is None else fold_train_df['ds'])]
        fold_test = block.loc[block.index.isin(pd.DatetimeIndex(fold_test_df['ds']).tz_localize('UTC') if fold_test_df['ds'].dt.tz is None else fold_test_df['ds'])]
        changepoints = prediction.covered_changepoints(fold_train.index, N_CHANGEPOINTS)
        model, out = prediction.neuralprophet_backtest(
            fold_train, fold_test, regressors=('tair', 'rh', 'sr'), task='nowcast',
            epochs=EPOCHS, freq=NATIVE_FREQ, growth='linear', changepoints=changepoints,
            n_changepoints=N_CHANGEPOINTS, changepoints_range=CHANGEPOINTS_RANGE,
            trend_reg=TREND_REG, yearly_order=YEARLY_ORDER, daily_order=DAILY_ORDER,
            conditional_seasonality=CONDITIONS, quantiles=(), seed=SEED,
            learning_rate=LEARNING_RATE)
        components = prediction.decompose_components(model, fold_train, regressors=('tair', 'rh', 'sr'))
        gain = prediction.regressor_gains(components, fold_train, ('tair',)).loc[0, 'gain']
        trend, rates = prediction.trend_parameters(model, fold_train, changepoints, regressors=('tair', 'rh', 'sr'))
        yearly = components['season_yearly'] if 'season_yearly' in components else components.get('yearly')
        stability_rows.append({'set': name, 'fold': k,
                               'tair_gain': float(gain),
                               'yearly_peak_to_peak': float(yearly.max() - yearly.min()) if yearly is not None else np.nan,
                               'trend_rate': float(rates['rate_mdeg_per_year'].mean()),
                               'mae_val': float((out['y'] - out['yhat']).abs().mean())})
stability = pd.DataFrame(stability_rows)
display(stability)
stability.to_csv(OUTPUT_DIR / 'GM_07_component_stability.csv', index=False)
tables.write_table(stability, str(OUTPUT_DIR / 'GM_07_body.tex'),
                   [('set', tables.texttt), ('fold', 'd'), ('tair_gain', '.3f'),
                    ('yearly_peak_to_peak', '.1f'), ('trend_rate', '.1f'), ('mae_val', '.2f')])

# %%
model_str, train_str_fit, changepoints_str = models_a['str']
figures.plot_fit_metrics(model_str.fit_metrics_, title='Model A · on-structure set',
                         save_path=str(OUTPUT_DIR), filename='GM_F03_fit_metrics')
trend, rates = prediction.trend_parameters(model_str, train_str_fit, changepoints_str,
                                           regressors=('tair', 'rh', 'sr'))
figures.plot_trend_parameters(trend, rates, changepoints_str, title='Trend on covered time',
                              save_path=str(OUTPUT_DIR), filename='GM_F04_trend')
curves = prediction.seasonal_parameters(
    model_str, ['2024-03-20', '2024-06-21', '2024-09-22', '2024-12-21'],
    freq=NATIVE_FREQ, conditions=CONDITIONS, regressors=('tair', 'rh', 'sr'))
figures.plot_seasonal_parameters(curves, title='Yearly and daily terms',
                                 save_path=str(OUTPUT_DIR), filename='GM_F05_seasonality')
figures.plot_decomposition_stack(components_a['str'], freq=NATIVE_FREQ,
                                 title='Decomposition, on-structure set',
                                 save_path=str(OUTPUT_DIR), filename='GM_F06_decomposition')
figures.plot_regressor_gains(gains, title='Learned gains against Study 03',
                             save_path=str(OUTPUT_DIR), filename='GM_F06b_gains')
```

The fold-index reconstruction line is deliberately defensive about time zones: `crossvalidation_split_df` returns naive `ds`; if the check shows the model frame keeps UTC-aware stamps, simplify to `block.loc[fold_train_df['ds']]`.

- [x] **Step 2: Sync, execute (timeout 14400 s), read back, copy back. Fix `TREND_REG` in the parameter cell to the chosen value with a guidance bullet.**

- [x] **Step 3: Commit**

```bash
git add studies/05_greybox_monitoring
git commit -m "feat(study05): attribution fits on the three regressor sets, shares, gains, fold stability and residual diagnostics"
```

### Task 2.4b (S): Provenance guard — every report graphic and table body is written by the notebook

**Files:**
- Modify: `studies/05_greybox_monitoring/tests/test_folder_honesty.py`

Implements the Global Constraint "Every image in the report comes from the paired notebook" (user rule, 2026-09-06).

- [x] **Step 1: Add a class `TestEveryGraphicComesFromTheNotebook`.** It reads `greybox_monitoring_study.py` once and collects two sets with regular expressions: every `filename='…'` literal (the stems the notebook's `figures.*` calls write) and every `'GM_\d\d[a-z]?_body.tex'` literal (the bodies its `tables.write_table` calls write). Three tests: (a) every `\includegraphics` stem in the report is in the notebook's filename set; (b) every `\input{../outputs/…}` body in the report is in the notebook's body set; (c) every `GM_F*.png` in `outputs/` is named by the notebook (no orphan image can be picked up by the report). Each assertion message names the offending stem and says that a figure is produced by the notebook or not at all.
- [x] **Step 2: Run the file; expected: the three new tests pass on the current tree (every included graphic so far is a notebook `filename` literal).** If one fails, the fix is in the notebook or the report, never in the test.
- [x] **Step 3: Commit** `test(study05): every figure and table body in the report must be written by the notebook`.

### Task 2.4c (S): The residual period scan, and three corrections to Movement 2's figures

**Files:**
- Modify: `studies/shmlib/prediction.py` (`seasonal_parameters`), `studies/shmlib/figures.py` (`plot_seasonal_parameters`, `plot_trend_parameters`), their tests, the notebook's Movement 2.

Rides on one notebook run. (The coverage chart that an earlier version of this task carried is deferred to Task 7.3, with every other item of `report/report05_check.md`, by the user's instruction of 2026-09-06: the check file is not acted on until the report's first complete implementation exists.)

- [x] **Step 1: `GM_08b`, the residual period scan the spec's §4.1 asks for.** In Movement 2, after the fold-stability step, one `prediction.period_scan` per set on `components_a[name]['residual']` over the two bands of `PERIOD_SCAN_BANDS` with `spacing='log'`, `n_periods=PERIOD_SCAN_N`, `top=5`, concatenated with `set` and `band` columns into `GM_08b_residual_periods.csv` and `GM_08b_body.tex` (`set`, `band`, `rank`, `period_days` at four decimals, `power` at four decimals), under its own `###` Markdown cell.
- [x] **Step 2: Three figure corrections** (controller findings on the executed Movement 2): `seasonal_parameters`' per-day evaluations keep only the non-yearly components and `plot_seasonal_parameters` sorts the yearly block by date; `SEASONAL_CURVE_DATES` joins the Model A parameter cell and the figure cell evaluates all four dates only when `CONDITIONS` is not `None`; `plot_trend_parameters` gains `freq=None` and breaks the trend line across gaps through `figures._reindex_regular` when given. Tests for each.
- [x] **Step 3: Table refinements.** `GM_05b`'s share at two decimals; `GM_06`'s body without the `r2` column (kept in the CSV; one by construction); `curves` saved as `GM_05d_seasonal_curves.csv`.
- [x] **Step 4: Sync, execute to the scratch directory, verify, copy back; commit** `feat(study05): the residual period scan, and the seasonal and trend figures corrected` (library changes committed first as `feat(shmlib): corrections to the seasonal and trend parameter figures`).

### Task 2.5 (O): Checkpoint 2 gate

- [x] Read `GM_06`: the on-structure `tair` gain against −2.79 with Study 03's month-to-month range (−3.63 to −1.88) as the interval. Inside: proceed. Outside: **stop the study here**; the disagreement is the finding and becomes the report's §6 and Verdict (spec Phase 2 kill criterion).
- [x] Read `GM_08`: Ljung–Box at 1, 72, 216; name the missing component if structure remains.
- [x] Show the user `GM_05`–`GM_08` and the four figures.

### Task 2.6 (O): Report §4 Trend, §5 second half, §6 Regressors and gains

- [x] §4: the trend on covered time, `\includegraphics{GM_F04_trend.png}`, the rates per segment (from `GM_07`'s `trend_rate` spread and the rates table printed in the notebook; if a rates CSV is wanted, add `rates.to_csv(OUTPUT_DIR / 'GM_04d_trend_rates.csv')` to Movement 2 and cite it).
- [x] §5 second half: replace the remaining `\pending{}` with the fitted yearly and daily terms, `\includegraphics{GM_F05_seasonality.png}`, and the conditional-versus-plain comparison (write `conditional_test.to_csv(OUTPUT_DIR / 'GM_05b_conditional_test.csv')` and a body in Movement 2 first, so the number has an artefact).
- [x] §6: shares (`GM_05_body.tex`), gains confronted with Study 03 (`GM_06_body.tex`, `GM_F06b_gains.png`), the decomposition stack (`GM_F06_decomposition.png`), fold stability (`GM_07_body.tex`), residual diagnostics (`GM_08_body.tex`).
- [x] Build twice, honesty test, README status "Phase 2 complete", commit `docs(study05): write the trend, seasonality and regressor sections`.

> **Checkpoint 2 (O):** as Task 2.5, plus the PDF. Approval opens Phase 2b.

---

# Phase 2b · What the wall temperature and the pyranometer buy (spec D4)

### Task 2b.1 (S runs, O interprets): Movement 2b — the current-era ladder, `GM_16`, `GM_F14`

**Files:**
- Modify: notebook (append Movement 2b); `studies/shmlib/figures.py` (+ `plot_ladder`); `studies/shmlib/tests/test_shmlib.py` (one test).

- [x] **Step 1: Add `figures.plot_ladder`**

```python
def plot_ladder(ladder, title='', save_path=None, filename=None):
    """Held-out MAE per rung with its bootstrap increment interval below, one bar per rung."""
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=viz.figsize(viz.FIGURE_WIDTH, 3.4))
    positions = np.arange(len(ladder))
    axes[0].bar(positions, ladder['mae_val'], color=viz.INC_COLOUR, width=0.6)
    axes[0].set_ylabel('Held-out MAE [mdeg]')
    axes[1].errorbar(positions, ladder['skill'],
                     yerr=[ladder['skill'] - ladder['q05'], ladder['q95'] - ladder['skill']],
                     fmt='o', color=viz.INC_COLOUR, capsize=3)
    axes[1].axhline(0.0, color='black', linewidth=0.5)
    axes[1].set_ylabel('Skill over rung below')
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(ladder['rung'], rotation=20, ha='right', fontsize='small')
    for ax in axes:
        viz.format_spines(ax)
    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig
```

Test: a four-row frame with columns `rung, mae_val, skill, q05, q95`; assert two axes.

- [x] **Step 2: Append Movement 2b cells**

```python
# %% [markdown]
# ## Movement 2b · What the wall temperature and the pyranometer buy
#
# The current era only (D4). The same specification, refitted rung by rung
# on a matched window; each rung reports its held-out error and the paired
# block-bootstrap increment over the rung below.

# %%
current = frame.loc[CURRENT_ERA_START:].copy()
current['twall'] = sensor['twall_str'].reindex(current.index)
current['sr_str'] = sensor['sr_str'].reindex(current.index)
slots_per_hour = int(pd.Timedelta(hours=1) / pd.Timedelta(NATIVE_FREQ))
current['twall_tau'] = coupling.thermal_operator(
    current['twall'], delay=0, tau=TWALL_TAU_H, dt_hours=1.0 / slots_per_hour)
current['twall_lead'] = current['twall'].shift(TWALL_LEAD_H * slots_per_hour)  # negative lead: future values
weights = prediction.seasonal_weights(current.index, modulation=WEIGHT_CURVE)
current = pd.concat([current, weights], axis=1)

rungs = [
    ('1 on-structure set', ['tair_str', 'rh_str', 'sr_gs'], None),
    ('2 on-structure radiation', ['tair_str', 'rh_str', 'sr_str'], 'sr_str'),
    ('3 + twall, tau 4 h', ['tair_str', 'rh_str', 'sr_str', 'twall_tau'], 'twall_tau'),
    ('4 twall at its lead (diagnostic)', ['tair_str', 'rh_str', 'sr_str', 'twall_lead'], 'twall_lead'),
]
ladder_rows, errors = [], {}
for label, columns, gate in rungs:
    block = current.dropna(subset=['y'] + columns)
    if gate is not None:
        block = block.loc[block[gate].notna()]
    split = int(len(block) * (1 - VALID_P))
    train, valid = block.iloc[:split], block.iloc[split:]
    changepoints = prediction.covered_changepoints(train.index, max(2, N_CHANGEPOINTS // 3))
    model, out = prediction.neuralprophet_backtest(
        train, valid, regressors=tuple(columns), task='nowcast', epochs=EPOCHS,
        freq=NATIVE_FREQ, growth='linear', changepoints=changepoints,
        n_changepoints=max(2, N_CHANGEPOINTS // 3), changepoints_range=CHANGEPOINTS_RANGE,
        trend_reg=TREND_REG, yearly_order=1, daily_order=DAILY_ORDER,
        conditional_seasonality=CONDITIONS, quantiles=QUANTILES, seed=SEED,
        learning_rate=LEARNING_RATE)
    components = prediction.decompose_components(model, train, regressors=tuple(columns))
    gains = prediction.regressor_gains(components, train, tuple(columns))
    abs_error = (out['y'] - out['yhat']).abs()
    abs_error.index = pd.DatetimeIndex(out['ds'])
    errors[label] = abs_error
    coverage = float(((out['y'] >= out['q05']) & (out['y'] <= out['q95'])).mean()) if 'q05' in out else np.nan
    ladder_rows.append({'rung': label, 'rows': len(block), 'mae_val': float(abs_error.mean()),
                        'coverage': coverage,
                        'gain_last': float(gains['gain'].iloc[-1]),
                        'residual_r1': float(components['residual'].autocorr(1))})
ladder = pd.DataFrame(ladder_rows)
skills = [{'skill': np.nan, 'q05': np.nan, 'q95': np.nan}]
for previous, current_label in zip([r[0] for r in rungs[:-1]], [r[0] for r in rungs[1:]]):
    shared = errors[previous].index.intersection(errors[current_label].index)
    skills.append(prediction.paired_mae_skill(
        errors[previous].loc[shared], errors[current_label].loc[shared],
        block_hours=LADDER_BOOTSTRAP_BLOCK_HOURS,
        repetitions=LADDER_BOOTSTRAP_REPETITIONS, seed=SEED))
ladder = pd.concat([ladder, pd.DataFrame(skills)], axis=1)
display(ladder)
ladder.to_csv(OUTPUT_DIR / 'GM_16_current_era_ladder.csv', index=False)
tables.write_table(ladder, str(OUTPUT_DIR / 'GM_16_body.tex'),
                   [('rung', tables.texttt), ('rows', ',d'), ('mae_val', '.2f'),
                    ('skill', '.3f'), ('q05', '.3f'), ('q95', '.3f'),
                    ('coverage', tables.percent), ('gain_last', '.3f'), ('residual_r1', '.3f')])
figures.plot_ladder(ladder, title='What each on-structure channel buys',
                    save_path=str(OUTPUT_DIR), filename='GM_F14_ladder')
```

Before running, check `prediction.paired_mae_skill`'s return type (grep its docstring, `prediction.py:649`): if it returns a dict with keys `skill`, `q05`, `q95`, the code above stands; if it returns a one-row DataFrame, take `.iloc[0].to_dict()`.

- [x] **Step 3: Sync, execute, copy back, commit `feat(study05): the current-era ladder of on-structure channels`.**

### Task 2b.2 (O): Report §9

- [x] Write §9 from `GM_16_body.tex` and `GM_F14_ladder.png`: one paragraph per rung, the verdict sentence per channel (probe and pyranometer), and the caveat that rung 4 uses the probe's future values and is a diagnostic only. Build, honesty test, README status, commit `docs(study05): write what the wall temperature and the pyranometer buy`.

> **Checkpoint 2b (O):** a verdict per channel. Approval opens Phase 3.

---

# Phase 3 · The rolling expectation and its interval (spec D5, D8)

### Task 3.1 (S): `rolling_nowcast` additions and `prediction.rolling_conformal`

**Files:**
- Modify: `studies/shmlib/prediction.py:1033-1152` (`rolling_nowcast`) and append `rolling_conformal`
- Modify: `studies/05_greybox_monitoring/tests/test_model_a.py`

**Interfaces:**
- `rolling_nowcast(frame, regressors=(), refit_every='30d', min_train='180d', freq=None, train_window=None, changepoints_per_window=False, **model_kwargs)`: with `train_window` (offset alias) each window trains on `[origin − train_window, origin)` instead of all history; with `changepoints_per_window=True` each window computes `covered_changepoints(train.index, model_kwargs['n_changepoints'])` and passes them as `changepoints`. Defaults reproduce today's behaviour. Output gains a column `staleness_d = (ds − origin) / 1 day`.
- `prediction.rolling_conformal(predictions, alpha=0.10, window='180d', method='cqr') -> pd.DataFrame`: copy of the rolling output in which, for every origin, the calibration set is the rolling output's own rows with `ds` in `[origin − window, origin)` (all out of sample by construction); the nonconformity score is `max(q05 − y, y − q95)` for `'cqr'` and `|y − yhat|` for `'naive'`; `qhat` is the `⌈(n+1)(1−α)⌉/n` empirical quantile; the model's `q05`/`q95` are kept as `q05_model`/`q95_model` and replaced by the conformal `q05 = q05_model − qhat`, `q95 = q95_model + qhat` (`'naive'`: `yhat ∓ qhat`). Rows whose origin has no calibration rows get `NaN` bounds. Columns added: `qhat`, `n_cal`.

- [ ] **Step 1: Write the failing tests** (append)

```python
class TestRollingAdditions(unittest.TestCase):

    def test_rolling_conformal_covers_at_the_nominal_rate(self):
        rng = np.random.default_rng(3)
        ds = pd.date_range('2024-01-01', periods=24 * 400, freq='1h', tz='UTC')
        y = rng.normal(0, 2.0, len(ds))
        origins = ds.floor('30D')
        predictions = pd.DataFrame({'ds': ds, 'horizon_h': 0.0, 'y': y, 'yhat': 0.0,
                                    'q05': -1.0, 'q95': 1.0, 'origin': origins})
        out = prediction.rolling_conformal(predictions, alpha=0.10, window='120d')
        scored = out.dropna(subset=['q05', 'q95'])
        covered = ((scored['y'] >= scored['q05']) & (scored['y'] <= scored['q95'])).mean()
        self.assertAlmostEqual(covered, 0.90, delta=0.03)
        self.assertIn('q05_model', out.columns)
        self.assertTrue(out['qhat'].dropna().gt(0).all())

    def test_train_window_and_per_window_changepoints_are_accepted(self):
        frame = _frame(n=24 * 120)
        out = prediction.rolling_nowcast(
            frame, regressors=('tair',), refit_every='20d', min_train='40d',
            train_window='60d', changepoints_per_window=True, freq='1h',
            epochs=2, growth='linear', n_changepoints=2, quantiles=(0.05, 0.95))
        self.assertIn('staleness_d', out.columns)
        self.assertGreater(out['origin'].nunique(), 1)
```

- [ ] **Step 2: Run; expected TypeError on `train_window`, AttributeError on `rolling_conformal`.**

- [ ] **Step 3: Implement.** In `rolling_nowcast`: add the two keyword arguments before `**model_kwargs`; inside the loop replace `train = ordered.loc[index < origin]` by

```python
        lower = (origin - pd.Timedelta(train_window)) if train_window else index.min()
        train = ordered.loc[(index >= lower) & (index < origin)]
```

and, just before the `neuralprophet_backtest` call,

```python
            window_kwargs = dict(model_kwargs)
            if changepoints_per_window:
                window_kwargs['changepoints'] = covered_changepoints(
                    train.index, int(window_kwargs.get('n_changepoints', 10)))
```

passing `**window_kwargs`. After `predictions['origin'] = origin` add `predictions['staleness_d'] = (pd.DatetimeIndex(predictions['ds']) - origin) / pd.Timedelta(days=1)`. Document both in the docstring. Then append:

```python
def rolling_conformal(predictions, alpha=0.10, window='180d', method='cqr'):
    """
    Conformal bounds calibrated on the walk-forward output's own recent past.

    For each refit origin the calibration set is every row of ``predictions``
    with ``ds`` in ``[origin - window, origin)``. Those rows were predicted by
    earlier fits and are out of sample by construction, so the split
    conformal guarantee holds while every fit stays as fresh as the refit
    schedule allows (spec D8). The ``'cqr'`` method is conformalised quantile
    regression on the model's own quantile columns; ``'naive'`` is the
    symmetric absolute-residual interval about ``yhat``.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of :func:`rolling_nowcast` with ``ds``, ``y``, ``yhat``,
        ``origin`` and, for ``'cqr'``, ``q05`` and ``q95``.
    alpha : float, optional
        Miscoverage. Default ``0.10``.
    window : str, optional
        Length of the calibration set behind each origin. Default ``'180d'``.
    method : {'cqr', 'naive'}, optional

    Returns
    -------
    pd.DataFrame
        Copy with ``q05_model``/``q95_model`` (when present), conformal
        ``q05``/``q95``, ``qhat`` and ``n_cal``.
    """
    out = predictions.copy()
    ds = pd.DatetimeIndex(out['ds'])
    if method == 'cqr':
        for column in ('q05', 'q95'):
            out[f'{column}_model'] = out[column]
        scores = np.maximum(out['q05_model'] - out['y'], out['y'] - out['q95_model'])
    elif method == 'naive':
        scores = (out['y'] - out['yhat']).abs()
    else:
        raise ValueError("method must be 'cqr' or 'naive'")
    scores = pd.Series(scores.to_numpy(dtype=float), index=out.index)
    span = pd.Timedelta(window)
    qhat = pd.Series(np.nan, index=out.index)
    n_cal = pd.Series(0, index=out.index, dtype=int)
    for origin, rows in out.groupby('origin').groups.items():
        origin = pd.Timestamp(origin)
        calibration = scores[(ds >= origin - span) & (ds < origin)].dropna()
        n = int(calibration.size)
        n_cal.loc[rows] = n
        if n == 0:
            continue
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        qhat.loc[rows] = float(np.quantile(calibration, level))
    out['qhat'] = qhat
    out['n_cal'] = n_cal
    if method == 'cqr':
        out['q05'] = out['q05_model'] - out['qhat']
        out['q95'] = out['q95_model'] + out['qhat']
    else:
        out['q05'] = out['yhat'] - out['qhat']
        out['q95'] = out['yhat'] + out['qhat']
    return out
```

- [ ] **Step 4: Run new tests (PASS) and Study 04's `test_prediction.py` (unchanged).**

- [ ] **Step 5: Commit** `feat(shmlib): bounded training windows, per-window changepoints and rolling conformal bounds for the walk-forward expectation`.

### Task 3.2 (S runs, O reviews): Movement 3 — the rolling expectation per set, `GM_09`, `GM_F03` already done, `GM_F08`

**Files:**
- Modify: notebook (parameter cell: add `TRAIN_WINDOW = '1095d'` with guidance in the Model A group; append Movement 3).

- [ ] **Step 1: Append the cells**

```python
# %% [markdown]
# ## Movement 3 · Is this reading the expected one?
#
# A walk-forward expectation, refitted every `REFIT_EVERY` on the trailing
# `TRAIN_WINDOW` with the trend on (D5), and a conformal interval calibrated
# on the previous `CONFORMAL_CALIBRATION_WINDOW` of out-of-sample residuals
# (D8). Scored per set and per days since refit.

# %%
rolling_sets, metric_rows = {}, []
staleness_edges = [-0.01, 7, 14, 21, 31]
for name, block in def_frames.items():
    rolling = prediction.rolling_nowcast(
        block, regressors=('tair', 'rh', 'sr'), refit_every=REFIT_EVERY,
        min_train=MIN_TRAIN, train_window=TRAIN_WINDOW,
        changepoints_per_window=True, freq=NATIVE_FREQ, epochs=EPOCHS,
        growth='linear', n_changepoints=N_CHANGEPOINTS,
        changepoints_range=CHANGEPOINTS_RANGE, trend_reg=TREND_REG,
        yearly_order=YEARLY_ORDER, daily_order=DAILY_ORDER,
        conditional_seasonality=CONDITIONS, quantiles=QUANTILES, seed=SEED,
        learning_rate=LEARNING_RATE)
    rolling = prediction.rolling_conformal(
        rolling, alpha=CONFORMAL_ALPHA, window=CONFORMAL_CALIBRATION_WINDOW,
        method=CONFORMAL_METHOD)
    rolling['set'] = name
    rolling['staleness'] = pd.cut(rolling['staleness_d'], staleness_edges,
                                  labels=['0-7 d', '8-14 d', '15-21 d', '22-30 d'])
    rolling_sets[name] = rolling
    print(f'{name}: {len(rolling):,} rows from {rolling["origin"].nunique()} refits')

all_rolling = pd.concat(rolling_sets.values(), ignore_index=True)
nowcast = prediction.score_predictions(
    all_rolling.dropna(subset=['q05', 'q95']), ['set', 'staleness'],
    alpha=CONFORMAL_ALPHA)
display(nowcast)
nowcast.to_csv(OUTPUT_DIR / 'GM_09_nowcast_metrics.csv', index=False)
tables.write_table(nowcast, str(OUTPUT_DIR / 'GM_09_body.tex'),
                   [('set', tables.texttt), ('staleness', tables.texttt), ('n', ',d'),
                    ('mae', '.2f'), ('rmse', '.2f'), ('bias', '.2f'),
                    ('coverage_q05_q95', tables.percent), ('width_q05_q95', '.1f'),
                    ('interval_score', '.1f')])

# %%
view = rolling_sets['str'].set_index('ds').loc['2025-11-01':'2025-12-01']
figures.plot_prediction_band(
    view['y'], view['yhat'], view['q05'], view['q95'], freq=NATIVE_FREQ,
    title='Observed against expected, on-structure set, November 2025',
    save_path=str(OUTPUT_DIR), filename='GM_F08_observed_expected')

# Native conformal prediction, once, as the diagnostic counterpart (D14).
model_str, train_str_fit, _ = models_a['str']
split = int(len(train_str_fit) * 0.8)
passthrough = ('tair', 'rh', 'sr') + (tuple(CONDITIONS.values()) if CONDITIONS else ())
native = model_str.conformal_predict(
    prediction._model_frame(def_frames['str'].iloc[len(train_str_fit):], passthrough),
    calibration_df=prediction._model_frame(train_str_fit.iloc[split:], passthrough),
    alpha=CONFORMAL_ALPHA, method=CONFORMAL_METHOD)
model_str.conformal_plot(native)
```

Check `score_predictions`'s column names for the interval (`coverage_q05_q95`, `width_q05_q95`, `interval_score`, per `NP_07_nowcast_metrics.csv`) and adjust the body columns if they differ.

- [ ] **Step 2: Sync, execute (this is the long run: ~70 refits × 3 sets; timeout 43200 s; run in the background and poll), read back, copy back. Commit** `feat(study05): the rolling expectation with rolling conformal bounds, scored by staleness`.

### Task 3.3 (O): Report §8 Uncertainty and validation

- [ ] From `GM_09_body.tex`, `GM_07_body.tex`, `GM_F03_fit_metrics.png`, `GM_F08_observed_expected.png`: the fit curve, the folds and component stability, the rolling expectation's accuracy and the interval's coverage by staleness, the comparison with Study 04's 67.7 %. Build, honesty test, README status, commit `docs(study05): write the uncertainty and validation section`.

> **Checkpoint 3 (O):** coverage within a stated tolerance of 90 %, or the shortfall explained by staleness. Approval opens Phase 4.

---

# Phase 4 · Model B, the learned impulse response (spec D9)

### Task 4.1 (S): `lagged_regressor_weights`, `impulse_response_summary`, `figures.plot_impulse_response`

**Files:**
- Modify: `studies/shmlib/prediction.py` (append two functions), `studies/shmlib/figures.py` (append one)
- Create: `studies/05_greybox_monitoring/tests/test_model_b.py`; modify `studies/shmlib/tests/test_shmlib.py` (one axes test)

**Interfaces:**
- `prediction.lagged_regressor_weights(model) -> pd.DataFrame` with `regressor`, `lag` (1 = the most recent past sample), `weight`, from `model.model.get_covar_weights()` (dict of name → tensor shaped `(n_forecasts, n_lags)`; take the first forecast row; NeuralProphet orders the lags oldest first, so `lag = n_lags − i`). The test below settles the orientation: if the peak lands at `n_lags − 3 + 1` instead of 3, flip the mapping and keep the test.
- `prediction.impulse_response_summary(weights, dt_hours) -> pd.DataFrame` with `regressor`, `gain` (Σ weight), `delay_h` (Σ w·lag·dt / Σ w), `tau_h` (one-pole time constant fitted to the cumulative response by grid search over 0.1–48 h), `r2_onepole`.
- `figures.plot_impulse_response(weights, summary, reference=None, dt_hours=1/3, title='', save_path=None, filename=None)`: one panel per regressor, weights against lag in hours, with Study 03's operator (a `reference` dict `{name: {'delay_h': d, 'tau_h': t}}`) drawn as the equivalent normalised one-pole response through `coupling.thermal_operator` on a unit impulse.

- [ ] **Step 1: Write the failing tests**

```python
"""
Tests for Model B's learned impulse response (spec D9).

Run from studies/:  python 05_greybox_monitoring/tests/test_model_b.py
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


class TestLaggedWeights(unittest.TestCase):

    def test_the_peak_weight_sits_at_the_injected_lag(self):
        rng = np.random.default_rng(0)
        n = 24 * 60
        index = pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC')
        x = rng.normal(0, 1, n)
        y = 5.0 + 3.0 * np.roll(x, 3) + rng.normal(0, 0.1, n)
        frame = pd.DataFrame({'y': y, 'x': x}, index=index)
        model, _ = prediction.neuralprophet_backtest(
            frame, frame, regressors=(), task='nowcast', epochs=20, freq='1h',
            yearly=False, daily_order=1, quantiles=(), lagged_regressors=('x',),
            lagged_n_lags=8, learning_rate=0.05)
        weights = prediction.lagged_regressor_weights(model)
        peak = int(weights.loc[weights['weight'].abs().idxmax(), 'lag'])
        self.assertEqual(peak, 3)

    def test_summary_reads_delay_and_time_constant_from_a_known_response(self):
        lags = np.arange(1, 25)
        w = 2.0 * np.exp(-(lags - 1) / 4.0) * (lags >= 1)
        weights = pd.DataFrame({'regressor': 'x', 'lag': lags, 'weight': w})
        summary = prediction.impulse_response_summary(weights, dt_hours=1.0)
        self.assertAlmostEqual(summary.loc[0, 'tau_h'], 4.0, delta=1.0)
        self.assertGreater(summary.loc[0, 'gain'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Run; expected AttributeError.**

- [ ] **Step 3: Implement**

```python
def lagged_regressor_weights(model):
    """
    The weight per lag of every lagged regressor, one row per lag.

    Read from ``model.model.get_covar_weights()``; lag 1 is the most recent
    past sample. This is the data behind
    ``plot_parameters(components=['lagged_regressors'])``.

    Returns
    -------
    pd.DataFrame
        ``regressor``, ``lag``, ``weight``.
    """
    rows = []
    for name, tensor in model.model.get_covar_weights().items():
        values = np.asarray(tensor.detach().cpu().numpy(), dtype=float)
        values = values.reshape(values.shape[0], -1)[0]
        n_lags = values.size
        for i, weight in enumerate(values):
            rows.append({'regressor': name, 'lag': n_lags - i, 'weight': float(weight)})
    out = pd.DataFrame(rows, columns=['regressor', 'lag', 'weight'])
    return out.sort_values(['regressor', 'lag']).reset_index(drop=True)


def impulse_response_summary(weights, dt_hours):
    """
    Effective delay and time constant read from a learned impulse response.

    The delay is the first moment of the weights in hours; the time constant
    is the one-pole fit ``G (1 - exp(-t / tau))`` to the cumulative response,
    found by grid search. Both are the quantities Study 03 measured by its
    delay-and-time-constant scan, now read from what the model learned.

    Returns
    -------
    pd.DataFrame
        ``regressor``, ``gain``, ``delay_h``, ``tau_h``, ``r2_onepole``.
    """
    rows = []
    taus = np.concatenate([np.arange(0.1, 2.0, 0.1), np.arange(2.0, 48.5, 0.5)])
    for name, group in weights.groupby('regressor'):
        group = group.sort_values('lag')
        w = group['weight'].to_numpy(dtype=float)
        t = group['lag'].to_numpy(dtype=float) * float(dt_hours)
        gain = float(w.sum())
        delay = float((w * t).sum() / w.sum()) if w.sum() != 0 else np.nan
        cumulative = np.cumsum(w)
        best_tau, best_sse = np.nan, np.inf
        for tau in taus:
            fitted = gain * (1.0 - np.exp(-t / tau))
            sse = float(((cumulative - fitted) ** 2).sum())
            if sse < best_sse:
                best_tau, best_sse = float(tau), sse
        total = float(((cumulative - cumulative.mean()) ** 2).sum())
        rows.append({'regressor': name, 'gain': gain, 'delay_h': delay,
                     'tau_h': best_tau,
                     'r2_onepole': 1.0 - best_sse / total if total > 0 else np.nan})
    return pd.DataFrame(rows, columns=['regressor', 'gain', 'delay_h', 'tau_h', 'r2_onepole'])
```

```python
def plot_impulse_response(weights, summary, reference=None, dt_hours=1.0 / 3.0,
                          title='', save_path=None, filename=None):
    """One panel per driver: the learned weight per lag, with Study 03's operator as a normalised one-pole response."""
    from shmlib import coupling as _coupling

    names = list(weights['regressor'].unique())
    fig, axes = plt.subplots(1, len(names), figsize=viz.figsize(viz.FIGURE_WIDTH, 2.6),
                             squeeze=False)
    for ax, name in zip(axes[0], names):
        group = weights[weights['regressor'] == name].sort_values('lag')
        hours = group['lag'].to_numpy() * dt_hours
        colour = viz.driver_colour(name)
        ax.bar(hours, group['weight'], width=dt_hours * 0.9, color=colour)
        if reference and name in reference:
            impulse = pd.Series(0.0, index=pd.RangeIndex(len(group) + 1))
            impulse.iloc[0] = 1.0
            response = _coupling.thermal_operator(
                impulse, delay=int(round(reference[name]['delay_h'] / dt_hours)),
                tau=reference[name]['tau_h'], dt_hours=dt_hours).iloc[1:]
            scale = group['weight'].sum() / response.sum() if response.sum() else 1.0
            ax.plot(hours, response.to_numpy() * scale, color=viz.MARK_COLOUR,
                    linewidth=1.2, label='Study 03 operator')
        ax.set_title(name, fontsize='small')
        ax.set_xlabel('Lag [h]')
        ax.set_ylabel('Weight')
        viz.format_spines(ax)
    axes[0][0].legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, -0.30),
                      frameon=False)
    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig
```

- [ ] **Step 4: Run; PASS. Commit** `feat(shmlib): read and summarise the impulse response a lagged-regressor model learned`.

### Task 4.2 (S runs, O interprets): Movement 4 — Model B, `GM_10`, `GM_F07`

- [ ] **Step 1: Append the cells**

```python
# %% [markdown]
# ## Movement 4 · Does the wall answer with a delay the 20-minute grid can resolve?
#
# The same specification, air temperature and radiation entering as lagged
# regressors over `LAGGED_N_LAGS` slots (D9). The learned weight per lag is
# the impulse response; its first moment and one-pole fit are set beside
# Study 03's operator.

# %%
block_b = def_frames['str']
slots_per_hour = int(pd.Timedelta(hours=1) / pd.Timedelta(NATIVE_FREQ))
model_b, _ = prediction.neuralprophet_backtest(
    block_b, block_b, regressors=('rh',), task='nowcast', epochs=EPOCHS,
    freq=NATIVE_FREQ, growth='linear',
    changepoints=prediction.covered_changepoints(block_b.index, N_CHANGEPOINTS),
    n_changepoints=N_CHANGEPOINTS, changepoints_range=CHANGEPOINTS_RANGE,
    trend_reg=TREND_REG, yearly_order=YEARLY_ORDER, daily_order=DAILY_ORDER,
    conditional_seasonality=CONDITIONS, quantiles=(), seed=SEED,
    learning_rate=LEARNING_RATE, lagged_regressors=('tair', 'sr'),
    lagged_n_lags=LAGGED_N_LAGS, lagged_regularization=LAGGED_REG)
weights_b = prediction.lagged_regressor_weights(model_b)
summary_b = prediction.impulse_response_summary(weights_b, dt_hours=1.0 / slots_per_hour)
STUDY03_OPERATOR = {'tair': {'delay_h': 0.0, 'tau_h': 0.0},
                    'sr': {'delay_h': 1.0, 'tau_h': 0.0}}
summary_b['study03_delay_h'] = [STUDY03_OPERATOR[r]['delay_h'] for r in summary_b['regressor']]
summary_b['study03_tau_h'] = [STUDY03_OPERATOR[r]['tau_h'] for r in summary_b['regressor']]
display(summary_b)
weights_b.to_csv(OUTPUT_DIR / 'GM_10_impulse_response_weights.csv', index=False)
summary_b.to_csv(OUTPUT_DIR / 'GM_10_impulse_response.csv', index=False)
tables.write_table(summary_b, str(OUTPUT_DIR / 'GM_10_body.tex'),
                   [('regressor', tables.texttt), ('gain', '.3f'), ('delay_h', '.2f'),
                    ('tau_h', '.2f'), ('r2_onepole', '.3f'),
                    ('study03_delay_h', '.1f'), ('study03_tau_h', '.1f')])
figures.plot_impulse_response(weights_b, summary_b, reference=STUDY03_OPERATOR,
                              dt_hours=1.0 / slots_per_hour,
                              title='Learned impulse response, on-structure set',
                              save_path=str(OUTPUT_DIR), filename='GM_F07_impulse_response')
model_b.set_plotting_backend('plotly-static')
model_b.plot_parameters(components=['lagged_regressors'])
```

If Task 0.1's lagged-regressor test failed, resample `block_b` to `MODEL_B_FALLBACK_FREQ` with `.resample(MODEL_B_FALLBACK_FREQ).mean()` and use `lagged_n_lags=MODEL_B_FALLBACK_LAGS`, `dt_hours=1.0`, and say so in the printed summary.

- [ ] **Step 2: Sync, execute, copy back, commit** `feat(study05): learn the impulse response of air temperature and radiation at twenty minutes`.

### Task 4.3 (O): Report §7 Impulse response

- [ ] From `GM_10_body.tex` and `GM_F07_impulse_response.png`: the learned delay and time constant per driver against Study 03's cell; agreement validates the operator imposed in Model A, disagreement is stated as a finding. Build, honesty test, README status, commit `docs(study05): write the impulse response section`.

> **Checkpoint 4 (O):** `GM_10` beside Study 03's operator, agreement or finding stated. Approval opens Phase 5.

---

# Phase 5 · The monitor, three charts (spec D10, D11)

### Task 5.1 (S): `monitoring.prewhiten`

**Files:**
- Modify: `studies/shmlib/monitoring.py` (append)
- Create: `studies/05_greybox_monitoring/tests/test_monitor.py`

**Interfaces:**
- `monitoring.prewhiten(residuals, phi=None, start=None, end=None, freq='20min') -> (pd.Series, float)`: innovations `e(t) = r(t) − φ·r(t − 1 step)`, with `φ` the lag-1 autocorrelation of the residual over `[start, end]` when `phi` is `None`; `e` is missing wherever `r(t)` or `r(t − 1 step)` is missing, so no gap is bridged.

- [ ] **Step 1: Write the failing test**

```python
"""
Tests for the monitor additions of Study 05 (spec D10, D11).

Run from studies/:  python 05_greybox_monitoring/tests/test_monitor.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import monitoring  # noqa: E402


def _ar1(n=20000, phi=0.99, seed=0, freq='20min'):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1.0, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + e[i]
    index = pd.date_range('2019-01-01', periods=n, freq=freq, tz='UTC')
    return pd.Series(r, index=index)


class TestPrewhiten(unittest.TestCase):

    def test_innovations_of_an_ar1_are_white_and_phi_is_recovered(self):
        residual = _ar1()
        innovations, phi = monitoring.prewhiten(residual, start='2019-01-01', end='2019-06-30')
        self.assertAlmostEqual(phi, 0.99, delta=0.01)
        self.assertLess(abs(innovations.autocorr(1)), 0.05)
        self.assertAlmostEqual(innovations.std(), 1.0, delta=0.05)

    def test_a_gap_is_not_bridged(self):
        residual = _ar1(n=500)
        residual.iloc[200:210] = np.nan
        innovations, _ = monitoring.prewhiten(residual, phi=0.99)
        self.assertTrue(innovations.iloc[200:211].isna().all())
        self.assertFalse(np.isnan(innovations.iloc[211]))
```

- [ ] **Step 2: Run; expected AttributeError. Step 3: Implement**

```python
def prewhiten(residuals, phi=None, start=None, end=None, freq='20min'):
    """
    The part of the residual its own previous value could not predict.

    Control-chart limits are derived for independent samples, and this
    project's residual is nothing of the kind (lag-1 autocorrelation 0.995
    at 20 minutes in Study 04). Charting ``e(t) = r(t) - phi r(t - 1)``
    restores the assumption for sudden departures; slow ones need the daily
    and slow charts instead (spec D10).

    Parameters
    ----------
    residuals : pd.Series
        Residual on a regular grid.
    phi : float or None, optional
        AR(1) coefficient. ``None`` estimates it as the lag-1 autocorrelation
        over ``[start, end]``. Default ``None``.
    start, end : str or pd.Timestamp or None, optional
        Reference window for the estimate.
    freq : str, optional
        Grid spacing; a previous sample farther than this is a gap. Default
        ``'20min'``.

    Returns
    -------
    (pd.Series, float)
        Innovations aligned to ``residuals``, and the ``phi`` used.
    """
    values = pd.to_numeric(residuals, errors='coerce')
    if phi is None:
        window = values.loc[start:end].dropna()
        phi = float(window.autocorr(1))
    previous = values.shift(1, freq=freq).reindex(values.index)
    innovations = values - float(phi) * previous
    return innovations, float(phi)
```

- [ ] **Step 4: Run; PASS. `python shmlib/tests/test_monitoring.py` unchanged. Commit** `feat(shmlib): prewhiten a residual with its reference AR(1) coefficient`.

### Task 5.2 (S): `monitoring.channel_coincidence`

**Interfaces:**
- `monitoring.channel_coincidence(alarm, channels, scale_start=None, scale_end=None, window='24h', threshold=5.0, instrument=('batt',), environment=('tair', 'rh')) -> pd.Series` of strings aligned to `alarm[alarm]`: each channel in `channels` (a DataFrame) is reduced to its departure from its centred rolling median over `window`, scaled by `1.4826 · median|departure|` over `[scale_start, scale_end]` (Study 01's test, Equations 1–2 of its anomaly section); a slot is an excursion where `|departure| > threshold · scale`. An alarm slot is `'instrument'` if any `instrument` channel is in excursion in the same slot, else `'environment'` if any `environment` channel is, else `'unattributed'`.

- [ ] **Step 1: Test** (append to `test_monitor.py`)

```python
class TestChannelCoincidence(unittest.TestCase):

    def test_alarms_are_attributed_by_the_coincident_channel(self):
        index = pd.date_range('2024-01-01', periods=1000, freq='20min', tz='UTC')
        rng = np.random.default_rng(1)
        channels = pd.DataFrame({'tair': rng.normal(0, 0.1, 1000),
                                 'rh': rng.normal(0, 0.1, 1000),
                                 'batt': rng.normal(0, 0.001, 1000)}, index=index)
        channels.loc[index[500], 'batt'] += 1.0     # instrument excursion
        channels.loc[index[700], 'tair'] += 10.0    # environment excursion
        alarm = pd.Series(False, index=index)
        alarm.iloc[[500, 700, 900]] = True
        out = monitoring.channel_coincidence(alarm, channels)
        self.assertEqual(out.loc[index[500]], 'instrument')
        self.assertEqual(out.loc[index[700]], 'environment')
        self.assertEqual(out.loc[index[900]], 'unattributed')
```

- [ ] **Step 2: Implement**

```python
def channel_coincidence(alarm, channels, scale_start=None, scale_end=None,
                        window='24h', threshold=5.0, instrument=('batt',),
                        environment=('tair', 'rh')):
    """
    What else moved in the slot of each alarm: the instrument, the weather, or nothing.

    Study 01 settled the summer 2026 excursions by asking whether the
    environmental and supply channels carried them too. This applies that
    test to every fast-chart alarm, so the episode table can say which alarms
    are the wall's to answer for (spec D10).

    Returns
    -------
    pd.Series
        ``'instrument'``, ``'environment'`` or ``'unattributed'`` for every
        slot where ``alarm`` is true.
    """
    alarm = alarm.astype(bool)
    excursions = {}
    for column in channels.columns:
        series = pd.to_numeric(channels[column], errors='coerce')
        departure = series - series.rolling(window, center=True, min_periods=1).median()
        scale = 1.4826 * departure.loc[scale_start:scale_end].abs().median()
        excursions[column] = departure.abs() > float(threshold) * float(scale)
    excursions = pd.DataFrame(excursions).reindex(alarm.index).fillna(False)
    labels = pd.Series('unattributed', index=alarm.index[alarm])
    env = excursions[[c for c in environment if c in excursions]].any(axis=1)
    ins = excursions[[c for c in instrument if c in excursions]].any(axis=1)
    labels[env.reindex(labels.index).fillna(False).to_numpy()] = 'environment'
    labels[ins.reindex(labels.index).fillna(False).to_numpy()] = 'instrument'
    return labels
```

- [ ] **Step 3: Run; PASS. Commit** `feat(shmlib): attribute each alarm to the instrument, the environment or neither`.

### Task 5.3 (S): `detectability_curve(statistic=…)` and `figures.plot_daily_harmonic_chart`

**Files:**
- Modify: `studies/shmlib/monitoring.py:495-640` (`detectability_curve`), `studies/shmlib/figures.py` (append), tests.

**Interfaces:**
- `detectability_curve(residuals, mu, sigma, magnitudes, durations, freq='20min', lam=0.2, L=3.0, k=0.5, h=5.0, seed=0, kind='pulse', period='24h', response_window='24h', statistic='residual', phi=None, injection_starts=None, min_slots=60)`: with `statistic='residual'` (default) today's behaviour. Otherwise the contaminated and the uncontaminated series are both transformed before charting: `'innovation'` → `prewhiten(series, phi=phi)[0]` on the 20-minute grid; `'daily_amplitude'` / `'daily_phase'` → the `amplitude` / `phase_h` column of `daily_harmonic(series, min_slots)` on a daily grid; `'daily_mean'` → `series.resample('1D').mean()`. For the daily statistics the chart frequency is `'1D'`, the joint window one day, and `delay_h` is counted in days × 24. With `injection_starts` (sequence of timestamps) the sweep runs once per start and `detected` is the fraction of starts detected, `delay_h` their mean; `None` keeps the single mid-record injection.

- [ ] **Step 1: Test** (append to `test_monitor.py`)

```python
class TestStatisticAwareDetectability(unittest.TestCase):

    def test_innovation_statistic_finds_a_step_the_raw_residual_needs_a_wide_limit_for(self):
        residual = _ar1(n=30000, phi=0.99)
        innovations, phi = monitoring.prewhiten(residual, start='2019-01-01', end='2019-04-30')
        reference = monitoring.reference_stats(innovations, start='2019-01-01', end='2019-04-30')
        curve = monitoring.detectability_curve(
            residual, reference['mu'], reference['sigma'], magnitudes=(8.0,),
            durations=('24h',), kind='step', statistic='innovation', phi=phi,
            lam=0.2, L=4.0)
        self.assertTrue(bool(curve.loc[0, 'detected']))

    def test_daily_amplitude_statistic_runs_on_a_daily_grid(self):
        index = pd.date_range('2019-01-01', periods=72 * 400, freq='20min', tz='UTC')
        hours = index.hour + index.minute / 60.0
        rng = np.random.default_rng(2)
        residual = pd.Series(2.0 * np.cos(2 * np.pi * (hours - 14) / 24.0)
                             + rng.normal(0, 0.5, len(index)), index=index)
        daily = monitoring.daily_harmonic(residual)['amplitude']
        reference = monitoring.reference_stats(daily, start='2019-01-01', end='2019-06-30')
        curve = monitoring.detectability_curve(
            residual, reference['mu'], reference['sigma'], magnitudes=(6.0,),
            durations=('168h',), kind='amplitude', statistic='daily_amplitude',
            injection_starts=('2019-09-01', '2019-11-01'))
        self.assertIn(curve.loc[0, 'detected'], (0.5, 1.0, True))
```

- [ ] **Step 2: Implement.** Add a private helper above `detectability_curve`:

```python
def _monitor_statistic(series, statistic, phi=None, freq='20min', min_slots=60):
    """Reduce a residual to the series a chart is built on, and say its grid."""
    if statistic == 'residual':
        return series, freq
    if statistic == 'innovation':
        return prewhiten(series, phi=phi, freq=freq)[0], freq
    if statistic in ('daily_amplitude', 'daily_phase'):
        column = 'amplitude' if statistic == 'daily_amplitude' else 'phase_h'
        return daily_harmonic(series, min_slots=min_slots)[column], '1D'
    if statistic == 'daily_mean':
        return series.resample('1D').mean(), '1D'
    raise ValueError(f'unknown statistic {statistic!r}')
```

Then in `detectability_curve`: add the four keyword arguments; compute `base_values, chart_freq = _monitor_statistic(values, statistic, phi, freq, min_slots)` and build `baseline` on `base_values` with `window=chart_freq`; the injection points are `[index[len(index) // 2]]` or `pd.DatetimeIndex(injection_starts)`; for each (magnitude, duration) loop over the injection points, contaminate the **raw** `values`, transform with `_monitor_statistic`, chart, compare with the baseline over `[injection, horizon]`, and aggregate: `detected` = fraction of points detected (a `float`; equal to `1.0`/`0.0` for a single point, which keeps `bool(...)` working for existing callers), `delay_h` = mean delay over the points that detected. Keep the columns `magnitude, duration_h, detected, delay_h` and add `statistic` and `n_starts`.

- [ ] **Step 3: `figures.plot_daily_harmonic_chart(chart_amplitude, chart_phase, episodes=None, title='', save_path=None, filename=None)`** — two stacked control charts (amplitude and phase EWMA with limits), reusing the drawing logic of `plot_control_chart` for each panel (call it with `ax=` if that function accepts an axes; if not, draw the `ewma`, `upper`, `lower` columns directly with the same colours and shade `episodes` spans black at 5 %). One axes-count test.

- [ ] **Step 4: Run all tests; `test_monitoring.py` unchanged (the default path is untouched). Commit** `feat(shmlib): detectability on the statistic each chart is built on, and the daily-harmonic chart figure`.

### Task 5.4 (O designs the cells; S runs the sweeps): Movement 5 — reference, three charts, attribution, detectability; `GM_11`–`GM_13`, `GM_F09`–`GM_F11`, `GM_F13`

- [ ] **Step 1: Append the cells**

```python
# %% [markdown]
# ## Movement 5 · What departure does each chart catch?
#
# The rolling residual of the on-structure set is charted three ways (D10):
# prewhitened innovations at 20 minutes, the daily cycle's amplitude and
# phase at daily cadence, and the daily mean for drift. Reference statistics
# come from the fixed window. Each chart's limit is swept to its own
# false-alarm budget. Detectability is measured per mechanism on the chart
# built for it, with injections sized by the wall's daily response (D11).

# %%
rolling_str = rolling_sets['str'].set_index('ds').sort_index()
residual = (rolling_str['y'] - rolling_str['yhat']).asfreq(NATIVE_FREQ)
reference_slice = residual.loc[REFERENCE_START:REFERENCE_END]
monitored = residual.loc[MONITORED_START:]

charts = {}
# Fast chart: innovations.
innovations, phi = monitoring.prewhiten(residual, start=REFERENCE_START, end=REFERENCE_END)
charts['fast'] = dict(series=innovations, freq=NATIVE_FREQ, budget=BUDGET_FAST_DAYS,
                      lam=EWMA_LAMBDA, joint=JOINT_WINDOW)
# Daily chart: amplitude and phase of the residual's daily cycle.
daily = monitoring.daily_harmonic(residual, min_slots=DAILY_HARMONIC_MIN_SLOTS)
charts['daily_amplitude'] = dict(series=daily['amplitude'], freq='1D', budget=BUDGET_DAILY_DAYS,
                                 lam=0.2, joint='1D')
charts['daily_phase'] = dict(series=daily['phase_h'], freq='1D', budget=BUDGET_DAILY_DAYS,
                             lam=0.2, joint='1D')
# Slow chart: daily mean.
charts['slow'] = dict(series=residual.resample('1D').mean(), freq='1D',
                      budget=BUDGET_SLOW_DAYS, lam=0.1, joint='1D')
print(f'phi = {phi:.4f}; residual sd {reference_slice.std():.2f}, '
      f'innovation sd {innovations.loc[REFERENCE_START:REFERENCE_END].std():.2f}')

# %%
L_CANDIDATES = np.arange(2.0, 15.01, 0.25)
episode_rows, run_rows, tuned = [], [], {}
for name, spec in charts.items():
    series = spec['series']
    reference = monitoring.reference_stats(series, start=REFERENCE_START, end=REFERENCE_END)
    in_control = series.loc[REFERENCE_START:REFERENCE_END]
    budget = []
    for candidate in L_CANDIDATES:
        ewma = monitoring.ewma_chart(in_control, reference['mu'], reference['sigma'],
                                     lam=spec['lam'], L=candidate)
        cusum = monitoring.cusum_chart(in_control, reference['mu'], reference['sigma'],
                                       k=CUSUM_K, h=CUSUM_H)
        joint = monitoring.joint_alarm(ewma['alarm'], cusum['alarm'], window=spec['joint'])
        budget.append({'L': candidate, **monitoring.average_run_length(joint, freq=spec['freq'])})
    budget = pd.DataFrame(budget)
    meeting = budget.loc[budget['arl_days'] >= spec['budget'], 'L']
    if meeting.empty:
        raise RuntimeError(f'{name}: no limit meets {spec["budget"]} days; reference not in control')
    L = float(meeting.min())
    watched = series.loc[MONITORED_START:]
    ewma = monitoring.ewma_chart(watched, reference['mu'], reference['sigma'], lam=spec['lam'], L=L)
    cusum = monitoring.cusum_chart(watched, reference['mu'], reference['sigma'], k=CUSUM_K, h=CUSUM_H)
    joint = monitoring.joint_alarm(ewma['alarm'], cusum['alarm'], window=spec['joint'])
    episodes = monitoring.alarm_episodes(joint, watched).assign(chart=name)
    run = monitoring.average_run_length(joint.loc[REFERENCE_START:REFERENCE_END]
                                        if name == 'fast' else joint, freq=spec['freq'])
    tuned[name] = dict(reference=reference, L=L, ewma=ewma, cusum=cusum, joint=joint, episodes=episodes)
    run_rows.append({'chart': name, 'L': L, 'budget_days': spec['budget'],
                     'achieved_arl_days': budget.loc[budget['L'] == L, 'arl_days'].item(),
                     'episodes_in_reference': budget.loc[budget['L'] == L, 'n_episodes'].item()
                     if 'n_episodes' in budget else np.nan})
    episode_rows.append(episodes)

# Attribution of the fast chart's alarms.
attribution_channels = pd.DataFrame({
    'tair': frame['tair_str'], 'rh': frame['rh_str'],
    'batt': sensor['batt_str'].reindex(frame.index)}).loc[MONITORED_START:]
labels = monitoring.channel_coincidence(tuned['fast']['joint'], attribution_channels,
                                        scale_start=REFERENCE_START, scale_end=REFERENCE_END)
fast_episodes = tuned['fast']['episodes']
fast_episodes['attribution'] = [
    labels.loc[start:end].mode().iloc[0] if not labels.loc[start:end].empty else 'unattributed'
    for start, end in zip(fast_episodes['start'], fast_episodes['end'])]
episodes = pd.concat(episode_rows, ignore_index=True)
runs = pd.DataFrame(run_rows)
display(episodes); display(runs)
episodes.to_csv(OUTPUT_DIR / 'GM_11_alarm_episodes.csv', index=False)
runs.to_csv(OUTPUT_DIR / 'GM_12_run_lengths.csv', index=False)
tables.write_table(episodes, str(OUTPUT_DIR / 'GM_11_body.tex'),
                   [('chart', tables.texttt), ('start', tables.date_cell('start')),
                    ('end', tables.date_cell('end')), ('duration_h', ',.0f'),
                    ('mean_z', '.2f'), ('attribution', tables.texttt)])
tables.write_table(runs, str(OUTPUT_DIR / 'GM_12_body.tex'),
                   [('chart', tables.texttt), ('L', '.2f'), ('budget_days', '.0f'),
                    ('achieved_arl_days', '.0f')])

# %%
figures.plot_control_chart(tuned['fast']['ewma'], statistic='ewma',
                           episodes=tuned['fast']['episodes'], freq=NATIVE_FREQ,
                           title='Fast chart: prewhitened innovations',
                           save_path=str(OUTPUT_DIR), filename='GM_F09_fast_chart')
figures.plot_daily_harmonic_chart(tuned['daily_amplitude']['ewma'], tuned['daily_phase']['ewma'],
                                  episodes=pd.concat([tuned['daily_amplitude']['episodes'],
                                                      tuned['daily_phase']['episodes']]),
                                  title='Daily chart: amplitude and phase of the daily cycle',
                                  save_path=str(OUTPUT_DIR), filename='GM_F10_daily_chart')
figures.plot_control_chart(tuned['slow']['cusum'], statistic='cusum',
                           episodes=tuned['slow']['episodes'], freq='1D',
                           title='Slow chart: daily-mean residual',
                           save_path=str(OUTPUT_DIR), filename='GM_F11_slow_chart')

# %%
# Detectability per mechanism, on the chart built for it, at several dates.
injection_dates = DETECT_INJECTION_DATES or ('2023-01-15', '2023-04-15', '2023-07-15', '2023-10-15')
daily_response = (components_a['str']['future_regressor_tair']
                  + components_a['str'].filter(like='daily').sum(axis=1))
response_amplitude = monitoring.daily_harmonic(coupling.diurnal_band(daily_response, window=72))['amplitude']
mechanisms = [
    ('amplitude', 'daily_amplitude', DETECT_MAGNITUDES),
    ('phase', 'daily_phase', tuple(monitoring.phase_shift_amplitude(
        float(response_amplitude.loc[[pd.Timestamp(d, tz='UTC').floor('D') for d in injection_dates]].mean()), h)
        for h in DETECT_PHASE_SHIFTS_H)),
    ('drift', 'slow', DETECT_DRIFT_RATES),
    ('step', 'fast', DETECT_MAGNITUDES),
]
detect_rows = []
for kind, chart_name, magnitudes in mechanisms:
    spec, fit = charts[chart_name], tuned[chart_name]
    statistic = {'fast': 'innovation', 'daily_amplitude': 'daily_amplitude',
                 'daily_phase': 'daily_phase', 'slow': 'daily_mean'}[chart_name]
    curve = monitoring.detectability_curve(
        residual.loc[REFERENCE_START:REFERENCE_END], fit['reference']['mu'], fit['reference']['sigma'],
        magnitudes=magnitudes, durations=DETECT_DURATIONS, freq=NATIVE_FREQ,
        lam=spec['lam'], L=fit['L'], k=CUSUM_K, h=CUSUM_H, kind=kind,
        statistic=statistic, phi=phi, response_window=DETECT_RESPONSE_WINDOW,
        injection_starts=[d for d in injection_dates if REFERENCE_START <= d <= REFERENCE_END] or None,
        min_slots=DAILY_HARMONIC_MIN_SLOTS)
    detect_rows.append(curve.assign(mechanism=kind, chart=chart_name))
detectability = pd.concat(detect_rows, ignore_index=True)
display(detectability)
detectability.to_csv(OUTPUT_DIR / 'GM_13_detectability.csv', index=False)
tables.write_table(detectability, str(OUTPUT_DIR / 'GM_13_body.tex'),
                   [('mechanism', tables.texttt), ('chart', tables.texttt),
                    ('magnitude', '.2f'), ('duration_h', '.0f'), ('detected', '.2f'),
                    ('delay_h', '.1f')])
figures.plot_detectability(detectability[detectability['mechanism'] == 'amplitude'],
                           title='Amplitude growth on the daily chart',
                           save_path=str(OUTPUT_DIR), filename='GM_F13_detectability_amplitude')
figures.plot_detectability(detectability[detectability['mechanism'] == 'phase'],
                           title='Phase change on the daily chart',
                           save_path=str(OUTPUT_DIR), filename='GM_F13_detectability_phase')
figures.plot_detectability(detectability[detectability['mechanism'] == 'drift'],
                           title='Drift on the slow chart',
                           save_path=str(OUTPUT_DIR), filename='GM_F13_detectability_drift')
```

The injection dates must fall inside the reference window (the uncontaminated stretch); replace the four 2023 dates by four dates across the seasons of 2020 and set `DETECT_INJECTION_DATES` in the parameter cell accordingly. Check the column names `average_run_length` returns (`arl_days`, episode count) and `alarm_episodes` returns (`start`, `end`, `duration_h`, `mean_z`) against `NP_09`/`NP_15` of Study 04 and adjust the body columns.

- [ ] **Step 2: Sync, execute (sweeps are long; background, timeout 43200 s), copy back, commit** `feat(study05): three monitoring charts at three time scales, attribution and detectability by mechanism`.

### Task 5.5 (O): Report §10 The monitor

- [ ] From `GM_11`–`GM_13` bodies and `GM_F09`–`GM_F11`, `GM_F13_*`: the reference window and its statistics, the prewhitening coefficient and the limit each chart met, the episodes with attribution (the summer 2026 artefact attributed to the instrument, stated as a demonstration and not a sensitivity), detectability per mechanism on its own chart with the blind spots named. Build, honesty test, README status, commit `docs(study05): write the monitor section`.

> **Checkpoint 5 (O):** run length per chart, smallest departure per mechanism, blind spots. Approval opens Phase 6.

---

# Phase 6 · Outages as hypotheses (spec D12)

### Task 6.1 (S): `prediction.outage_bridge` and `figures.plot_outage_bridge`

**Files:**
- Modify: `studies/shmlib/prediction.py`, `studies/shmlib/figures.py`; create `studies/05_greybox_monitoring/tests/test_bridges.py`

**Interfaces:**
- `prediction.outage_bridge(runner, frame, outages, settle_days=1, window_days=7) -> (table, paths)`: `runner(train, test) -> long predictions` (a closure over `neuralprophet_backtest` with the study's settings, returning the frame with `ds`, `y`, `yhat`, `q05`, `q95`); for each `(start, end)` in `outages`, `train = frame.loc[:start)`, `test = frame.loc[start : end + settle + window]`; `expected` = mean `yhat` and mean `q05`/`q95` over the first `window_days` after `end + settle_days`, `observed` = mean `y` there; `shift = observed − expected`; `verdict` = `'inside'` if `observed` lies in `[mean q05, mean q95]` else `'outside'`; `'no data'` if fewer than 24 observed slots. `paths` is the concatenated long predictions with an `outage` label, for the figure.
- `figures.plot_outage_bridge(paths, table, title='', save_path=None, filename=None)`: one panel per outage, observed before and after, expected with its band through the gap, the resumption window shaded black at 5 %.

- [ ] **Step 1: Test**

```python
"""
Tests for the outage bridge (spec D12).

Run from studies/:  python 05_greybox_monitoring/tests/test_bridges.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import prediction  # noqa: E402


class TestOutageBridge(unittest.TestCase):

    def test_a_level_shift_across_an_outage_is_reported_outside_the_band(self):
        index = pd.date_range('2024-01-01', periods=24 * 90, freq='1h', tz='UTC')
        y = pd.Series(np.zeros(len(index)), index=index)
        y.loc['2024-02-20':] = 10.0                      # the wall moved during the gap
        y.loc['2024-02-01':'2024-02-19'] = np.nan        # the outage
        frame = pd.DataFrame({'y': y})

        def runner(train, test):
            return pd.DataFrame({'ds': test.index, 'y': test['y'].to_numpy(),
                                 'yhat': 0.0, 'q05': -1.0, 'q95': 1.0})

        table, paths = prediction.outage_bridge(
            runner, frame, [('2024-02-01', '2024-02-19')], settle_days=1, window_days=7)
        self.assertEqual(table.loc[0, 'verdict'], 'outside')
        self.assertAlmostEqual(table.loc[0, 'shift'], 10.0, places=6)
        self.assertIn('outage', paths.columns)


if __name__ == '__main__':
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Implement**

```python
def outage_bridge(runner, frame, outages, settle_days=1, window_days=7):
    """
    The expected level at resumption after each outage, against the level seen.

    The model is fitted on everything before the outage; the proxies that
    kept recording carry the expectation through the gap; the first days
    after resumption test it. Nothing is written into the gap (spec D12).

    Parameters
    ----------
    runner : callable
        ``runner(train, test) -> long predictions`` with ``ds``, ``y``,
        ``yhat``, ``q05``, ``q95``.
    frame : pd.DataFrame
        Modelling frame with ``y`` and the regressors.
    outages : sequence of (start, end)
        Inclusive date bounds of each outage.
    settle_days, window_days : int, optional
        Days skipped after resumption (restart transient), then days averaged.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        ``table``: ``outage``, ``start``, ``end``, ``expected``, ``lower``,
        ``upper``, ``observed``, ``shift``, ``n_observed``, ``verdict``;
        ``paths``: the long predictions of every bridge with an ``outage``
        label.
    """
    rows, paths = [], []
    for number, (start, end) in enumerate(outages, 1):
        start, end = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
        resume = end + pd.Timedelta(days=1) + pd.Timedelta(days=settle_days)
        stop = resume + pd.Timedelta(days=window_days)
        train = frame.loc[:start - pd.Timedelta(minutes=1)]
        test = frame.loc[start:stop]
        if train['y'].notna().sum() < 100 or test.empty:
            rows.append({'outage': number, 'start': start, 'end': end,
                         'expected': np.nan, 'lower': np.nan, 'upper': np.nan,
                         'observed': np.nan, 'shift': np.nan, 'n_observed': 0,
                         'verdict': 'no data'})
            continue
        long = runner(train, test)
        long = long.assign(outage=number)
        paths.append(long)
        after = long[(pd.DatetimeIndex(long['ds']) >= resume)
                     & (pd.DatetimeIndex(long['ds']) < stop)]
        observed = after['y'].dropna()
        expected = float(after['yhat'].mean())
        lower, upper = float(after['q05'].mean()), float(after['q95'].mean())
        if observed.size < 24:
            verdict, shift, seen = 'no data', np.nan, np.nan
        else:
            seen = float(observed.mean())
            shift = seen - expected
            verdict = 'inside' if lower <= seen <= upper else 'outside'
        rows.append({'outage': number, 'start': start, 'end': end,
                     'expected': expected, 'lower': lower, 'upper': upper,
                     'observed': seen, 'shift': shift,
                     'n_observed': int(observed.size), 'verdict': verdict})
    table = pd.DataFrame(rows)
    paths = pd.concat(paths, ignore_index=True) if paths else pd.DataFrame(
        columns=['ds', 'y', 'yhat', 'q05', 'q95', 'outage'])
    return table, paths
```

`plot_outage_bridge`: `len(outages)` stacked panels; in each, `y` as points in the inclination colour, `yhat` as a line with `q05`–`q95` filled at low alpha, the resumption window `axvspan` black at 5 %; test asserts one axes per outage in `paths`.

- [ ] **Step 3: Run; PASS. Commit** `feat(shmlib): test the level at resumption after each outage against the bridged expectation`.

### Task 6.2 (S): Movement 6 — `GM_14`, `GM_F12`

```python
# %% [markdown]
# ## Movement 6 · What happened across each outage?
#
# For every outage the station and ERA5 sets carry the expectation through
# the gap; the first `OUTAGE_WINDOW_DAYS` after resumption, less the settle
# day, test it (D12).

# %%
bridge_rows, bridge_paths = [], []
for name in BRIDGE_SETS:
    block = def_frames[name]

    def runner(train, test, _block=block):
        _, out = prediction.neuralprophet_backtest(
            train, test, regressors=('tair', 'rh', 'sr'), task='nowcast',
            epochs=EPOCHS, freq=NATIVE_FREQ, growth='linear',
            changepoints=prediction.covered_changepoints(train.index, N_CHANGEPOINTS),
            n_changepoints=N_CHANGEPOINTS, changepoints_range=CHANGEPOINTS_RANGE,
            trend_reg=TREND_REG, yearly_order=YEARLY_ORDER, daily_order=DAILY_ORDER,
            conditional_seasonality=CONDITIONS, quantiles=QUANTILES, seed=SEED,
            learning_rate=LEARNING_RATE)
        return out

    table, paths = prediction.outage_bridge(runner, block, OUTAGES,
                                            settle_days=OUTAGE_SETTLE_DAYS,
                                            window_days=OUTAGE_WINDOW_DAYS)
    bridge_rows.append(table.assign(set=name))
    bridge_paths.append(paths.assign(set=name))
bridges = pd.concat(bridge_rows, ignore_index=True)
display(bridges)
bridges.to_csv(OUTPUT_DIR / 'GM_14_outage_bridges.csv', index=False)
tables.write_table(bridges, str(OUTPUT_DIR / 'GM_14_body.tex'),
                   [('set', tables.texttt), ('outage', 'd'),
                    ('start', tables.date_cell('start')), ('end', tables.date_cell('end')),
                    ('expected', '.1f'), ('observed', '.1f'), ('shift', '.1f'),
                    ('lower', '.1f'), ('upper', '.1f'), ('verdict', tables.texttt)])
figures.plot_outage_bridge(bridge_paths[0], bridge_rows[0],
                           title=f'Outage bridges, {BRIDGE_SETS[0]} set',
                           save_path=str(OUTPUT_DIR), filename='GM_F12_outage_bridges')
```

Note that the on-structure set has no regressor data inside an outage, which is why `BRIDGE_SETS` excludes it; the model's interval through the gap is the calibrated quantile band of the fit before the outage, since no conformal calibration exists inside the gap, and the report says so. Sync, execute, copy back, commit `feat(study05): bridge every outage as a hypothesis and test it at resumption`.

### Task 6.3 (O): Report §11

- [ ] From `GM_14_body.tex` and `GM_F12_outage_bridges.png`: one paragraph per outage, the verdict per proxy set, the caveat on the interval through the gap. Build, honesty test, README status, commit `docs(study05): write the outages section`.

> **Checkpoint 6 (O):** a verdict per outage and proxy set. Approval opens Phase 7.

---

# Phase 7 · Verdict, limitations, metadata, closure

### Task 7.1 (O): Movement 7 metadata and the last three sections

- [ ] **Step 1: Append the metadata cell**

```python
# %% [markdown]
# ## Movement 7 · Run metadata

# %%
import neuralprophet, scipy
metadata = pd.DataFrame([
    ('window_start', WINDOW_START), ('window_end', str(frame.index.max().date())),
    ('native_freq', NATIVE_FREQ), ('target_column', TARGET_COLUMN),
    ('regressor_sets', ';'.join(f"{k}:{v}" for k, v in REGRESSOR_SETS.items())),
    ('radiation_delay_h', RADIATION_DELAY_H), ('regressor_fill_max_gap', REGRESSOR_FILL_MAX_GAP),
    ('yearly_order', YEARLY_ORDER), ('daily_order', DAILY_ORDER),
    ('conditional_daily_kept', keep_conditional), ('weight_curve', str(WEIGHT_CURVE)),
    ('n_changepoints', N_CHANGEPOINTS), ('changepoints_range', CHANGEPOINTS_RANGE),
    ('trend_reg', TREND_REG), ('epochs', EPOCHS), ('learning_rate', LEARNING_RATE),
    ('seed', SEED), ('min_train', MIN_TRAIN), ('refit_every', REFIT_EVERY),
    ('train_window', TRAIN_WINDOW), ('conformal_alpha', CONFORMAL_ALPHA),
    ('conformal_method', CONFORMAL_METHOD), ('conformal_calibration_window', CONFORMAL_CALIBRATION_WINDOW),
    ('cv_folds', CV_FOLDS), ('lagged_n_lags', LAGGED_N_LAGS), ('lagged_reg', LAGGED_REG),
    ('reference_window', f'{REFERENCE_START} to {REFERENCE_END}'), ('monitored_start', MONITORED_START),
    ('phi', round(phi, 5)), ('budgets_days', f'{BUDGET_FAST_DAYS}/{BUDGET_DAILY_DAYS}/{BUDGET_SLOW_DAYS}'),
    ('limits_L', ';'.join(f"{k}:{v['L']}" for k, v in tuned.items())),
    ('neuralprophet_version', neuralprophet.__version__), ('pandas_version', pd.__version__),
    ('numpy_version', np.__version__), ('scipy_version', scipy.__version__),
], columns=['parameter', 'value'])
metadata.to_csv(OUTPUT_DIR / 'GM_15_run_metadata.csv', index=False)
tables.write_table(metadata, str(OUTPUT_DIR / 'GM_15_body.tex'),
                   [('parameter', tables.texttt), ('value', tables.texttt)])
```

Sync, execute the whole notebook once more end to end (background, timeout 43200 s), copy back.

- [ ] **Step 2: Write §12 Verdict** — one paragraph per question of spec §1, each with its qualification, every number from a `GM_` body.
- [ ] **Step 3: Write §13 Limitations** — the compensation not independent of the fitted term; borrowed radiation; the reference window three years before the instrument change; the interval through a gap being the pre-outage band; one sensor, one site; anything Checkpoints 1b–6 surfaced.
- [ ] **Step 4: Write §14 Run metadata** with `\input{../outputs/GM_15_body.tex}`.

### Task 7.2 (S, O reviews): Closure

- [ ] Set `STUDY_COMPLETE = True` in `tests/test_folder_honesty.py`; run it — every artefact the README names must exist, no `\pending{` may remain. Fix the README's artefact list to what `outputs/` holds (add `GM_04b`, `GM_04c`, `GM_05b`, `GM_06b`, `GM_10_impulse_response_weights`, `GM_F13_detectability_{amplitude,phase,drift}` if they were produced under those names).
- [ ] Provenance and narrative closure: the honesty test's notebook-provenance class passes on the final tree, and the notebook is read top to bottom to confirm that every movement and every step carries its Markdown narrative cell and that every figure and table body the report includes traces to a call in it.
- [ ] README status → `Status: **complete.**` with the closing summary in Study 04's shape (one line per question with its headline number, each traceable). `studies/README.md` study 5 row → `Complete`.
- [ ] Build the PDF twice; run every test file listed in Global Constraints plus the five Study 05 test files; `graphify update .` from `studies/`.
- [ ] Commit `docs(study05): verdict, limitations, run metadata; the study is complete`.

> **Checkpoint 7 (O):** every number traces to a file; the README matches the folder; the report builds; the honesty test passes with `STUDY_COMPLETE = True`.

---

### Task 7.3 (O plans, S implements figures): Revision pass from `report/report05_check.md` — only after Task 7.2

The user keeps revision notes for the report in `studies/05_greybox_monitoring/report/report05_check.md`, section by section, and has ruled (2026-09-06) that they are not acted on until the report's first complete implementation exists. Once Task 7.2 has closed the study, this task reads the file as it stands then (it grows while the study runs), plans one change per note, and applies them under every rule of this plan: a figure a note asks for is a `shmlib.figures` function called from the notebook and exported under a `GM_` name, never drawn on the paper side; a note that asks for a document outside the study (for instance the data-quality report) is planned as its own change with its own commit; the report is rebuilt and the honesty test re-run after the pass. Commit per note or per coherent group of notes.

## Self-review against the spec

- **Coverage.** D1 (Task 1.3 window), D2 (1.3 sets), D3 (1.3 operator; radiation global horizontal by construction), D4 (2b.1), D5 (3.1–3.2 trend on, rolling refit; 5.4 slow chart), D6 (1b.1–1b.6), D7 (2.1 conditional seasonality; 2.4 test), D8 (3.1–3.2 rolling CQR; native `conformal_predict` shown once), D9 (4.1–4.2), D10 (5.1–5.4), D11 (5.3–5.4 statistic-aware, re-sized phase, several dates), D12 (6.1–6.2), D13 (1.1–1.3), D14 (2.3, 2.4, 3.2, 4.2 native plots plus redraws). §6 artefacts: `GM_01`–`GM_16` and `GM_F01`–`GM_F14` each produced by a named task; extra bodies (`GM_04b/c`, `GM_05b`, `GM_06b`, `GM_F13_*`) are declared in Task 7.2. §7 report: every section has a writing task. §8 phases and checkpoints: one per phase, plus 2b. §9 risks: the lagged-regressor fallback (0.1, 4.2), the diagnostic's outcomes (1b.6), the gain kill criterion (2.5), the refit cost (`TRAIN_WINDOW`, 3.2), residual still autocorrelated (5.4 raises if the reference is not in control), reference window not in control (5.4 `RuntimeError`).
- **Deviation from the spec, stated.** D8 says the rolling calibration goes "through `conformal_predict`"; Task 3.1 implements the same conformalised-quantile-regression step (`rolling_conformal`) on the walk-forward output, because `conformal_predict` needs a held-out calibration frame per fit and would either stale the training window by six months or re-use in-sample rows. `conformal_predict` is still executed once per set as the native diagnostic (Task 3.2). The report's §8 states this.
- **Type consistency.** `seasonal_weights(index, modulation=None, peak_doy=196)` is called with a `fit` dict everywhere (1b.6, 2.4, 2b.1, 6.2); `WEIGHT_CURVE` in the parameter cell is that dict. `neuralprophet_backtest`'s new keywords are used with the same names in 2.4, 2b.1, 3.2, 4.2, 6.2. `daily_harmonic(series, min_slots)` in 1b.6, 5.3, 5.4. `detectability_curve(..., statistic, phi, injection_starts, min_slots)` in 5.3 and 5.4. `outage_bridge(runner, frame, outages, settle_days, window_days)` in 6.1 and 6.2.
- **Known checks left to the implementer, each named where it occurs:** `_model_frame`'s column handling (2.1), `paired_mae_skill`'s return type (2b.1), `score_predictions`'s interval column names (3.2), `alarm_episodes`/`average_run_length` column names (5.4), `thermal_operator`'s delay unit (1.3), `plot_control_chart`'s `ax` support (5.3), `crossvalidation_split_df`'s time-zone handling (2.4). Each has an assertion or a one-line adjustment beside it.

## Execution

Plan complete. Two execution options:

1. **Subagent-driven (recommended):** a fresh Sonnet subagent per **S** task under `superpowers:subagent-driven-development`, the orchestrator reviewing between tasks and holding every **O** task and every checkpoint.
2. **Inline:** `superpowers:executing-plans`, batch execution with the same checkpoints.
