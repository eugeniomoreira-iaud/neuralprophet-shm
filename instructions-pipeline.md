# Instructions · Pipeline — `neuralprophet-shm` code repository

Shared, harness-neutral rules for this repository. This file is not auto-loaded directly: it is
imported by the entry-point file each harness discovers by name (`CLAUDE.md`, `GEMINI.md`), which
adds only the delta specific to that harness. Everything below is binding regardless of which
tool is running.

## Operating Protocol (project-wide)

Defined in `instructions-core.md` in the parent folder (one level up, outside this git repo) —
binding here. In short: caveman ultra mode in chat from the first message, full prose for code,
comments, docstrings, commit messages, security warnings and irreversible-action confirmations;
a capable orchestrator that plans and decides, with heavy or mechanical work delegated to
subagents whose spawn prompt includes "respond in caveman ultra mode"; plan first, delegate
second; and no unprompted edits — if the user has not expressly asked for a change, answer the
question first. The model identifiers filling the orchestrator and subagent roles are stated in
the entry point that imported this file.

---

## Overview
This repository implements a **Physics-Informed Grey-Box Framework for Static Structural Health Monitoring (SHM)**. It provides a transferable Python methodology for processing structural sensor data (e.g., inclinometers, strain gauges) and environmental proxies (e.g., ERA5-Land skin temperature, solar radiation) to detect structural anomalies using interpretable machine learning. The `heritageshm` library is not installed as a package — it is imported from the repo root via `sys.path` manipulation in every notebook (see Import Pattern below).

---

## Development Commands

### Environment Setup
All development must be performed within the Conda environment specified in `environment.yml`. The environment name is `neuralprophet_env`. The same specification is used on macOS (Apple Silicon and Intel) and Windows, and resolves to identical package versions on both.

```bash
# Create from scratch
conda env create -f environment.yml
conda activate neuralprophet_env

# Verify the installation (versions, numpy pin, training smoke test)
python verify_env.py

# Update an existing environment after changes to environment.yml
conda activate neuralprophet_env
conda env update -f environment.yml --prune
```

Key packages in the environment: `neuralprophet` (via pip, pinned to 0.8.0), `xgboost`, `scikit-learn`, `statsmodels`, `jupytext`, `watchdog`, and the full scientific Python stack (`numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`).

Two constraints in `environment.yml` are load-bearing and must not be relaxed: the channel list is `conda-forge` only (the `defaults` channel has no `osx-arm64` build of `jupyterlab-github`, so creation fails on Apple Silicon), and numpy is pinned to `>=1.25,<2` (neuralprophet 0.8.0 requires numpy below 2.0; without the pin, pip downgrades numpy underneath Conda-built binaries and breaks them at import time).

### Running the Pipeline
The analysis is driven by a sequential Jupyter Notebook pipeline. Execute the following notebooks in order:

1. `00_Sensor_Preprocessing.ipynb` — Raw data extraction and cleaning.
2. `01_Data_Quality_and_Gaps.ipynb` — Proxy alignment and gap characterization.
3. `02_Proxy_Validation_and_Lags.ipynb` — Thermal lag screening and feature engineering.
4. `03_Imputation_Benchmark.ipynb` — XGBoost virtual sensing and uncertainty quantification.
5. `04_GreyBox_Decomposition_and_Monitoring.ipynb` — NeuralProphet decomposition and anomaly detection.

To launch the interactive environment:
```bash
jupyter lab
```

---

## Jupytext Pairing and the Auto-Watcher

### The Pairing Rule (CRITICAL)
Every Jupyter Notebook (`.ipynb`) is strictly paired with a human-readable Python script (`.py`) in `py:percent` format, configured globally in `jupytext.toml`.

**NEVER edit `.ipynb` files directly.** The underlying JSON structure is heavily prone to corruption during automated edits. Always perform all reading, reasoning, and modifications on the paired `.py` file. The user handles syncing `.py` edits back to `.ipynb` locally.

Paired files:
- `00_Sensor_Preprocessing.ipynb` ↔ `00_Sensor_Preprocessing.py`
- `01_Data_Quality_and_Gaps.ipynb` ↔ `01_Data_Quality_and_Gaps.py`
- `02_Proxy_Validation_and_Lags.ipynb` ↔ `02_Proxy_Validation_and_Lags.py`
- `03_Imputation_Benchmark.ipynb` ↔ `03_Imputation_Benchmark.py`
- `04_GreyBox_Decomposition_and_Monitoring.ipynb` ↔ `04_GreyBox_Decomposition_and_Monitoring.py`

### Auto-Watcher (`auto_watcher.py`)
The repo includes `auto_watcher.py`, a file-system watcher that automatically runs `jupytext --sync` whenever a `.py` or `.ipynb` file is saved, keeping pairs in sync without manual intervention. It requires the `watchdog` package.

To start the watcher (in a dedicated terminal, before editing notebooks) — identical on macOS (zsh) and Windows (PowerShell), once Conda has been initialised for that shell:
```bash
conda activate neuralprophet_env
python auto_watcher.py
```

Leave this terminal running in the background during the session. See `auto_watcher instructions.md` for full details.

---

## Architecture

### Core Library (`heritageshm/`)
The `heritageshm` package contains the functional logic. **Changes to any module affect all notebooks** — never modify function signatures without updating all call sites in the `.py` notebook files.

| Module | Role | Key outputs |
|---|---|---|
| `dataloader.py` | Ingestion of `.adc` sensor files and `.csv` proxy files; saving interim data | Pandas DataFrames |
| `preprocessing.py` | Signal cleaning, resampling, and multi-proxy alignment onto the sensor index | Aligned DataFrame |
| `diagnostics.py` | Gap taxonomy (MCAR/MAR/MNAR), ADF, Engle-Granger cointegration, Ljung-Box tests; gap histogram figure and gap statistics table | Figure `.png`, stats `.csv` |
| `features.py` | Physically motivated thermal inertia lagged-feature generation | Feature DataFrame |
| `imputation.py` | Wrappers for benchmark gap-filling models (XGBoost virtual sensing with conformal bootstrap) | Imputed series, uncertainty bounds |
| `decomposition.py` | NeuralProphet grey-box decomposition: trend, seasonality, AR memory, exogenous regressors | Decomposed DataFrame, model object |
| `monitoring.py` | EWMA and CUSUM control chart implementation for residual-based anomaly detection | Control chart figure, alarm table |
| `viz.py` | Seaborn/Matplotlib visualization utilities; `apply_theme()` for consistent plot styling | Figure objects |

### Import Pattern
The `heritageshm` library is not installed as a package. Every notebook and standalone script must include the following path injection at the top before importing from the library:

```python
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from heritageshm.dataloader import load_preprocessed_sensor
# ... other imports
```

When generating new notebook cells or scripts, always include this block.

### Data Flow
1. **Raw Data**: `data/raw/sensor/` (`.adc` files) and `data/raw/proxies/` (`.csv` files).
2. **Interim Data**: Cleaned/aligned datasets in `data/interim/sensor/` and `data/interim/aligned/`.
3. **Processed Data**: Final feature matrices and imputed series in `data/processed/`.
4. **Outputs**: Plots in `outputs/figures/`, metrics in `outputs/tables/`, models in `outputs/models/`.

> **Important:** `data/` and `outputs/` are gitignored and do not exist in the cloned repository. They must be created locally and populated with data before running any notebook. A `FileNotFoundError` on these paths is expected behaviour on a fresh clone — it is not a code bug.

### Raw Data Documentation (READ BEFORE TOUCHING NOTEBOOK 00)

The raw `.adc` archive is **not** in this repository. It is synchronised by Google Drive into
`_UNIPG/__Mura-realtime/`, outside the repository tree, and is strictly read-only — agent access
is granted read-only, with `Write`, `Edit` and `NotebookEdit` explicitly denied on that path, by
the harness settings file named in the parent folder's entry point. Nothing in this project may
modify the archive, and no `.adc` file is ever committed.

Two documents in `docs/` are binding on any code that reads these files:

| Document | Contents |
|---|---|
| `docs/raw-data-format.md` | Authoritative `.adc` specification: the 14-column and 20-column eras, the 21 February 2025 changeover, channel layout and units, sentinel values, the minimum parsing contract, and (Section 7) the inclinometer cleaning and compensation pipeline |
| `docs/data-quality-report-2026-08-10.md` | Measured state of the archive: outage structure, usable windows, channel defects, and their consequences for the study design |

The five facts most likely to be assumed wrongly:

1. **The absolute level of the inclinometer signal means nothing.** It is set by how the
   instrument sat in its mount at installation. Only changes carry structural information, and the
   first-order difference is the one quantity invariant to every arbitrary constant in the
   pipeline. See `docs/raw-data-format.md` Section 7.1.
2. **`0.000` is a missing-data sentinel**, not a measurement, in every `Batt`, `Tair`, `RH` and
   `I` channel. It must become `NaN` at read time. Solar radiation `n_SR` is the exception — its
   night-time zeros are real.
3. **`n_Twall` = `-55.0` is a probe-failure sentinel** and covers 219 of the 416 days in the
   current era.
4. **The decimal separator is mixed**, sometimes within a single file. It must be normalised per
   field, never detected once per file.
5. **A negative correlation between the inclination and a heating driver is the expected result,
   not an anomaly.** The wall sits in an embankment: the valley side is more sun-exposed, expands
   more under daytime heating, and tips the wall towards the mountain, which is a *negative* change
   by the instrument's convention. Against the **external forcings** (`Tair`, `SR`) lags longer
   than **12 hours** are not physically expected, and a scan that reports one is almost certainly
   returning the 24-hour alias of a lead. **`Twall` is exempt**: it is an internal state variable at
   an unknown depth, not a forcing, so it may legitimately lag the deformation — scan it over
   *signed* delays, then clamp any lead to zero before building a predictive feature. Note also
   that the *raw* channel measures a **positive** slope of +1.63 mdeg/°C against the sign
   expectation — an unresolved contradiction documented in full in `docs/raw-data-format.md`
   Section 7.5, which must be read before any sign or lag is interpreted.

Legacy blocks map to stations in order: b1 = `st01`, b2 = `st02`, b3 = `st03`. The instrument
package installed on 2025-02-21 is at `st02` and replaced legacy b2. The two are different units of
hardware and are kept in separate columns, but **their recorded levels are continuous across the
changeover** — +1.58 mdeg on 30-day window means. Anything that spans both eras must therefore be
compensated and anchored **once**, as a single series; anchoring each unit separately and
concatenating plants a ~126 mdeg step that is arithmetic, not instrumentation. Never estimate and
subtract an "era offset". See `docs/raw-data-format.md` Sections 3.3 and 7.3.

### Output Artifact Naming Convention
All saved figures and tables follow a consistent naming pattern:
```
{notebook_id}_{artifact_id}_{station}_{description}.{ext}
```
Examples: `01_01_st02_gap_histogram.png`, `03_02_st02_imputation_metrics.csv`.

When generating new output cells, follow this convention. The station identifier is controlled by the `TARGET_STATION` parameter at the top of each notebook.

### Primary User Parameter: `TARGET_STATION`
All notebooks are parameterized around `TARGET_STATION` (e.g., `'st02'`). This string controls which sensor file is loaded and which output files are named. It is always defined near the top of each notebook as a clearly marked user input. When modifying notebooks, never hardcode a station identifier — always reference `TARGET_STATION`.

---

## Coding Conventions

### Notebook Structure Standard (Notebooks 01–04)

Notebooks 01 through 04 form the **transferable core pipeline**. They are designed to work with any static structural sensor dataset and any compatible environmental proxy dataset — not only the Gubbio case. When writing or modifying these notebooks, preserve this generality: never hardcode site-specific values, units, or assumptions.

Every notebook in the 01–04 sequence must follow this internal structure:

**1. Title cell (Markdown)**
The first cell must be a Markdown title block containing:
- The notebook number and descriptive title as an H1 heading.
- A one-paragraph statement of the notebook's goal within the overall pipeline.
- A numbered list of the steps the notebook executes, matching the step headers used throughout.

Example pattern (from Notebook 01):
```markdown
# Notebook 01 · Data Quality, Proxies, and Gap Characterization

This notebook executes **Phase A, Step 1** of the `heritageshm` pipeline:

1. **Sensor Loading** — Load the preprocessed sensor CSV from Notebook 00.
2. **Proxy Loading** — Load environmental proxy data and select relevant columns.
3. **Alignment** — Resample and synchronize proxies onto the sensor index.
4. **Gap Characterization** — Classify missing data and diagnose gap taxonomy.
5. **Save** — Export the aligned dataset to `/data/interim/aligned/`.
```

**2. Imports cell (Code)**
One code cell containing all imports and `apply_theme()`. No logic, no parameters.

**3. Step cells (alternating Markdown + Code)**
Each step consists of:
- A Markdown header cell (`## Step N · Name`) containing:
  - A plain-language description of what the step does.
  - A **Parameter Tuning Guidance** subsection (`### Parameter Tuning Guidance`) that documents every user-facing parameter in the following code cell: its name, purpose, accepted values, default, and effect on downstream steps.
- One or more code cells implementing the step by calling library functions. Complex logic must not be written inline — it belongs in a module.

**4. Save / export cell**
The final step must always save all outputs required by the next notebook. Saves must be gated on a success boolean (e.g., `if step_ok: save(...)`) so that partial failures do not silently produce corrupt files.

**5. No inline complex logic**
Notebook cells must not contain multi-function algorithms, statistical tests, or model fitting code written directly in the cell. All such logic must be encapsulated in a function in the appropriate `heritageshm/` module and called from the notebook. A cell should read like an orchestration script, not an implementation.

---

### Notebook 00 — Special Status

`00_Sensor_Preprocessing.ipynb` is intentionally **ad hoc** and site-specific. It encodes the particular raw data format, column layout, file extension, and compensation coefficients of the author's on-site inclinometer system (`.adc` files, tab-separated, temperature-compensation coefficient). It is not expected to be transferable without modification. When working on Notebook 00:
- Do not attempt to generalise it to match the 01–04 pattern.
- Parameters such as `COMP_COEFF`, `STATIONS`, `SEPARATOR`, and `FILE_EXT` are site-specific and must be updated by the user for a new deployment.
- Its output (`{station}_preprocessed.csv` in `data/interim/sensor/`) is the standard entry point for the transferable pipeline starting at Notebook 01.

---

### Expected Dataset Contracts (Notebooks 01–04)

The transferable pipeline (01–04) expects two input datasets with the following structure. Any new sensor or proxy dataset must conform to these contracts before being fed into the pipeline.

**Sensor dataset** (`data/interim/sensor/{station}_preprocessed.csv`)
- Produced by Notebook 00 (or any equivalent preprocessing step).
- A CSV file with a `datetime` column parseable as a `DatetimeIndex`.
- At minimum one structural response column (e.g., `absinc` for absolute inclination). Column name is user-configurable via `TARGET_COL`.
- Regular or near-regular time steps (gaps allowed; the pipeline handles them). Typical resolution: hourly or sub-hourly.
- Units: SI or consistent engineering units. No requirement for specific units, but units must be consistent across the full series.
- No pre-imputation expected: the pipeline assumes this file contains NaN where data is missing.

**Proxy dataset** (`data/raw/proxies/{name}.csv`)
- A CSV file with a datetime column named `datetime (UTC)`, parseable as a `DatetimeIndex`.
- One or more environmental variable columns. Column names and units are user-configurable via `PROXY_COLS`.
- Must cover the full temporal window of the sensor dataset (checked explicitly in Notebook 01 before alignment).
- Typical sources: ERA5-Land reanalysis (skin temperature, solar radiation), local weather station exports, or any reanalysis provider (e.g., Oikolab). The pipeline does not assume a specific source.
- Resolution: hourly or finer. The pipeline resamples to `TARGET_FREQ` during alignment.
- Metadata columns (coordinates, model name, elevation, UTC offset) are automatically dropped during loading in Notebook 01 if listed in `META_COLS`.

---

### Module Coding Standard (`heritageshm/`)

**Module-level docstring**
Every `.py` file in `heritageshm/` must begin with a module-level docstring that states: (1) the module name, (2) its responsibility within the pipeline, and (3) its primary inputs and outputs. Example:

```python
"""
Module: diagnostics.py
Handles gap taxonomy characterization, cointegration testing,
and residual diagnostics (ADF, Ljung-Box).
"""
```

**Function-level docstrings (mandatory)**
Every public function must have a NumPy-style or Google-style docstring containing:
- A one-line summary of what the function does.
- `Parameters` section: name, type, description, and default for every argument.
- `Returns` section: type and description of every return value.
- Any side effects (e.g., saves a file, prints to stdout) must be noted.

Example pattern (already used in `diagnostics.py`):
```python
def test_cointegration(df, target_col, proxy_col, alpha=0.05):
    """
    Performs the Engle-Granger two-step cointegration test to validate
    the physical long-run equilibrium between the structural response and the proxy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing both variables.
    target_col : str
        The structural sensor column (e.g., 'absinc').
    proxy_col : str
        The environmental proxy column (e.g., 'skin_temperature (degC)').
    alpha : float, optional
        Significance level for the test. Default 0.05.

    Returns
    -------
    is_cointegrated : bool
        True if the null hypothesis of no cointegration is rejected.
    p_value : float
        P-value from the Engle-Granger test.
    """
```

**No logic in notebooks**
If a block of code in a notebook cell is longer than ~10 lines, performs a statistical test, trains a model, or produces a figure, it must be refactored into a named function in the appropriate module. The notebook cell then becomes a single function call with named arguments.

**No hardcoded paths in modules**
Module functions must never hardcode file paths. All paths must be passed as arguments. Modules are path-agnostic; notebooks control all I/O paths.

**No side effects without explicit opt-in**
Functions that save files or produce plots must have an optional parameter (e.g., `save_plot_path=None`) that defaults to no side effect. The notebook controls whether outputs are saved.

---

## Studies (`studies/`)

Each numbered folder under `studies/` is one self-contained investigation, with its own library, its own outputs and its own report. `studies/README.md` is their index: it lists the sequence, states what each study asks, and records which retired studies must not be built upon. The rules below are binding on every one of them, current and future.

`studies/shmlib/` is the library shared by all studies, and it is the **only** library the studies have. Every function and every constant lives there — the `.adc` file format and its parsers, the instrument constants, the site and its eras, the solar geometry, the figure conventions, and equally the functions that encode a decision, such as which day to condemn or which lag to accept. There is no per-study library tier and no `shmlib`-versus-study-library judgement to make. `studies/shmlib/` and `heritageshm/` are different libraries with different jobs; the first is the studies' common ground, the second is the production pipeline.

A study folder still owns its notebook, its parameters, its outputs, its report and its tests. What it does not own is code. A decision a study makes is expressed in the **arguments its notebook passes**, not in a private function only that study can call — which is what keeps the decision visible in the notebook, where a reader can argue with it, instead of buried one directory away.

### `/graphify` is how this tree is searched — before anything else

**Any question about `studies/` is a graph query first.** The knowledge graph in `studies/graphify-out/` is
built and current. Before grepping, before globbing, before opening files one at a time, and before
answering from memory of an earlier session, run:

```
/graphify query "<the question>"
```

This is not advice and it is not limited to the moment before new code is written. It applies to every
question of the form "what already does X", "what calls Y", "where does Z live", "what would this change
break", "does this exist twice" — asked while writing code, while reviewing it, while planning a study,
or while merely answering the user. `/graphify path "A" "B"` traces the connection between two things and
`/graphify explain "N"` describes one of them.

Two reasons it is binding rather than suggested. It is complete where a text search is not: it carries
call edges and definition sites across every study and every module at once, so it answers "what calls
this" correctly the first time, and a refactor planned on an incomplete answer breaks a caller nobody
looked for. And it is cheap: one query returns what reading a dozen files returns, at a fraction of the
context, which is what keeps a long session able to hold the whole problem.

Grep and direct reads stay available and are the right tool once the graph has said **where** to look —
reading the function the graph named, checking the exact text of a line. They are the wrong tool for
finding out what exists.

### The library-first rule

**A study notebook is written out of `shmlib` functions. Code is not typed into a notebook cell when it could be a call, and no code lives outside `shmlib`.** Before any new code is written, in this order:

1. **Understand what already exists.** Query the graph, as the section above requires, and search `studies/shmlib/` for a function that already does it. Most of what a new study needs has been written once already, and a second implementation of it is a defect rather than a shortcut — two implementations of one idea disagree eventually, and the disagreement surfaces as a result nobody can explain.
2. **If none does, adapt one.** Extending a function with a new optional parameter, or generalising it to accept a column name it previously hard-coded, is preferred to writing a second function beside it. **The adaptation must not change what existing callers get.** Every current call site must keep its present behaviour, defaults included, and the burden of showing that is on the change: re-run the tests of every study that calls the function, and re-run the study itself when its result is not covered by a test. Never break a previous result to make a new one convenient.
3. **Only then write a new function — in `shmlib`.** There is no other destination. Put it in the module whose subject it shares, or open a new module when it has no subject in common with any existing one. A function that encodes a choice takes that choice as an argument with a documented default, so that the study which made the choice states it in its parameter cell.

A function written into a study folder instead of `shmlib` is the failure this rule exists to prevent, and so is the same code appearing in two studies. Either way it belongs in `shmlib` and must be moved there, with every call site repointed and every affected study re-run and verified.

### What a study notebook is for

**The notebook orchestrates; the library implements.** A cell should read as a sequence of named calls whose arguments say what is being asked. A cell that is longer than about ten lines, defines a function, performs a statistical test, builds a table row by row, fits a model, or composes a figure axis by axis has stopped orchestrating and must be refactored into a documented function.

**Every important variable is visible in the notebook.** Data pointers — archive directories, input CSV paths, output directories, filenames — and the parameters that govern a result — station, target column, frequency, window lengths, thresholds, date bounds — are declared in the notebook's parameter cell and passed into functions as arguments. They are never buried in a function body, and never left as a default that a reader of the notebook cannot see. A study must be re-aimed at another station, another window or another dataset by editing the parameter cell alone, and a reader must be able to learn what a study was run on without opening its library.

The two halves of this are one rule: the library holds the *how*, the notebook holds the *what* and the *where*. A default value inside a library function is acceptable only when it is a documented constant of the domain rather than a choice — `shmlib.adc.DOCUMENTED_COEFF` is the manufacturer's calibration and belongs in the library; a threshold that a study chose belongs in the notebook that chose it, passed in explicitly even when its value happens to match the library's default.

Each parameter is documented where it is declared, in the **Parameter Tuning Guidance** subsection of the Markdown cell above it: its name, purpose, accepted values, default, and effect on downstream steps.

---

### Graphical Guidelines
- **Categorical Data:** Use the **Okabe-Ito** palette to discriminate categories. In Seaborn, `palette='colorblind'` is an acceptable approximation, but explicit Okabe-Ito hex codes are preferred when manual colors are assigned.
- **Scalar Data:** Use the **Cividis** colormap. Orient it so that the meaningful end is dark: for a coverage or density map use `cmap='cividis_r'`, so that present data reads as ink and absent data as empty page.
- **Variable identity.** Where a study plots several measured channels, colour identifies the channel and nothing else, held constant across every figure in the study: inclination Blue `#0072B2`, air temperature Orange `#E69F00`, relative humidity Bluish Green `#009E73`, supply voltage Black `#000000`, wall temperature Goldenrod `#DAA520`, solar radiation Reddish Purple `#CC79A7`. When two instrument eras or other groups must be told apart inside one axes, use line style — colour is already spent on identity and cannot also carry the grouping.
- **Data lines are single-colour strokes.** Do not put gray or black outlines, halos or `path_effects.Stroke` effects around data lines. If a channel needs more contrast against the plotting ground, use its approved darker identity colour and/or increase the line width; never add a boundary in a second colour.
- **Span highlights are black at 5% opacity.** Every highlighted interval or reference band drawn with `axvspan` or `axhspan` uses Black `#000000`, `alpha=0.05` and no edge. This rule does not apply to quantitative areas such as confidence intervals drawn with `fill_between`.
- **Annotations and event markers** carry the accent colour, Okabe-Ito Vermilion `#D55E00`: instrument-changeover lines, isolated event markers, and the short text notes that explain a clipped axis. The accent is reserved for these roles — it never encodes a data category or a span fill.
- **A legend on a line chart goes below the axes, never inside them.** Binding, with no case-by-case exception: a legend placed inside covers data, and `loc='best'` decides where by an algorithm no reader can predict, so the same figure re-run on new data hides a different part of the record. Draw it centred beneath the axes, horizontal, unframed — one row where the entries fit:

  ```python
  ax.legend(fontsize='small', ncol=<number of entries>, loc='upper center',
            bbox_to_anchor=(0.5, -0.30), frameon=False)
  ```

  `bbox_to_anchor`'s vertical offset is tuned per figure to clear the x-axis label; `-0.30` suits a single full-width axes and a shorter figure needs more. For a multi-panel figure the legend belongs to the figure rather than to one panel — `fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=…, frameon=False)` — and its handles are built explicitly when the panels differ in colour, so that the legend does not assert one panel's identity colour for all of them. `de_lib.phase_chain` (`DE_F24`) is the reference implementation. This rule governs legends only: axis-corner *annotations*, such as the note explaining a clipped axis, stay where they are.
- Do **not** alter black structural elements (e.g., axes, tick labels, titles) or gray background helper elements (e.g., gridlines, individual background traces, historical average lines).
