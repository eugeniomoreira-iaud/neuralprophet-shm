# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) and other AI coding agents when working with code in this repository.

## Overview
This repository implements a **Physics-Guided Grey-Box Framework for Static Structural Health Monitoring (SHM)**. It provides a *transferable* Python methodology for predicting a structural response (e.g., inclination) from environmental drivers, reconstructing data outages, and detecting anomalies as departures from a frozen healthy-state model. The intended demonstration develops the full procedure on one monitored wall section and then applies the **same procedure, unchanged**, to other sections — so transferability across sections is the primary design goal.

Two points of terminology, deliberately chosen:
- **"Physics-guided," not "physics-informed."** The physics enters through feature construction (plane-of-array radiation, thermal-memory lags) and driver selection, not through governing equations embedded in a loss. The weaker, accurate term is used throughout.
- **No cointegration.** Earlier versions used Engle-Granger cointegration to justify the proxy and define anomalies. This is removed. Covariate selection is now a time-series-aware screening problem, and anomalies are residuals from a frozen predictive model (residual-based novelty detection under environmental normalization).

The `heritageshm` library is not installed as a package — it is imported from the repo root via `sys.path` manipulation in every notebook (see Import Pattern below).

---

## Development Commands

### Environment Setup
All development must be performed within the Conda environment specified in `environment.yml`. The environment name is `neuralprophet_env`.

```bash
# Create from scratch
conda env create -f environment.yml
conda activate neuralprophet_env

# Update an existing environment after changes to environment.yml
conda activate neuralprophet_env
conda env update -f environment.yml --prune
```

Key packages in the environment: `neuralprophet` (via pip), `xgboost`, `scikit-learn`, `statsmodels`, `pvlib` (plane-of-array irradiance / solar geometry), `jupytext`, `watchdog`, and the full scientific Python stack (`numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`).

### Running the Pipeline
The analysis is driven by a sequential Jupyter Notebook pipeline. Execute the following notebooks in order:

1. `00_Sensor_Preprocessing.ipynb` — Raw data extraction and cleaning (site-specific).
2. `01_Data_Quality_and_Gaps.ipynb` — Proxy alignment, joint-availability check, and gap characterization.
3. `02_Proxy_Screening_and_Lags.ipynb` — Plane-of-array radiation, time-series-aware covariate screening, the radiation-value experiment, and thermal-lag feature engineering.
4. `03_Imputation_Benchmark.ipynb` — Benchmarked virtual sensing (XGBoost and alternatives) with train/validation discipline and uncertainty quantification.
5. `04_GreyBox_Decomposition.ipynb` — Frozen-baseline NeuralProphet decomposition, component extraction, and residual diagnostics on a held-out healthy test period.
6. `05_Anomaly_Detection_and_Monitoring.ipynb` — EWMA control chart on the frozen residuals, with the reconstruction–detection firewall applied via provenance flags.
7. `06_Detectability_Assessment.ipynb` — FEM-based damage injection and detectability assessment (ROC, latency, observability).

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
- `02_Proxy_Screening_and_Lags.ipynb` ↔ `02_Proxy_Screening_and_Lags.py`
- `03_Imputation_Benchmark.ipynb` ↔ `03_Imputation_Benchmark.py`
- `04_GreyBox_Decomposition.ipynb` ↔ `04_GreyBox_Decomposition.py`
- `05_Anomaly_Detection_and_Monitoring.ipynb` ↔ `05_Anomaly_Detection_and_Monitoring.py`
- `06_Detectability_Assessment.ipynb` ↔ `06_Detectability_Assessment.py`

### Auto-Watcher (`auto_watcher.py`)
The repo includes `auto_watcher.py`, a file-system watcher that automatically runs `jupytext --sync` whenever a `.py` or `.ipynb` file is saved, keeping pairs in sync without manual intervention. It requires the `watchdog` package.

To start the watcher (in a dedicated terminal, before editing notebooks):
```powershell
# Windows — activate environment first
& "C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1"
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
| `dataloader.py` | Ingestion of `.adc` sensor files, `.csv` proxy files, and FEM response `.csv` files (healthy/damaged) for detectability testing; saving interim data | Pandas DataFrames |
| `preprocessing.py` | Signal cleaning, resampling, and multi-proxy alignment onto the sensor index (resample to source resolution; do not upsample beyond the proxy's native step) | Aligned DataFrame |
| `diagnostics.py` | Gap taxonomy (MCAR/MAR/MNAR) and residual diagnostics (Ljung-Box, ACF, prediction-interval coverage); gap histogram figure and gap statistics table | Figure `.png`, stats `.csv` |
| `solar.py` | Plane-of-array (POA) irradiance from global horizontal irradiance using solar geometry and a section's azimuth/tilt (`pvlib`-based) | POA series per section |
| `screening.py` | Time-series-aware covariate screening: deseasonalized/partial correlation, lagged cross-correlation, VIF, regularized selection; nested-model comparison for the radiation-value experiment | Ranking tables `.csv` |
| `features.py` | Physically motivated thermal-inertia lagged-feature generation | Feature DataFrame |
| `imputation.py` | Wrappers for benchmark gap-filling models (e.g., XGBoost virtual sensing with conformal bootstrap), with provenance flags | Imputed series, uncertainty bounds, provenance labels |
| `decomposition.py` | NeuralProphet grey-box decomposition with fit→**freeze**→apply support: trend, seasonality, AR memory, exogenous regressors | Decomposed DataFrame, frozen model object |
| `monitoring.py` | EWMA (primary) and optional CUSUM control charts for residual-based anomaly detection; consumes provenance flags to enforce the reconstruction–detection firewall | Control chart figure, alarm table, false-alarm rate |
| `detectability.py` | FEM damage-signature injection, detection-threshold/ROC computation, detection latency, and per-section observability mapping | ROC figure, detectability table |
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
1. **Raw Data**: `data/raw/sensor/` (`.adc` files), `data/raw/proxies/` (`.csv` files, e.g. local weather-station export), and `data/raw/fem/` (`.csv` files: FEM healthy and damaged response series).
2. **Interim Data**: Cleaned/aligned datasets in `data/interim/sensor/` and `data/interim/aligned/`.
3. **Processed Data**: Final feature matrices, imputed series (with provenance), and frozen-baseline predictions in `data/processed/`.
4. **Outputs**: Plots in `outputs/figures/`, metrics in `outputs/tables/`, models in `outputs/models/`.

> **Important:** `data/` and `outputs/` are gitignored and do not exist in the cloned repository. They must be created locally and populated with data before running any notebook. A `FileNotFoundError` on these paths is expected behaviour on a fresh clone — it is not a code bug.

### Output Artifact Naming Convention
All saved figures and tables follow a consistent naming pattern:
```
{notebook_id}_{artifact_id}_{station}_{description}.{ext}
```
Examples: `01_01_st02_gap_histogram.png`, `03_02_st02_imputation_metrics.csv`, `06_01_st02_detectability_roc.png`.

When generating new output cells, follow this convention. The station identifier is controlled by the `TARGET_STATION` parameter at the top of each notebook.

### Primary User Parameters
All notebooks are parameterized around a small set of clearly marked user inputs near the top of each notebook. Never hardcode these values anywhere else.

- `TARGET_STATION` — e.g. `'st02'`. Controls which sensor file is loaded and how outputs are named. **This is the transferability switch:** the develop-once / apply-everywhere workflow is realised by re-running the unchanged 01–06 pipeline with a different `TARGET_STATION`.
- `TARGET_COL` — the structural response column (e.g. `'absinc'`).
- `SECTION_AZIMUTH`, `SECTION_TILT` — orientation of the wall section, used by `solar.py` to compute the per-section plane-of-array irradiance. These differ per station and are the main reason a single shared weather station can serve differently oriented sections.
- `PROXY_COLS`, `META_COLS`, `TARGET_FREQ` — proxy column selection, metadata columns to drop, and the alignment frequency.
- `BASELINE_SPLIT` — chronological train / validation / held-out-healthy-test boundaries used to fit, tune, and evaluate the frozen baseline (see Notebook 04); the validation portion also defines the EWMA control limits consumed by Notebook 05.

---

## Methodological Rules (must be enforced in code)

These rules encode the scientific design and must not be violated by notebook or module changes:

1. **No cointegration / equilibrium tests.** Do not reintroduce Engle-Granger, ADF-as-model-validation, or "long-run equilibrium" language. Residual quality is assessed with Ljung-Box / ACF and prediction-interval coverage, not unit-root tests.
2. **Reconstruction–detection firewall.** Implemented in its own dedicated notebook (05). Anomaly detection runs only on `observed` (and small-interpolation) segments. `monitoring.py` must receive provenance flags and exclude `imputed_large_gap` segments from residual/control-chart analysis.
3. **Frozen baseline.** The NeuralProphet model is fit on the training period, **frozen**, and applied forward without refitting on validation/test/operational data. Refitting on the evaluation window would absorb gain-type damage and is prohibited.
4. **Train/validation/test discipline everywhere.** Imputation (Notebook 03) uses chronological train/validation with strict no-leakage artificial-gap masking. The baseline (Notebook 04) uses train (fit+freeze) / validation (tune + set EWMA limits) / held-out-healthy-test (predictive error, residual whiteness, empirical false-alarm rate). The detection stage (Notebook 05) consumes the frozen model and the EWMA control limits set in 04; it must not refit.
5. **Damage injection by difference, not replacement.** In Notebook 06, construct the damaged series as `y_mod = y_measured + (FEM_damaged − FEM_healthy)` so the FEM's healthy-state modelling error cancels and the residual reflects damage, not model mismatch. Never substitute the absolute FEM output for the measurement.
6. **Transferability invariance.** Notebooks 01–06 must contain no site-specific constants. Anything that varies per section is a top-of-notebook parameter (`TARGET_STATION`, `SECTION_AZIMUTH`, `SECTION_TILT`, `BASELINE_SPLIT`).

---

## Coding Conventions

### Notebook Structure Standard (Notebooks 01–06)

Notebooks 01 through 06 form the **transferable core pipeline**. They are designed to work with any static structural sensor dataset and any compatible environmental proxy dataset — not only the Gubbio case. When writing or modifying these notebooks, preserve this generality: never hardcode site-specific values, units, or assumptions.

Every notebook in the 01–06 sequence must follow this internal structure:

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
2. **Proxy Loading** — Load weather-station proxy data and select relevant columns.
3. **Alignment** — Resample and synchronize proxies onto the sensor index.
4. **Joint Availability** — Quantify overlap of proxy coverage with sensor gaps.
5. **Gap Characterization** — Classify missing data and diagnose gap taxonomy.
6. **Save** — Export the aligned dataset to `/data/interim/aligned/`.
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
- Do not attempt to generalise it to match the 01–06 pattern.
- Parameters such as `COMP_COEFF`, `STATIONS`, `SEPARATOR`, and `FILE_EXT` are site-specific and must be updated by the user for a new deployment.
- Its output (`{station}_preprocessed.csv` in `data/interim/sensor/`) is the standard entry point for the transferable pipeline starting at Notebook 01.

---

### Expected Dataset Contracts (Notebooks 01–06)

The transferable pipeline expects the following input datasets. Any new dataset must conform to these contracts before being fed into the pipeline.

**Sensor dataset** (`data/interim/sensor/{station}_preprocessed.csv`)
- Produced by Notebook 00 (or any equivalent preprocessing step).
- A CSV file with a `datetime` column parseable as a `DatetimeIndex`.
- At minimum one structural response column (e.g., `absinc`). Column name is user-configurable via `TARGET_COL`.
- Regular or near-regular time steps (gaps allowed; the pipeline handles them). Typical resolution: hourly or sub-hourly.
- Units must be consistent across the full series (no specific unit required).
- No pre-imputation expected: the file contains NaN where data is missing.

**Proxy dataset** (`data/raw/proxies/{name}.csv`)
- A CSV file with a datetime column (e.g. `datetime` or `datetime (UTC)`), parseable as a `DatetimeIndex`.
- One or more environmental variable columns. Column names and units are user-configurable via `PROXY_COLS`. For the Gubbio deployment these are local weather-station variables (air temperature, relative humidity, **global horizontal solar radiation**, wind, rain, etc.); global horizontal radiation is converted to per-section POA in Notebook 02.
- Must cover the full temporal window of the sensor dataset (checked explicitly in Notebook 01 before alignment), and its own gaps are quantified against the sensor gaps (joint-availability check).
- Resolution: hourly or finer. The pipeline resamples to `TARGET_FREQ` during alignment but does not upsample beyond the proxy's native step.
- Metadata columns (coordinates, model name, elevation, UTC offset) are dropped during loading if listed in `META_COLS`.

**FEM dataset** (`data/raw/fem/{station}_{scenario}.csv`) — required only for Notebook 06
- A CSV with a `datetime` column aligned to the sensor index and one response column matching `TARGET_COL`.
- Provided as matched pairs: a healthy run and one or more damaged runs (e.g. elastic-modulus reduction per component/leaf), all driven by the same real environmental forcing.
- Used only to form the injected damage signature `(FEM_damaged − FEM_healthy)`; absolute FEM values are never substituted for measurements.

---

### Module Coding Standard (`heritageshm/`)

**Module-level docstring**
Every `.py` file in `heritageshm/` must begin with a module-level docstring that states: (1) the module name, (2) its responsibility within the pipeline, and (3) its primary inputs and outputs. Example:

```python
"""
Module: diagnostics.py
Handles gap taxonomy characterization (MCAR/MAR/MNAR) and residual
diagnostics (Ljung-Box, ACF, prediction-interval coverage).
"""
```

**Function-level docstrings (mandatory)**
Every public function must have a NumPy-style or Google-style docstring containing:
- A one-line summary of what the function does.
- `Parameters` section: name, type, description, and default for every argument.
- `Returns` section: type and description of every return value.
- Any side effects (e.g., saves a file, prints to stdout) must be noted.

Example pattern (from `screening.py`):
```python
def lagged_cross_correlation(df, target_col, proxy_col, max_lag, deseasonalize=True):
    """
    Computes the lagged cross-correlation between a candidate proxy and the
    structural target to identify the wall's thermal-memory lag structure.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing both variables on a common time index.
    target_col : str
        The structural sensor column (e.g., 'absinc').
    proxy_col : str
        The candidate environmental proxy column (e.g., 'poa_irradiance').
    max_lag : int
        Maximum lag (in samples) to evaluate in both directions.
    deseasonalize : bool, optional
        If True, remove diurnal/seasonal components before correlating to
        avoid spurious shared-cycle association. Default True.

    Returns
    -------
    lags : np.ndarray
        Array of evaluated lags.
    corr : np.ndarray
        Cross-correlation value at each lag.
    best_lag : int
        Lag maximising absolute cross-correlation.
    """
```

**No logic in notebooks**
If a block of code in a notebook cell is longer than ~10 lines, performs a statistical test, trains a model, or produces a figure, it must be refactored into a named function in the appropriate module. The notebook cell then becomes a single function call with named arguments.

**No hardcoded paths in modules**
Module functions must never hardcode file paths. All paths must be passed as arguments. Modules are path-agnostic; notebooks control all I/O paths.

**No side effects without explicit opt-in**
Functions that save files or produce plots must have an optional parameter (e.g., `save_plot_path=None`) that defaults to no side effect. The notebook controls whether outputs are saved.