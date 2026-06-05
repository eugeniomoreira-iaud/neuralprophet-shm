# Physics-Guided Grey-Box Framework for Static SHM
**A transferable Python methodology for static Structural Health Monitoring (SHM) under incomplete data.**

This repository contains the `heritageshm` Python library and a guided Jupyter Notebook pipeline. It provides a complete workflow for predicting a static structural response (e.g., inclination) from environmental drivers, reconstructing data outages, and performing residual-based anomaly detection using interpretable machine learning.

The framework is designed to be **developed once on a single monitored section and applied unchanged to others**: re-running the pipeline with a different `TARGET_STATION` (and that section's orientation) reproduces the entire procedure on a new wall section. Transferability across sections is the primary goal.

*Note: Documented examples and validation datasets (e.g., the medieval urban walls of Gubbio) will be added to an `/examples` directory in future updates.*

---

## 📖 Methodology Overview
The project moves from raw sensor cleaning to operational monitoring and, finally, to a quantitative detectability assessment, all under a grey-box paradigm. The structural response is predicted from measurable environmental drivers; an anomaly is a sustained, unexplained departure of the measurement from the frozen healthy-state prediction.

**Phase A — Historical Model Building**
1. **Data Preprocessing & Gap Diagnosis:** Synchronization of on-site structural sensors with a nearby weather-station proxy. Includes a joint-availability check (does the proxy cover the periods where the sensor is missing?) and gap taxonomy classification (MCAR, MAR, MNAR).
2. **Physics-Guided Covariate Screening:** Conversion of global horizontal solar radiation to **per-section plane-of-array irradiance**, then *time-series-aware* covariate screening — deseasonalized/partial correlation, lagged cross-correlation for thermal-memory lags, VIF, and regularized selection — followed by thermal-lag feature generation. A nested-model comparison ({T, RH} vs {T, RH, radiation}) quantifies the **marginal predictive value of solar radiation**, providing an evidence base for whether to install dedicated on-site radiation sensors.
3. **Compact Imputation Benchmark:** Benchmarked virtual sensing (XGBoost and alternatives) with conformal-bootstrap calibration to reconstruct contiguous outages, evaluated on artificial gaps under strict chronological train/validation discipline. Every reconstructed value is provenance-flagged.

**Phase B — Operational Monitoring**
4. **Frozen Grey-Box Decomposition:** Separation of structural trend, environmental seasonality, and structural residuals using `NeuralProphet` (autoregressive memory + exogenous environmental regressors). The model is fit on a healthy training period, **frozen**, and applied forward; residual quality is verified on a held-out healthy test period (Ljung-Box / ACF, prediction-interval coverage).
5. **Residual-Based Anomaly Detection (EWMA):** The frozen residuals from Step 4 are monitored with an **EWMA control chart** (CUSUM optional), with control limits set on the validation portion of the healthy baseline. A **reconstruction–detection firewall** — enforced via the imputation provenance flags — ensures detection runs only on `observed` and small-interpolation segments, so imputed gaps never generate false anomalies. The empirical false-alarm rate is measured on held-out healthy data.

**Phase C — Detectability Assessment**
6. **FEM Damage Injection:** No real damage event has occurred, so detection capability is established by injecting FEM-simulated damage signatures into past windows — `y_mod = y_measured + (FEM_damaged − FEM_healthy)` — and overlaying the measured, predicted, and damage-modified signals. Detection-threshold/ROC curves over damage magnitude and false-alarm rate, detection latency, and a per-section observability map quantify what the system can and cannot detect, demonstrating readiness for a future event.

---

## 📂 Project Structure

```text
heritageshm/                        # Repository root
│
├── heritageshm/                    # Core Python library
│   ├── dataloader.py               # I/O (.adc sensors, .csv proxies, FEM csv)
│   ├── preprocessing.py            # Alignment and resampling
│   ├── diagnostics.py              # Gap taxonomy (MCAR/MAR/MNAR) + residual diagnostics
│   ├── solar.py                    # Plane-of-array irradiance (pvlib, per-section)
│   ├── screening.py                # Time-series-aware covariate screening + radiation-value test
│   ├── features.py                 # Thermal-inertia lagged-feature generation
│   ├── imputation.py               # Benchmark imputation wrappers + provenance flags
│   ├── decomposition.py            # NeuralProphet fit→freeze→apply wrappers
│   ├── monitoring.py               # EWMA/CUSUM control charts + firewall enforcement
│   ├── detectability.py            # FEM injection, ROC, latency, observability
│   └── viz.py                      # Seaborn/matplotlib visualization utilities
│
├── data/                           # Datasets — tracked locally, ignored by Git
│   ├── raw/
│   │   ├── sensor/                 # Raw sensor files (e.g., .adc)
│   │   ├── proxies/                # Weather-station proxy files (.csv)
│   │   └── fem/                    # FEM healthy & damaged response series (.csv)
│   ├── interim/
│   │   ├── sensor/                 # Cleaned and standardized sensor DataFrames
│   │   └── aligned/                # Synchronized sensor + proxy datasets
│   └── processed/                  # Imputed (provenance-flagged) and decomposed series
│
├── outputs/                        # Generated artifacts — tracked locally, ignored by Git
│   ├── figures/                    # High-resolution output plots
│   ├── tables/                     # Exported CSV metrics and tables
│   └── models/                     # Saved frozen-baseline model weights
│
├── auxiliary/                      # Supplementary datasets and secondary notebooks
│
├── 00_Sensor_Preprocessing.ipynb   # Notebook 00 — raw data extraction and cleaning
├── 00_Sensor_Preprocessing.py      # Jupytext paired source (edit this, not the .ipynb)
├── 01_Data_Quality_and_Gaps.ipynb
├── 01_Data_Quality_and_Gaps.py
├── 02_Proxy_Screening_and_Lags.ipynb
├── 02_Proxy_Screening_and_Lags.py
├── 03_Imputation_Benchmark.ipynb
├── 03_Imputation_Benchmark.py
├── 04_GreyBox_Decomposition.ipynb
├── 04_GreyBox_Decomposition.py
├── 05_Anomaly_Detection_and_Monitoring.ipynb
├── 05_Anomaly_Detection_and_Monitoring.py
├── 06_Detectability_Assessment.ipynb
├── 06_Detectability_Assessment.py
│
├── auto_watcher.py                 # Jupytext file watcher (auto-syncs .py → .ipynb)
├── auto_watcher instructions.md    # Instructions for using the auto-watcher
├── jupytext.toml                   # Jupytext pairing configuration
├── environment.yml                 # Conda environment specification
├── file_structure.md               # Extended file structure reference
├── CLAUDE.md                       # Instructions for AI coding agents
└── LICENSE
```

> **Note on Jupytext pairing:** Every `.ipynb` notebook has a paired `.py` file (in `py:percent` format). **Always edit the `.py` file**, never the `.ipynb` directly. Use `auto_watcher.py` or the Jupytext CLI to sync changes back to the notebook. See `auto_watcher instructions.md` for details.

---

## 🚀 Installation & Execution

### 1. Create the Conda Environment

The environment is fully specified in `environment.yml`. To create it from scratch:

```bash
conda env create -f environment.yml
conda activate neuralprophet_env
```

This installs all required dependencies, including `neuralprophet`, `xgboost`, `pvlib`, `jupytext`, `statsmodels`, and the full scientific Python stack.

### 2. Update an Existing Environment

If the `environment.yml` has changed (e.g., new packages were added) and you want to sync your local environment without recreating it:

```bash
conda activate neuralprophet_env
conda env update -f environment.yml --prune
```

The `--prune` flag removes any packages that are no longer listed in the file, keeping the environment consistent with the specification.

To fully recreate the environment from scratch (e.g., after a major update or to resolve conflicts):

```bash
conda deactivate
conda env remove -n neuralprophet_env
conda env create -f environment.yml
conda activate neuralprophet_env
```

### 3. Export Your Current Environment

To save the exact state of your working environment (including resolved dependency versions) for reproducibility:

```bash
# Export full specification with exact versions (for exact reproduction)
conda env export -n neuralprophet_env > environment_frozen.yml

# Export only explicitly installed packages (portable across platforms)
conda env export -n neuralprophet_env --from-history > environment.yml
```

Commit `environment.yml` to Git to keep the specification up to date.

### 4. Supplying the Data

Due to size and privacy limits, the `data/` and `outputs/` directories are tracked locally but ignored by Git. To run this pipeline:

1. Place your target static sensor data into `data/raw/sensor/`.
2. Place your environmental proxy data (weather-station export) into `data/raw/proxies/` as CSV files.
3. For the detectability stage, place matched FEM healthy/damaged response series into `data/raw/fem/`.

### 5. Running the Pipeline

The methodology is executed sequentially via the Jupyter Notebooks. Launch JupyterLab:

```bash
jupyter lab
```

Execute the notebooks in order from `00` to `06`. To reproduce the procedure on another wall section, change `TARGET_STATION` (and that section's `SECTION_AZIMUTH` / `SECTION_TILT`) and re-run `01`–`06` unchanged.

| Notebook | Description |
|---|---|
| `00_Sensor_Preprocessing` | Raw data extraction, cleaning, and standardization (site-specific) |
| `01_Data_Quality_and_Gaps` | Proxy alignment, joint-availability check, gap characterization (MCAR/MAR/MNAR) |
| `02_Proxy_Screening_and_Lags` | Plane-of-array radiation, time-series-aware screening, radiation-value experiment, thermal-lag features |
| `03_Imputation_Benchmark` | Benchmarked virtual sensing with train/validation discipline and uncertainty quantification |
| `04_GreyBox_Decomposition` | Frozen-baseline NeuralProphet decomposition, component extraction, residual diagnostics on a held-out healthy test period |
| `05_Anomaly_Detection_and_Monitoring` | EWMA control chart on the frozen residuals, with the reconstruction–detection firewall enforced via provenance flags |
| `06_Detectability_Assessment` | FEM damage injection, detectability ROC, latency, observability |
```