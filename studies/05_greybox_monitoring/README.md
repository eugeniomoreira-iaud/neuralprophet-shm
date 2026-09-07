# Study 5 · Grey-box expectation and monitoring of the station 02 inclination

Status: **in progress — Phase 5 complete (record, clocks, regressor sets, harmonic diagnostics, Model A attribution on three sets, current-era ladder, rolling expectation with rolling conformal bounds, Model B impulse response, the three-scale monitor with attribution and detectability by mechanism on every chart).** Built to the design in
`docs/superpowers/specs/2026-09-05-study05-greybox-monitoring-design.md`.

This study answers the question Study 04 posed and did not fully answer, for the same
inclinometer at station 02, over the whole eight-year record rather than one post-outage window.
It asks five questions:

1. **What is the record made of, over eight years?** Trend, annual cycle, daily cycle, and the
   response to each measured driver, with the share of variance each carries and the gain each
   learns, on three regressor sets.
2. **Does the wall answer its drivers with a delay or an inertia the 20-minute grid can resolve?**
   Learned impulse responses on the on-structure set, confronted with Study 03's delay-and-time-
   constant scan.
3. **Is this reading the expected one?** A rolling expectation with a conformal interval whose
   coverage is measured as a function of refit staleness.
4. **What departure does each chart catch, at a stated false-alarm budget?** Three charts, three
   damage mechanisms, injections sized by the wall's measured daily response.
5. **What happened across each outage?** The expected level at resumption, from proxies that kept
   recording, against the level observed.

Forecasting stays out of scope: Study 04 already answered it, finding that skill comes from the
response's own memory and that the environment adds nothing distinguishable from zero beyond six
hours.

## Input

Three sources, each read at its own native cadence and joined onto the study's working grid:

- `../../data/interim/archive/gubbio_archive_20min.csv` — Study 1's verdict-aware archive product,
  the on-structure sensor package, read at its native 20-minute cadence. The study reads no raw
  `.adc` file.
- `../../data/raw/proxies/meteosystem_gubbio.csv` — the town ground station, Study 02's better
  proxy for air temperature and relative humidity.
- `../../data/raw/proxies/oikolab_weather.csv` — ERA5 reanalysis, Study 02's transferable arm.

One model specification runs through three regressor sets — the on-structure set, the station set
and the ERA5 set — so the paper narrates two of them and tables all three:

| Role | On-structure set (`str`) | Station set (`gs`) | ERA5 set (`era5`) |
|---|---|---|---|
| Air temperature | `tair` joined across eras | station `Temp` | ERA5 `temperature` |
| Relative humidity | `rh` joined across eras | station `Umid` | ERA5 `relative_humidity` |
| Solar radiation | station `Rad.Sol.`, labelled as borrowed | station `Rad.Sol.` | ERA5 `surface_solar_radiation` |

The on-structure set borrows the station's radiation because the wall's own pyranometer does not
exist before 2025-02-21 and Study 01 condemned 161 of its days after that; the borrowing is stated
in every table that reports the `str` set.

## Outputs

| Artefact | What it holds |
|---|---|
| `GM_01_window_coverage` | Coverage per role per set over the whole window |
| `GM_02_gap_inventory` | Every target gap over eight years, classified by duration |
| `GM_03_clock_check` | Clock offset per source |
| `GM_04_harmonic_diagnostics` | Certified periods per series, annual fits of daily amplitude and phase with held-out order, singular values of the daily-by-annual surface |
| `GM_05_component_shares` | Variance share and peak-to-peak per component, per set |
| `GM_06_learned_gains` | Learned gain per driver beside Study 03's, per set |
| `GM_07_component_stability` | Gain, yearly amplitude, trend slope per fold, per set |
| `GM_08_residual_diagnostics` | Ljung–Box, autocorrelation, scale, period scan, per set |
| `GM_09_nowcast_metrics` | Expectation and interval metrics, per set and per days since refit |
| `GM_10_impulse_response` | Effective delay and time constant per driver beside Study 03's operator |
| `GM_11_alarm_episodes` | Episodes per chart with attribution |
| `GM_12_run_lengths` | Achieved run length per chart with episode count |
| `GM_13_detectability` | Smallest detected departure by mechanism, chart and persistence |
| `GM_14_outage_bridges` | Expected, observed, shift, interval and verdict per outage and proxy set |
| `GM_15_run_metadata` | Every parameter, seed and library version |
| `GM_16_current_era_ladder` | What `twall` and the pyranometer buy, rung by rung |

Figures `GM_F01` through `GM_F14`, each written as both `.png` and `.svg`:

| Figure | NeuralProphet counterpart | Content |
|---|---|---|
| `GM_F01` | — | The record and the three regressor sets on one clock, gaps as gaps |
| `GM_F02` | — | Harmonic diagnostics: the spectral scan of both series, and the daily cycle's amplitude and phase through the year with their annual fits |
| `GM_F03` | fit metrics | Training and validation loss by epoch |
| `GM_F04` | `plot_parameters` trend | Trend on covered time with rate changes marked in the accent colour |
| `GM_F05` | `plot_parameters` seasonality | Yearly curve, and the daily curve at the two solstices and two equinoxes if the smooth conditional term is kept |
| `GM_F06` | `plot_components` | Decomposition stack over the whole record |
| `GM_F07` | `plot_parameters` lagged regressors | Impulse response per driver with Study 03's operator overlaid |
| `GM_F08` | `plot`, `conformal_plot` | Observed against expected with the conformal band, one month |
| `GM_F09`–`GM_F11` | — | The three charts over the monitored record, episodes shaded, attribution marked |
| `GM_F12` | — | Outage bridges: expected level and interval through each outage, observed at resumption |
| `GM_F13` | — | Detectability per mechanism on its own chart |
| `GM_F14` | — | Current-era ladder, rung by rung |

## Reproducing

```bash
conda activate neuralprophet_env
jupytext --to ipynb greybox_monitoring_study.py
jupyter nbconvert --to notebook --execute --inplace \
  greybox_monitoring_study.ipynb --ExecutePreprocessor.timeout=14400
```

Two cautions, both learned the hard way during Study 04 and equally binding here. `nbconvert`
**exits 0 even when a cell raises**, so verify a run by reading the printed lines back out of the
`.ipynb` rather than by trusting the exit code. And if `auto_watcher.py` is running it will
re-sync the `.py` onto the `.ipynb` after `nbconvert` writes it, silently leaving newly added
cells unexecuted; execute to a scratch path with `--output-dir` and copy the result back, or stop
the watcher first.

## Tests

No implementation exists yet, so no tests exist yet. Tests are added phase by phase, alongside the
notebook cells each phase adds: one file under `tests/` per `shmlib` area a phase draws on, on the
model of Study 04's `test_prediction.py`, `test_gaps.py` and `test_decomposition.py`, plus a
folder-honesty test on the model of Study 04's `test_folder_honesty.py`, guarding this folder
against any prose claim without a `GM_` artefact behind it.

## Rebuilding the report

Once the notebook has run, rebuild the report from `report/`:

```bash
pdflatex greybox_monitoring_report.tex
pdflatex greybox_monitoring_report.tex
```

All reusable logic lives in `../shmlib/`. This folder holds the notebook, its parameters, its
outputs, its tests and its report — and no library code.
