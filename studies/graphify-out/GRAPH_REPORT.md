# Graph Report - studies  (2026-09-06)

## Corpus Check
- 41 files · ~122,615 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1046 nodes · 1524 edges · 115 communities (62 shown, 53 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `4286958e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- _diurnal
- calibrate_against_sensor
- _quiet
- tables.py
- apply_report_style
- couple
- figures.py
- de_lib.py
- format_spines
- Inclinometer conditioner output
- write_adc
- clock_check
- TestPlotDiurnalSeasonGrid
- _parse
- plot_coupling_scatter
- WallTemperatureFilterTests
- TestDrawCycle
- finish
- TestMeteo
- TestTables
- TestViz
- TestSensorLoading
- season_of
- to_utc
- test_prediction.py
- phase_chain
- solar_elevation
- gubbio_archive_20min.csv
- TestAgreement
- TestCompensation
- TestSolar
- Study 2 — Proxy Forcing Characterization (README)
- TestCertifiedWindow
- TestClockConversion
- TestShortlist
- TestResponseLoading
- TestSite
- ReportSpanHighlightTests
- TestVerdictTable
- TestMaskImplausible
- prediction.py
- plot_source_panels
- TestSeasons
- test_gaps.py
- monitoring.py
- anomaly_by_channel
- clock_cell
- solar_night_mask
- coverage_by_period
- export_columns
- filter_impulsive_segment
- sentinel_census
- test_decomposition.py
- Data exploration report review notes
- test_neuralprophet_capabilities.py
- TestScorePredictionsExtension
- _as_series
- Study 4 · Decomposition, expectation and anomaly judgement
- block_view
- build_grid
- clock_step
- Q: What connects Compensated inclination, Solar radiation, Wall temperature to the rest of the system?
- Q: Implement Study 04 using NeuralProphet to predict inclination from on-structure variables
- condemned_modes
- excursions_by_month
- add_inclination
- current_era_availability
- flag_and_correct
- last_archive_day
- parse_era
- station_classification
- summary_by_era
- response_ratios
- sr_valid_days
- spike_replacement_summary
- recorded_thermal_slope
- sync_archive
- TestModelFigures
- scan_raw_text
- circular_std
- wind_components
- driver_coverage
- lag_ranges
- r2_ceiling
- test_folder_honesty.py
- certified_sr_window
- sr_day_census
- verdict_table
- TestPhysicalInjections
- TestSingletonSegmentDroppedBeforeFitting
- set_context
- Study 3 README
- TestComponentVarianceShares
- TestPeriodScan
- plot_decomposition_stack
- neuralprophet_backtest
- Study 2 must not re-decide Study 1's rejections
- _long_predictions
- TestSingletonSegmentDroppedBeforePrediction
- Study 5 · Grey-box expectation and monitoring of the station 02 inclination
- TestCouple
- TestResidualDiagnostics
- TestWindComponents
- baseline_predictions
- decomposition_columns
- contiguous_segments
- _model_frame
- gap_inventory
- period_scan
- residual_diagnostics
- channel_inventory
- harmonise
- join_eras
- mask_implausible
- source_inventory

## God Nodes (most connected - your core abstractions)
1. `finish()` - 39 edges
2. `format_spines()` - 37 edges
3. `_hourly()` - 24 edges
4. `_diurnal()` - 24 edges
5. `figsize()` - 21 edges
6. `_quiet()` - 20 edges
7. `apply_report_style()` - 15 edges
8. `TestTables` - 13 edges
9. `TestMeteo` - 12 edges
10. `Study 2 — Proxy Forcing Characterization (README)` - 11 edges

## Surprising Connections (you probably didn't know these)
- `solar_night_mask()` --calls--> `solar_elevation()`  [EXTRACTED]
  01_data_exploration/de_lib.py → shmlib/solar.py
- `diurnal_cycle_grid()` --calls--> `apply_report_style()`  [EXTRACTED]
  01_data_exploration/de_lib.py → shmlib/viz.py
- `diurnal_cycle_grid()` --calls--> `format_spines()`  [EXTRACTED]
  01_data_exploration/de_lib.py → shmlib/viz.py
- `diurnal_cycle_pair()` --calls--> `apply_report_style()`  [EXTRACTED]
  01_data_exploration/de_lib.py → shmlib/viz.py
- `diurnal_cycle_pair()` --calls--> `finish()`  [EXTRACTED]
  01_data_exploration/de_lib.py → shmlib/viz.py

## Import Cycles
- 3-file cycle: `shmlib/__init__.py -> shmlib/quality.py -> shmlib/coupling.py -> shmlib/__init__.py`
- 4-file cycle: `shmlib/__init__.py -> shmlib/proxies.py -> shmlib/quality.py -> shmlib/coupling.py -> shmlib/__init__.py`
- 5-file cycle: `shmlib/__init__.py -> shmlib/viz.py -> shmlib/proxies.py -> shmlib/quality.py -> shmlib/coupling.py -> shmlib/__init__.py`

## Hyperedges (group relationships)
- **Study 2's three input data sources** — 02_proxy_forcing_characterization_readme, data_raw_proxies_oikolab_weather, data_raw_proxies_meteosystem_gubbio [EXTRACTED 0.90]
- **Study 2's analysis pipeline modules (proxies, quality, compare, figures)** — shmlib_proxies, shmlib_quality, shmlib_compare, shmlib_figures, 02_proxy_forcing_characterization_readme [EXTRACTED 0.95]
- **Monitoring chain from solar forcing to wall response** — 01_data_exploration_report_data_exploration_report_solar_radiation, 01_data_exploration_report_data_exploration_report_wall_temperature, 01_data_exploration_report_data_exploration_report_inclinometer [INFERRED 0.85]

## Communities (115 total, 53 thin omitted)

### Community 0 - "_diurnal"
Cohesion: 0.07
Nodes (22): _diurnal(), _hourly(), The second parameter: inertia, which a pure delay cannot express., The grid, and what it says that a single lag curve cannot., A negative coupling is invisible to a search that maximises ``r``., The filter must remove the drift and leave the daily cycle alone., The slope, and the standard error an hourly series actually deserves., The screen, run over the grid rather than over delays alone. (+14 more)

### Community 1 - "calibrate_against_sensor"
Cohesion: 0.07
Nodes (32): _agreement_scores(), agreement_stability(), calibrate_against_sensor(), daily_amplitude_phase(), diurnal_profile(), diurnal_table(), _paired(), pairwise_agreement() (+24 more)

### Community 2 - "_quiet"
Cohesion: 0.06
Nodes (13): _quiet(), Unit tests for shmlib.monitoring. Run from studies/: python…, TestAlarmEpisodes, TestAverageRunLength, TestCusumChart, TestDetectabilityCurve, TestDetectabilityKinds, TestEwmaChart (+5 more)

### Community 3 - "tables.py"
Cohesion: 0.10
Nodes (21): basename(), _cell(), date_cell(), latex_escape(), percent(), Module: shmlib.tables Writing the LaTeX table bodies that the study reports…, Format a table into LaTeX rows, one string per row of ``frame``. Parameters…, Write LaTeX rows to a table body file. The rows are joined by the row separator… (+13 more)

### Community 4 - "apply_report_style"
Cohesion: 0.16
Nodes (20): plot_channel_series(), plot_current_era_series(), plot_segment(), plot_sr_failure(), plot_sr_raw(), plot_twall_raw(), One full-archive plot of a single channel, in that channel's colour. Parameters…, One plot of a current-era channel over the era it exists in. Unlike… (+12 more)

### Community 5 - "couple"
Cohesion: 0.12
Nodes (20): best_operator(), couple(), diurnal_band(), gain_at_lag(), gain_stability(), operator_scan(), _paired_correlation(), A driver put through the two things that can delay a response, in order. They… (+12 more)

### Community 6 - "figures.py"
Cohesion: 0.16
Nodes (15): # TODO: write one row per quantity per source once the numbers above have been…, Unit tests for the decisions study 2 relies on, now that they live in…, Unit tests for the decisions study 3 relies on, which live in ``shmlib``. Run…, Module: shmlib.adc The raw ``.adc`` archive: its file format, the constants of…, Module: shmlib.compare How a source behaves on its own, and how two sources…, Module: shmlib.coupling How one series moves with another, and after how long.…, Module: shmlib.figures The multi-source figures a proxy-forcing comparison…, Package: shmlib The library shared by the studies under ``studies/``, and the… (+7 more)

### Community 7 - "de_lib.py"
Cohesion: 0.09
Nodes (21): assemble_wide(), coalesce_target(), coincidence(), condemned_by_month(), diurnal_response(), export_column_doc(), extreme_excursions(), Module: de_lib.py Support library for the data-exploration study… (+13 more)

### Community 8 - "format_spines"
Cohesion: 0.10
Nodes (28): plot_agreement_stability(), plot_cadence_evidence(), plot_certified_window(), plot_detectability(), plot_diurnal_comparison(), plot_gap_anatomy(), plot_metric_vs_horizon(), plot_pair_bias_grid() (+20 more)

### Community 9 - "Inclinometer conditioner output"
Cohesion: 0.12
Nodes (17): The monitoring system of the Mura Urbiche di Gubbio and its record, gubbio_archive_20min.csv exported dataset, Field-wise decimal-separator normalization, Twenty-four-hour Hampel impulsive-noise filter, Inclinometer conditioner output, Legacy-to-current hardware changeover, Mura Urbiche of Gubbio, Radiation day-quality verdict (+9 more)

### Community 10 - "write_adc"
Cohesion: 0.15
Nodes (8): The separator is mixed, sometimes within a single record., `CURRENT_RECORD` writes batt with a comma and tair with a point., Write lines to a temporary .adc file and return its path., The two era parsers, and the boundary between them., A file spanning the changeover carries both layouts. Each parser must take its…, TestDecimalSeparator, TestParsers, write_adc()

### Community 11 - "clock_check"
Cohesion: 0.17
Nodes (12): clock_offset(), The logger's clock offset from UTC, measured against computed solar noon. The…, best_lag(), lag_scan(), Correlation between a response and a driver at every lag in a range. The whole…, The winning lag of a scan, under the stated objective. Parameters ----------…, clock_check(), Two independent tests of the clock each source keeps, run on one quantity. A… (+4 more)

### Community 12 - "TestPlotDiurnalSeasonGrid"
Cohesion: 0.10
Nodes (13): `figures.plot_diurnal_season_grid`, the channel-by-season diurnal grid., Five hourly days each of summer and winter, on two channels of very different…, A temperature channel and a wind-direction channel, five hourly days each…, The returned figure carries exactly `len(columns) * len(seasons)` axes, laid…, The two seasonal panels of one row must return the same `get_ylim()`, since a…, The season name appears once, capitalised, at the start of the top-row panel…, The row of a channel declared circular is drawn on a 0-360 degree compass axis…, `figures.plot_channel_panels`, the channel-named resolver over… (+5 more)

### Community 13 - "_parse"
Cohesion: 0.25
Nodes (8): _parse(), parse_file(), parse_legacy_file(), Parse one numeric field, accepting either decimal separator. The archive mixes…, Parse the records of one era out of one ``.adc`` file. Shared body of…, Parse the current-era records of one ``.adc`` file. Handles the mixed decimal…, Parse the legacy-era records of one ``.adc`` file. Differs from…, to_float()

### Community 14 - "plot_coupling_scatter"
Cohesion: 0.18
Nodes (14): plot_coupling_scatter(), plot_gain_stability(), plot_lag_curves(), plot_operator_grid(), The correlation of each driver against the response, at every lag scanned. The…, The response against one driver, at the lag the scan chose. Drawn as a hexbin…, Each driver's gain re-estimated window by window, one panel per driver. One…, The whole delay-by-time-constant grid, one panel per driver. The figure that a… (+6 more)

### Community 15 - "WallTemperatureFilterTests"
Cohesion: 0.15
Nodes (7): Tests for auditable wall-temperature impulse filtering., A spike inside the segment is interpolated; earlier data is untouched., Filtering an observed spike must not fill a pre-existing missing slot., A run boundary cannot borrow context from the opposite side of a gap., Archive exports must include filtered values and their provenance flag., The analysis view must preserve corrected and filtered wall temperatures., WallTemperatureFilterTests

### Community 16 - "TestDrawCycle"
Cohesion: 0.15
Nodes (7): `viz.draw_cycle`, the single diurnal panel every study draws with., With `complete_day` set, a day short of that many slots contributes neither to…, `centre=True` returns a cycle whose own mean is zero and `centre=False` returns…, A position within the day is left `NaN` when fewer than `min_days` distinct…, `circular=True` averages by the unit-vector method, matching…, `background=True` draws every individual day behind the mean and…, TestDrawCycle

### Community 17 - "finish"
Cohesion: 0.17
Nodes (12): diurnal_cycle_grid(), plot_anomaly_channels(), plot_anomaly_comparison(), plot_coverage_heatmap(), plot_inclination_series(), Average diurnal cycle of one channel, by instrument era and season. A grid with…, Daily coverage of several channels or stations, as a heatmap. One strip per…, The inclination over the whole archive, on a clipped vertical axis. A handful… (+4 more)

### Community 19 - "TestTables"
Cohesion: 0.13
Nodes (4): The LaTeX row convention, and what a missing value prints as., A trailing separator opens an empty row that \\bottomrule lands inside., Unit strings are markup, not data, and must reach LaTeX intact., TestTables

### Community 20 - "TestViz"
Cohesion: 0.18
Nodes (3): The figure conventions that the Graphical Guidelines make binding., Study 1's `_complete_days` and `_draw_cycle` moved here as `complete_days` and…, TestViz

### Community 21 - "TestSensorLoading"
Cohesion: 0.20
Nodes (4): Write a frame to a temporary CSV and return its path., Study 1's verdicts are consumed, never revisited., TestSensorLoading, _write_csv()

### Community 22 - "season_of"
Cohesion: 0.16
Nodes (14): The slices a screen is run over, as boolean masks on one index. Three kinds,…, strata(), plot_diurnal_grid(), plot_diurnal_season_grid(), plot_diurnal_source_grid(), Several sources' mean daily cycles overlaid, a quantity per row and a season…, The mean daily cycle of every channel of one source, as a grid of panels. One…, One source's mean daily cycles, a channel per row and a season per column. The… (+6 more)

### Community 23 - "to_utc"
Cohesion: 0.25
Nodes (8): load_response(), load_sensor_forcings(), Load the on-structure forcings from study 1's archive onto the analysis grid.…, Load the structural response from study 1's archive onto the analysis grid. The…, load_sr_record(), The radiation channel with its verdicts beside it, before the verdicts bite.…, Convert a naive index kept on the site's civil clock to naive UTC. Study 1…, to_utc()

### Community 24 - "test_prediction.py"
Cohesion: 0.04
Nodes (27): Tests for Study 04 prediction helpers. Run directly, with no test runner…, Routes must reflect what information exists for each prediction task., Prediction scores use paired observed and predicted values only., Bootstrap skill must keep parent and child errors timestamp-aligned., Robust quantile column matching across varied library string formats., A tiny real fit protects the wrapper contract at the library boundary., First differences must not bridge records the model may not learn across., Notebook orchestration must keep fold boundaries explicit. (+19 more)

### Community 25 - "phase_chain"
Cohesion: 0.20
Nodes (10): anomaly_and_reference(), cycle_extreme(), diurnal_cycle_pair(), phase_chain(), Position of a cycle's extreme, or nothing when the extreme is a plateau. A mean…, The anomaly window, and the same calendar window averaged over earlier years.…, Average diurnal cycle of a current-era channel, by season. The current-era…, The four quantities of the thermal chain, on one clock and one scale. The sun… (+2 more)

### Community 26 - "solar_elevation"
Cohesion: 0.18
Nodes (12): flag_sr_day_quality(), Name the single state each day of the failure window belongs to. The window is…, The radiation that survived rejection, before the night correction. Both…, Condemn the days on which the radiation channel has no diurnal cycle. A working…, Solar radiation grouped by solar elevation, condemned days against the rest.…, sr_by_elevation(), sr_day_states(), sr_measurements() (+4 more)

### Community 27 - "gubbio_archive_20min.csv"
Cohesion: 0.25
Nodes (8): Study 1 data exploration README, Raw ADC archive ingest, Mura Urbiche di Gubbio monitoring record, Inclination cleaning chain, Night radiation correction, Radiation suspect-day verdict, Recorded flag and ok-column schema, gubbio_archive_20min.csv

### Community 29 - "TestCompensation"
Cohesion: 0.25
Nodes (3): The documented formula, and the anchoring it depends on., The coefficient is 0.005, applied times 1000: 5 mdeg per °C., TestCompensation

### Community 31 - "Study 2 — Proxy Forcing Characterization (README)"
Cohesion: 0.29
Nodes (7): Study 2 — Proxy Forcing Characterization (README), Usable radiation window (n_sr_ok night-zero correction intersected with sr_suspect's 161-day exclusion), proxy_forcing_report.tex, auxiliary/meteosystem_italy.py, auxiliary/oiko.py, meteosystem_gubbio.csv (Gubbio town ground station), oikolab_weather.csv (ERA5 reanalysis via Oikolab)

### Community 37 - "ReportSpanHighlightTests"
Cohesion: 0.33
Nodes (4): Regression tests for span highlights in the generated report figures., The report uses one visual treatment for every highlighted span., Every generated span is black at 5% opacity, with none omitted., ReportSpanHighlightTests

### Community 39 - "TestMaskImplausible"
Cohesion: 0.07
Nodes (10): Restriction is by calendar day, not by timestamp range., Wind direction is a direction, and the study must never average it flat., The inventory counts defects; it never removes them., The counterpart to the inventory's count: what it counts, this removes., TestCircularHandling, TestInventories, TestMaskImplausible, TestRestrictToDays (+2 more)

### Community 40 - "prediction.py"
Cohesion: 0.13
Nodes (15): availability_route(), conformal_interval(), execution_folds(), expanding_segment_folds(), paired_mae_skill(), Prediction helpers for Study 04. The functions in this module prepare and score…, Choose expanding refits or one frozen-origin evaluation fold. Parameters…, Chronological expanding folds over whole contiguous segments. Each fold trains… (+7 more)

### Community 41 - "plot_source_panels"
Cohesion: 0.20
Nodes (10): _clip_note(), plot_channel_panels(), plot_source_panels(), The complete record of one source, one panel per channel, on one shared clock.…, The complete record of the on-structure package, one panel per measured…, The note a clipped axis carries, naming what it leaves outside. Parameters…, channel_name(), Axis limits that exclude the extreme tails of a series. A handful of very large… (+2 more)

### Community 43 - "test_gaps.py"
Cohesion: 0.11
Nodes (6): Unit tests for the gap-anatomy and cadence functions this study relies on., _series(), TestCadenceEvidence, TestGapInventory, TestPhaseOneFigures, TestSegmentSurvival

### Community 44 - "monitoring.py"
Cohesion: 0.12
Nodes (22): alarm_episodes(), average_run_length(), cusum_chart(), detectability_curve(), ewma_chart(), inject_anomaly(), joint_alarm(), phase_shift_amplitude() (+14 more)

### Community 45 - "anomaly_by_channel"
Cohesion: 0.50
Nodes (4): anomaly_by_channel(), Departure of a channel from its own rolling median, and its ordinary scale.…, Whether each channel carries the anomaly, judged on its own ordinary noise. The…, residual_and_scale()

### Community 46 - "clock_cell"
Cohesion: 0.50
Nodes (4): clock_cell(), format_clock(), Render a fractional hour as a clock time. Parameters ---------- hours : float…, A phase hour formatted for a table, or nothing where there is no cycle. A…

### Community 47 - "solar_night_mask"
Cohesion: 0.50
Nodes (4): correct_sr_night(), True where the sun stands below a given elevation at the site. The archive…, Set the radiation recorded while the sun is down to the zero it must be. A…, solar_night_mask()

### Community 48 - "coverage_by_period"
Cohesion: 0.50
Nodes (4): coverage_by_period(), Fraction of each period observed, one column per station. Parameters ----------…, Fraction of each calendar year observed, one column per station. Parameters…, yearly_coverage()

### Community 49 - "export_columns"
Cohesion: 0.50
Nodes (4): export_columns(), The column order of the exported table. Grouped so that the file reads as what…, Write the archive table and the manifest that describes how it was built.…, save_archive()

### Community 50 - "filter_impulsive_segment"
Cohesion: 0.50
Nodes (4): filter_impulsive_segment(), hampel_filter(), Replace isolated spikes with a linear interpolation across them. Each sample is…, Filter isolated impulses inside one dated segment of a longer series. Values…

### Community 51 - "sentinel_census"
Cohesion: 0.50
Nodes (4): The interval over which each acquisition block was actually in service. A…, Measure every documented defect of the archive, block by block and channel by…, sentinel_census(), service_windows()

### Community 52 - "test_decomposition.py"
Cohesion: 0.10
Nodes (5): Unit tests for the decomposition, changepoint and metric additions to shmlib.…, TestBacktestDefaultsAreUnchanged, TestConformalInterval, TestCoveredChangepoints, TestRollingNowcast

### Community 53 - "Data exploration report review notes"
Cohesion: 0.50
Nodes (4): Data exploration report review notes, Academic report tone, Legacy-current instrument changeover, Three-station wall monitoring

### Community 54 - "test_neuralprophet_capabilities.py"
Cohesion: 0.16
Nodes (13): _frame(), _model(), Smoke tests for the NeuralProphet 0.8.0 capabilities Study 05 relies on. Run…, Spec D7: two daily series blended by float weights in 0..1.…, Spec D9: Model B carries lagged regressors with n_lags=0 on the target., Spec D8: split conformal prediction with the cqr method. For method='cqr',…, Spec D14: redraws read public methods, not figures., NeuralProphet 0.8.0's matplotlib plot_parameters always returns None, because… (+5 more)

### Community 56 - "_as_series"
Cohesion: 0.22
Nodes (10): _as_series(), cadence_evidence(), covered_changepoints(), gap_closure_summary(), hourly_change(), Summarise whether predicted hourly changes close observed inclination gaps.…, Trend changepoints placed on time the record actually covers. A changepoint…, Return ``values`` as a Series aligned to ``index``. (+2 more)

### Community 57 - "Study 4 · Decomposition, expectation and anomaly judgement"
Cohesion: 0.40
Nodes (4): Input, Outputs, Reproducing, Study 4 · Decomposition, expectation and anomaly judgement

### Community 61 - "Q: What connects Compensated inclination, Solar radiation, Wall temperature to the rest of the system?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: What connects Compensated inclination, Solar radiation, Wall temperature to the rest of the system?, Source Nodes

### Community 62 - "Q: Implement Study 04 using NeuralProphet to predict inclination from on-structure variables"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Implement Study 04 using NeuralProphet to predict inclination from on-structure variables, Source Nodes

### Community 65 - "add_inclination"
Cohesion: 0.50
Nodes (4): add_inclination(), Derive the inclination in millidegrees, and compensate it, block by block. Two…, compensate(), Apply the documented temperature compensation to the inclination. The…

### Community 68 - "last_archive_day"
Cohesion: 0.50
Nodes (4): last_archive_day(), Date of the most recent ``.adc`` file present in the archive. Parameters…, last_archive_day(), Date of the most recent ``.adc`` file present in the archive. Lets a study…

### Community 76 - "sync_archive"
Cohesion: 0.50
Nodes (4): Copy the archive files covering a date range to the local cache. A thin wrapper…, sync_archive(), Copy the ``.adc`` files covering a date range to a local cache. The archive is…, sync_cache()

### Community 82 - "lag_ranges"
Cohesion: 0.33
Nodes (4): lag_ranges(), The lags each driver may be scanned over, from what kind of thing it is. An…, Which drivers clear the control, and which of those move the right way. A…, shortlist()

### Community 84 - "test_folder_honesty.py"
Cohesion: 0.22
Nodes (4): Guards that the study folder never again claims a result it does not hold.…, TestNoCodeOutsideShmlib, TestReadmeMatchesTheFolder, TestReportClaimsAreSupported

### Community 94 - "plot_decomposition_stack"
Cohesion: 0.25
Nodes (8): plot_control_chart(), plot_decomposition_stack(), plot_prediction_band(), One panel per additive component, on a shared clock. The panels are stacked…, Observed against expected, with the prediction interval drawn behind them.…, A control statistic against its limits, with alarming episodes shaded.…, Reindex a series or frame onto a regular grid, or leave it untouched. Segmented…, _reindex_regular()

### Community 95 - "neuralprophet_backtest"
Cohesion: 0.25
Nodes (8): _analysis_freq(), backtest_specifications(), neuralprophet_backtest(), Walk-forward nowcast evaluation: keep every prediction as fresh as a deployed…, Run model specifications over expanding whole-segment folds. Parameters…, Infer an hourly-style frequency string for NeuralProphet., Fit one NeuralProphet model and return long out-of-sample predictions. The…, rolling_nowcast()

### Community 97 - "_long_predictions"
Cohesion: 0.25
Nodes (8): _drop_singleton_segments(), _long_predictions(), neuralprophet_predict(), _quantile_column(), Predict new same-time rows with an already fitted NeuralProphet model.…, Drop segments of fewer than two rows before a frame reaches NeuralProphet.…, Return NeuralProphet's quantile column name when present., Convert NeuralProphet's wide prediction table to study-long format.

### Community 99 - "Study 5 · Grey-box expectation and monitoring of the station 02 inclination"
Cohesion: 0.29
Nodes (6): Input, Outputs, Rebuilding the report, Reproducing, Study 5 · Grey-box expectation and monitoring of the station 02 inclination, Tests

### Community 103 - "baseline_predictions"
Cohesion: 0.50
Nodes (4): baseline_predictions(), Lag within each segment, never across segment IDs., Long zero, persistence and seasonal-naive predictions for fold test rows.…, _segment_lag()

### Community 104 - "decomposition_columns"
Cohesion: 0.50
Nodes (4): component_variance_shares(), decomposition_columns(), Component columns to treat as one term each, redundant aggregates dropped.…, What share of the fitted variation each additive component carries. This is the…

### Community 105 - "contiguous_segments"
Cohesion: 0.50
Nodes (4): contiguous_segments(), How many training windows survive contiguous segmentation, per configuration.…, Complete rows grouped into deterministic contiguous segment IDs. Rows with…, segment_survival()

### Community 106 - "_model_frame"
Cohesion: 0.50
Nodes (4): decompose_components(), _model_frame(), The additive parts NeuralProphet fitted, aligned to the study's index.…, Convert a study frame to NeuralProphet's ``ds``/``y`` table.

## Knowledge Gaps
- **28 isolated node(s):** `Input`, `Outputs`, `Reproducing`, `Input`, `Outputs` (+23 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **53 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TestScorePredictionsExtension` connect `TestScorePredictionsExtension` to `test_decomposition.py`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Why does `TestDrawCycle` connect `TestDrawCycle` to `figures.py`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **What connects `Input`, `Outputs`, `Reproducing` to the rest of the system?**
  _28 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `_diurnal` be split into smaller, more focused modules?**
  _Cohesion score 0.06636500754147813 - nodes in this community are weakly interconnected._
- **Should `calibrate_against_sensor` be split into smaller, more focused modules?**
  _Cohesion score 0.06854838709677419 - nodes in this community are weakly interconnected._
- **Should `_quiet` be split into smaller, more focused modules?**
  _Cohesion score 0.06183574879227053 - nodes in this community are weakly interconnected._
- **Should `tables.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10276679841897234 - nodes in this community are weakly interconnected._