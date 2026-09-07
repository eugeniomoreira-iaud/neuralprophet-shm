# Graph Report - studies  (2026-09-07)

## Corpus Check
- 56 files · ~184,962 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1491 nodes · 2212 edges · 141 communities (78 shown, 63 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 5 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `fc0bdebb`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- _diurnal
- calibrate_against_sensor
- _quiet
- tables.py
- apply_report_style
- coupling.py
- figures.py
- de_lib.py
- finish
- Inclinometer conditioner output
- write_adc
- solar_noon_utc
- TestPlotDiurnalSeasonGrid
- adc.py
- plot_coupling_scatter
- WallTemperatureFilterTests
- TestDrawCycle
- _frame
- TestMeteo
- TestTables
- TestViz
- TestSensorLoading
- season_of
- load_sensor_package
- test_prediction.py
- test_monitor.py
- solar_elevation
- gubbio_archive_20min.csv
- TestAgreement
- TestCompensation
- TestSolar
- meteosystem_gubbio.csv (Gubbio town ground station)
- TestCertifiedWindow
- TestClockConversion
- TestShortlist
- TestResponseLoading
- TestSite
- ReportSpanHighlightTests
- TestVerdictTable
- TestMaskImplausible
- prediction.py
- phase_chain
- TestSeasons
- test_gaps.py
- monitoring.py
- anomaly_by_channel
- clock_cell
- 3 Method
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
- test_harmonics.py
- build_grid
- clock_step
- Q: What connects Compensated inclination, Solar radiation, Wall temperature to the rest of the system?
- Q: Implement Study 04 using NeuralProphet to predict inclination from on-structure variables
- condemned_modes
- _daily_signal
- add_inclination
- current_era_availability
- 05_greybox_monitoring/tests/test_folder_honesty.py
- last_archive_day
- parse_era
- station_classification
- summary_by_era
- response_ratios
- temporal_alignment.py
- spike_replacement_summary
- test_proxies_grid.py
- sync_archive
- TestModelFigures
- _parallel_map
- circular_std
- wind_components
- TestOutageBridge
- bootstrap_correlation_delta
- TestFiguresStudy05
- test_folder_honesty.py
- certified_sr_window
- sr_day_census
- verdict_table
- load_proxy_variants
- TestSingletonSegmentDroppedBeforeFitting
- set_context
- Study 3 README
- TestComponentVarianceShares
- TestPeriodScan
- plot_decomposition_stack
- TestShiftSummary
- Study 2 must not re-decide Study 1's rejections
- neuralprophet_backtest
- TestSingletonSegmentDroppedBeforePrediction
- Study 5 · Grey-box expectation and monitoring of the station 02 inclination
- TestNativeDelayScan
- TestResidualDiagnostics
- TestWindComponents
- baseline_predictions
- _model_frame
- contiguous_segments
- TestConformalInterval
- _scan
- TestReferenceShiftScanFigure
- TestRollingNowcast
- channel_inventory
- run_experiment
- TestLaggedWeights
- mask_implausible
- source_inventory
- clock_check
- seasonal_weights
- TestCircularHandling
- TestInventories
- TestReferenceShiftScan
- TestLagCurvesFigure
- TestGainStabilityFigure
- Q: I need help to figure out if one idea is good for neuralprophet-shm/studies/05_greybox_monitoring. Can you check the content to gain some context, then I tell you my idea?
- Q: Can radiation be transformed so nighttime cooling is represented as a negative effect and the regressor resembles compensated inclination more closely?
- TestRestrictToDays
- TestJoinEras
- setup_sidequest.py
- test_run_metadata.py
- fold_stability
- _score_group
- build_regressor_sets
- reference_shift_scan
- assemble_wide
- coincidence
- condemned_by_month
- diurnal_response
- export_column_doc
- season_changepoints
- oikolab_weather.csv (ERA5 reanalysis via Oikolab)
- regressor_set_coverage
- show_static

## God Nodes (most connected - your core abstractions)
1. `finish()` - 50 edges
2. `format_spines()` - 48 edges
3. `figsize()` - 32 edges
4. `_hourly()` - 24 edges
5. `_diurnal()` - 24 edges
6. `_quiet()` - 20 edges
7. `_frame()` - 19 edges
8. `apply_report_style()` - 15 edges
9. `3 Method` - 15 edges
10. `TestTables` - 13 edges

## Surprising Connections (you probably didn't know these)
- `diurnal_cycle_grid()` --calls--> `finish()`  [EXTRACTED]
  01_data_exploration/de_lib.py → shmlib/viz.py
- `diurnal_cycle_grid()` --calls--> `format_spines()`  [EXTRACTED]
  01_data_exploration/de_lib.py → shmlib/viz.py
- `diurnal_cycle_pair()` --calls--> `apply_report_style()`  [EXTRACTED]
  01_data_exploration/de_lib.py → shmlib/viz.py
- `diurnal_cycle_pair()` --calls--> `finish()`  [EXTRACTED]
  01_data_exploration/de_lib.py → shmlib/viz.py
- `diurnal_cycle_pair()` --calls--> `format_spines()`  [EXTRACTED]
  01_data_exploration/de_lib.py → shmlib/viz.py

## Import Cycles
- 3-file cycle: `shmlib/__init__.py -> shmlib/quality.py -> shmlib/coupling.py -> shmlib/__init__.py`
- 3-file cycle: `shmlib/__init__.py -> shmlib/monitoring.py -> shmlib/coupling.py -> shmlib/__init__.py`
- 3-file cycle: `shmlib/__init__.py -> shmlib/proxies.py -> shmlib/coupling.py -> shmlib/__init__.py`
- 4-file cycle: `shmlib/__init__.py -> shmlib/proxies.py -> shmlib/quality.py -> shmlib/coupling.py -> shmlib/__init__.py`
- 4-file cycle: `shmlib/__init__.py -> shmlib/viz.py -> shmlib/proxies.py -> shmlib/coupling.py -> shmlib/__init__.py`
- 5-file cycle: `shmlib/__init__.py -> shmlib/viz.py -> shmlib/proxies.py -> shmlib/quality.py -> shmlib/coupling.py -> shmlib/__init__.py`

## Hyperedges (group relationships)
- **Study 2's three input data sources** — 02_proxy_forcing_characterization_readme, data_raw_proxies_oikolab_weather, data_raw_proxies_meteosystem_gubbio [EXTRACTED 0.90]
- **Study 2's analysis pipeline modules (proxies, quality, compare, figures)** — shmlib_proxies, shmlib_quality, shmlib_compare, shmlib_figures, 02_proxy_forcing_characterization_readme [EXTRACTED 0.95]
- **Monitoring chain from solar forcing to wall response** — 01_data_exploration_report_data_exploration_report_solar_radiation, 01_data_exploration_report_data_exploration_report_wall_temperature, 01_data_exploration_report_data_exploration_report_inclinometer [INFERRED 0.85]

## Communities (141 total, 63 thin omitted)

### Community 0 - "_diurnal"
Cohesion: 0.06
Nodes (24): _diurnal(), _hourly(), The second parameter: inertia, which a pure delay cannot express., The grid, and what it says that a single lag curve cannot., A negative coupling is invisible to a search that maximises ``r``., The filter must remove the drift and leave the daily cycle alone., The slope, and the standard error an hourly series actually deserves., The orchestration: every driver, in every band, over every stratum. (+16 more)

### Community 1 - "calibrate_against_sensor"
Cohesion: 0.08
Nodes (30): anomaly_and_reference(), The anomaly window, and the same calendar window averaged over earlier years.…, _agreement_scores(), agreement_stability(), calibrate_against_sensor(), daily_amplitude_phase(), diurnal_profile(), diurnal_table() (+22 more)

### Community 2 - "_quiet"
Cohesion: 0.05
Nodes (14): _quiet(), Unit tests for shmlib.monitoring. Run from studies/: python…, TestAlarmEpisodes, TestAverageRunLength, TestCusumChart, TestDetectabilityCurve, TestDetectabilityKinds, TestEwmaChart (+6 more)

### Community 3 - "tables.py"
Cohesion: 0.09
Nodes (23): basename(), _cell(), date_cell(), latex_escape(), percent(), Module: shmlib.tables Writing the LaTeX table bodies that the study reports…, Format a table into LaTeX rows, one string per row of ``frame``. Parameters…, Write LaTeX rows to a table body file. The rows are joined by the row separator… (+15 more)

### Community 4 - "apply_report_style"
Cohesion: 0.09
Nodes (32): diurnal_cycle_grid(), plot_anomaly_channels(), plot_anomaly_comparison(), plot_channel_series(), plot_coverage_heatmap(), plot_current_era_series(), plot_inclination_series(), plot_segment() (+24 more)

### Community 5 - "coupling.py"
Cohesion: 0.07
Nodes (38): _annual_design(), annual_modulation(), best_operator(), centre_phase(), couple(), cycle_surface_rank(), diurnal_band(), driver_coverage() (+30 more)

### Community 6 - "figures.py"
Cohesion: 0.16
Nodes (16): # TODO: write one row per quantity per source once the numbers above have been…, Study 2 — Proxy Forcing Characterization (README), Usable radiation window (n_sr_ok night-zero correction intersected with sr_suspect's 161-day exclusion), proxy_forcing_report.tex, Unit tests for the decisions study 2 relies on, now that they live in…, Unit tests for the decisions study 3 relies on, which live in ``shmlib``. Run…, Module: shmlib.compare How a source behaves on its own, and how two sources…, Module: shmlib.figures The multi-source figures a proxy-forcing comparison… (+8 more)

### Community 7 - "de_lib.py"
Cohesion: 0.08
Nodes (23): block_view(), coalesce_target(), excursions_by_month(), extreme_excursions(), flag_and_correct(), Module: de_lib.py Support library for the data-exploration study…, Build one continuous series for the target station across both eras. The legacy…, One acquisition block as a tidy frame, on corrected values. The wide table is… (+15 more)

### Community 8 - "finish"
Cohesion: 0.08
Nodes (46): plot_agreement_stability(), plot_cadence_evidence(), plot_certified_window(), plot_daily_harmonic_chart(), plot_detectability(), plot_diurnal_comparison(), plot_fit_metrics(), plot_gap_anatomy() (+38 more)

### Community 9 - "Inclinometer conditioner output"
Cohesion: 0.12
Nodes (17): The monitoring system of the Mura Urbiche di Gubbio and its record, gubbio_archive_20min.csv exported dataset, Field-wise decimal-separator normalization, Twenty-four-hour Hampel impulsive-noise filter, Inclinometer conditioner output, Legacy-to-current hardware changeover, Mura Urbiche of Gubbio, Radiation day-quality verdict (+9 more)

### Community 10 - "write_adc"
Cohesion: 0.15
Nodes (8): The separator is mixed, sometimes within a single record., `CURRENT_RECORD` writes batt with a comma and tair with a point., Write lines to a temporary .adc file and return its path., The two era parsers, and the boundary between them., A file spanning the changeover carries both layouts. Each parser must take its…, TestDecimalSeparator, TestParsers, write_adc()

### Community 11 - "solar_noon_utc"
Cohesion: 0.33
Nodes (6): clock_offset(), The logger's clock offset from UTC, measured against computed solar noon. The…, _equation_of_time(), The equation of time, in minutes, for a fractional year angle. Shared by…, Solar noon in UTC hours, for each date, by the NOAA formulation. Accounts for…, solar_noon_utc()

### Community 12 - "TestPlotDiurnalSeasonGrid"
Cohesion: 0.10
Nodes (13): `figures.plot_diurnal_season_grid`, the channel-by-season diurnal grid., Five hourly days each of summer and winter, on two channels of very different…, A temperature channel and a wind-direction channel, five hourly days each…, The returned figure carries exactly `len(columns) * len(seasons)` axes, laid…, The two seasonal panels of one row must return the same `get_ylim()`, since a…, The season name appears once, capitalised, at the start of the top-row panel…, The row of a channel declared circular is drawn on a 0-360 degree compass axis…, `figures.plot_channel_panels`, the channel-named resolver over… (+5 more)

### Community 13 - "adc.py"
Cohesion: 0.27
Nodes (9): _parse(), parse_file(), parse_legacy_file(), Module: shmlib.adc The raw ``.adc`` archive: its file format, the constants of…, Parse one numeric field, accepting either decimal separator. The archive mixes…, Parse the records of one era out of one ``.adc`` file. Shared body of…, Parse the current-era records of one ``.adc`` file. Handles the mixed decimal…, Parse the legacy-era records of one ``.adc`` file. Differs from… (+1 more)

### Community 14 - "plot_coupling_scatter"
Cohesion: 0.15
Nodes (16): plot_coupling_scatter(), plot_gain_stability(), plot_impulse_response(), plot_lag_curves(), plot_operator_grid(), The correlation of each driver against the response, at every lag scanned. The…, The response against one driver, at the lag the scan chose. Drawn as a hexbin…, Each driver's gain re-estimated window by window, one panel per driver. One… (+8 more)

### Community 15 - "WallTemperatureFilterTests"
Cohesion: 0.15
Nodes (7): Tests for auditable wall-temperature impulse filtering., A spike inside the segment is interpolated; earlier data is untouched., Filtering an observed spike must not fill a pre-existing missing slot., A run boundary cannot borrow context from the opposite side of a gap., Archive exports must include filtered values and their provenance flag., The analysis view must preserve corrected and filtered wall temperatures., WallTemperatureFilterTests

### Community 16 - "TestDrawCycle"
Cohesion: 0.15
Nodes (7): `viz.draw_cycle`, the single diurnal panel every study draws with., With `complete_day` set, a day short of that many slots contributes neither to…, `centre=True` returns a cycle whose own mean is zero and `centre=False` returns…, A position within the day is left `NaN` when fewer than `min_days` distinct…, `circular=True` averages by the unit-vector method, matching…, `background=True` draws every individual day behind the mean and…, TestDrawCycle

### Community 17 - "_frame"
Cohesion: 0.08
Nodes (12): _frame(), Tests for the Model A additions Study 05 makes to shmlib.prediction. Run from…, `prediction.ladder_frame` and `prediction.channel_ladder`, the current-era…, Task 5.4c: process-level parallelism over independent fits reproduces the…, Two hand-built regressor sets sharing one target, for the Movement 2…, _regressor_sets(), TestBacktestAdditions, TestExtractors (+4 more)

### Community 19 - "TestTables"
Cohesion: 0.13
Nodes (4): The LaTeX row convention, and what a missing value prints as., A trailing separator opens an empty row that \\bottomrule lands inside., Unit strings are markup, not data, and must reach LaTeX intact., TestTables

### Community 20 - "TestViz"
Cohesion: 0.17
Nodes (3): The figure conventions that the Graphical Guidelines make binding., Study 1's `_complete_days` and `_draw_cycle` moved here as `complete_days` and…, TestViz

### Community 21 - "TestSensorLoading"
Cohesion: 0.12
Nodes (6): ``stamp_offset`` moves the value read at a timestamp, nothing else., Write a frame to a temporary CSV and return its path., Study 1's verdicts are consumed, never revisited., TestGroundStationStampOffset, TestSensorLoading, _write_csv()

### Community 22 - "season_of"
Cohesion: 0.16
Nodes (14): The slices a screen is run over, as boolean masks on one index. Three kinds,…, strata(), plot_diurnal_grid(), plot_diurnal_season_grid(), plot_diurnal_source_grid(), Several sources' mean daily cycles overlaid, a quantity per row and a season…, The mean daily cycle of every channel of one source, as a grid of panels. One…, One source's mean daily cycles, a channel per row and a season per column. The… (+6 more)

### Community 23 - "load_sensor_package"
Cohesion: 0.18
Nodes (12): join_eras(), load_response(), load_sensor_forcings(), Load the on-structure forcings from study 1's archive onto the analysis grid.…, Load the structural response from study 1's archive onto the analysis grid. The…, One channel per quantity, from the block that was recording. The archive keeps…, load_sr_record(), The radiation channel with its verdicts beside it, before the verdicts bite.… (+4 more)

### Community 24 - "test_prediction.py"
Cohesion: 0.04
Nodes (27): Tests for Study 04 prediction helpers. Run directly, with no test runner…, Routes must reflect what information exists for each prediction task., Prediction scores use paired observed and predicted values only., Bootstrap skill must keep parent and child errors timestamp-aligned., Robust quantile column matching across varied library string formats., A tiny real fit protects the wrapper contract at the library boundary., First differences must not bridge records the model may not learn across., Notebook orchestration must keep fold boundaries explicit. (+19 more)

### Community 25 - "test_monitor.py"
Cohesion: 0.08
Nodes (13): _ar1(), _quiet(), Tests for the monitor additions of Study 05 (spec D10, D11). Run from studies/:…, TestAttributeEpisodes, TestChannelCoincidence, TestChartSeries, TestDailyResponseAmplitude, TestDetectabilityByMechanism (+5 more)

### Community 26 - "solar_elevation"
Cohesion: 0.13
Nodes (16): correct_sr_night(), flag_sr_day_quality(), Name the single state each day of the failure window belongs to. The window is…, True where the sun stands below a given elevation at the site. The archive…, Set the radiation recorded while the sun is down to the zero it must be. A…, The radiation that survived rejection, before the night correction. Both…, Condemn the days on which the radiation channel has no diurnal cycle. A working…, Solar radiation grouped by solar elevation, condemned days against the rest.… (+8 more)

### Community 27 - "gubbio_archive_20min.csv"
Cohesion: 0.25
Nodes (8): Study 1 data exploration README, Raw ADC archive ingest, Mura Urbiche di Gubbio monitoring record, Inclination cleaning chain, Night radiation correction, Radiation suspect-day verdict, Recorded flag and ok-column schema, gubbio_archive_20min.csv

### Community 29 - "TestCompensation"
Cohesion: 0.25
Nodes (3): The documented formula, and the anchoring it depends on., The coefficient is 0.005, applied times 1000: 5 mdeg per °C., TestCompensation

### Community 37 - "ReportSpanHighlightTests"
Cohesion: 0.33
Nodes (4): Regression tests for span highlights in the generated report figures., The report uses one visual treatment for every highlighted span., Every generated span is black at 5% opacity, with none omitted., ReportSpanHighlightTests

### Community 40 - "prediction.py"
Cohesion: 0.06
Nodes (31): availability_route(), conformal_interval(), _detach_trainer(), execution_folds(), expanding_segment_folds(), gap_inventory(), has_certified_period(), impulse_response_summary() (+23 more)

### Community 41 - "phase_chain"
Cohesion: 0.12
Nodes (16): cycle_extreme(), diurnal_cycle_pair(), phase_chain(), Position of a cycle's extreme, or nothing when the extreme is a plateau. A mean…, Average diurnal cycle of a current-era channel, by season. The current-era…, The four quantities of the thermal chain, on one clock and one scale. The sun…, _clip_note(), plot_channel_panels() (+8 more)

### Community 43 - "test_gaps.py"
Cohesion: 0.11
Nodes (6): Unit tests for the gap-anatomy and cadence functions this study relies on., _series(), TestCadenceEvidence, TestGapInventory, TestPhaseOneFigures, TestSegmentSurvival

### Community 44 - "monitoring.py"
Cohesion: 0.07
Nodes (44): alarm_episodes(), attribute_episodes(), average_run_length(), channel_coincidence(), chart_series(), cusum_chart(), daily_harmonic(), daily_response_amplitude() (+36 more)

### Community 45 - "anomaly_by_channel"
Cohesion: 0.50
Nodes (4): anomaly_by_channel(), Departure of a channel from its own rolling median, and its ordinary scale.…, Whether each channel carries the anomaly, judged on its own ordinary noise. The…, residual_and_scale()

### Community 46 - "clock_cell"
Cohesion: 0.50
Nodes (4): clock_cell(), format_clock(), Render a fractional hour as a clock time. Parameters ---------- hours : float…, A phase hour formatted for a table, or nothing where there is no cycle. A…

### Community 47 - "3 Method"
Cohesion: 0.07
Nodes (29): 10 The monitor, 11 Outages as hypotheses, 12 Verdict, 13 Limitations, 14 Run metadata, 1 Introduction, 2 The record and the three regressor sets, 3.10 D10 · Three charts, one per damage mechanism and time scale (+21 more)

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
Cohesion: 0.25
Nodes (3): Unit tests for the decomposition, changepoint and metric additions to shmlib.…, TestBacktestDefaultsAreUnchanged, TestCoveredChangepoints

### Community 53 - "Data exploration report review notes"
Cohesion: 0.50
Nodes (4): Data exploration report review notes, Academic report tone, Legacy-current instrument changeover, Three-station wall monitoring

### Community 54 - "test_neuralprophet_capabilities.py"
Cohesion: 0.15
Nodes (14): _frame(), _model(), Smoke tests for the NeuralProphet 0.8.0 capabilities Study 05 relies on. Run…, Spec D14: redraws read public methods, not figures., NeuralProphet 0.8.0's matplotlib plot_parameters always returns None, because…, Spec D7: two daily series blended by float weights in 0..1.…, Spec D9: Model B carries lagged regressors with n_lags=0 on the target., Spec D8: split conformal prediction with the cqr method. For method='cqr',… (+6 more)

### Community 56 - "_as_series"
Cohesion: 0.18
Nodes (12): _as_series(), cadence_evidence(), covered_changepoints(), gap_closure_summary(), hourly_change(), Walk-forward nowcast evaluation: keep every prediction as fresh as a deployed…, Summarise whether predicted hourly changes close observed inclination gaps.…, Trend changepoints placed on time the record actually covers. A changepoint… (+4 more)

### Community 57 - "Study 4 · Decomposition, expectation and anomaly judgement"
Cohesion: 0.40
Nodes (4): Input, Outputs, Reproducing, Study 4 · Decomposition, expectation and anomaly judgement

### Community 58 - "test_harmonics.py"
Cohesion: 0.09
Nodes (10): _days(), Tests for the harmonic diagnostics of Study 05 (spec D6). Run from studies/:…, TestAnnualModulation, TestCentrePhase, TestCycleSurfaceRank, TestDailyHarmonic, TestHasCertifiedPeriod, TestOlsResidual (+2 more)

### Community 61 - "Q: What connects Compensated inclination, Solar radiation, Wall temperature to the rest of the system?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: What connects Compensated inclination, Solar radiation, Wall temperature to the rest of the system?, Source Nodes

### Community 62 - "Q: Implement Study 04 using NeuralProphet to predict inclination from on-structure variables"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Implement Study 04 using NeuralProphet to predict inclination from on-structure variables, Source Nodes

### Community 64 - "_daily_signal"
Cohesion: 0.16
Nodes (6): _daily_signal(), Tests for the temporal alignment side quest. Run from studies/: python…, A repeatable diurnal signal with enough structure for lag tests., TestBootstrapCorrelationDelta, TestPairedShiftScan, TestRunExperimentContract

### Community 65 - "add_inclination"
Cohesion: 0.50
Nodes (4): add_inclination(), Derive the inclination in millidegrees, and compensate it, block by block. Two…, compensate(), Apply the documented temperature compensation to the inclination. The…

### Community 67 - "05_greybox_monitoring/tests/test_folder_honesty.py"
Cohesion: 0.14
Nodes (8): _notebook_artefact_names(), Guards that the study folder never claims a result it does not hold. Run from…, The filenames and table bodies the notebook writes, read from its own source…, Implements the global constraint "Every image in the report comes from the…, TestEveryGraphicComesFromTheNotebook, TestNoCodeOutsideShmlib, TestReadmeMatchesTheFolder, TestReportClaimsAreSupported

### Community 68 - "last_archive_day"
Cohesion: 0.50
Nodes (4): last_archive_day(), Date of the most recent ``.adc`` file present in the archive. Parameters…, last_archive_day(), Date of the most recent ``.adc`` file present in the archive. Lets a study…

### Community 73 - "temporal_alignment.py"
Cohesion: 0.21
Nodes (16): _dst_state(), _file_record(), _local_dates(), _manifest(), _not_transition_mask(), _period_bound(), _period_mask(), Module: shmlib.temporal_alignment Runs the temporal-alignment side quest for… (+8 more)

### Community 75 - "test_proxies_grid.py"
Cohesion: 0.18
Nodes (4): Tests for the proxy grid helpers Study 05 adds to shmlib.proxies. Run from…, TestFillShortGaps, TestRegressorSets, TestToNativeGrid

### Community 76 - "sync_archive"
Cohesion: 0.50
Nodes (4): Copy the archive files covering a date range to the local cache. A thin wrapper…, sync_archive(), Copy the ``.adc`` files covering a date range to a local cache. The archive is…, sync_cache()

### Community 78 - "_parallel_map"
Cohesion: 0.14
Nodes (14): attribution_fits(), channel_ladder(), outage_bridge(), paired_mae_skill(), _parallel_map(), Apply ``function`` to every element of ``items``, optionally in worker…, Rebuild the trainer :func:`_detach_trainer` dropped, from the model's own…, Score a nowcast fit at each candidate trend regularisation (spec D7). One… (+6 more)

### Community 81 - "TestOutageBridge"
Cohesion: 0.18
Nodes (5): _frame(), Tests for the outage bridge (spec D12). Run from studies/: python…, A tiny hourly frame with a target and one regressor, for fitting a real (if…, TestOutageBridge, TestPlotOutageBridge

### Community 82 - "bootstrap_correlation_delta"
Cohesion: 0.22
Nodes (13): _aligned_delta_series(), _block_stats(), bootstrap_correlation_delta(), _candidate_label(), _corrs(), _daily_demean(), _delta_from_stats(), _joint_env_delta() (+5 more)

### Community 84 - "test_folder_honesty.py"
Cohesion: 0.22
Nodes (4): Guards that the study folder never again claims a result it does not hold.…, TestNoCodeOutsideShmlib, TestReadmeMatchesTheFolder, TestReportClaimsAreSupported

### Community 88 - "load_proxy_variants"
Cohesion: 0.22
Nodes (10): load_era5(), load_ground_station(), Read one external proxy export and put it on the analysis grid. Shared by…, Load the ERA5 export onto the analysis grid. The export is natively hourly and…, Load the ground-station export onto the analysis grid. The export is natively…, Bring hourly or half-hourly proxies onto the sensor's native grid. Each column…, _read_proxy(), to_native_grid() (+2 more)

### Community 94 - "plot_decomposition_stack"
Cohesion: 0.20
Nodes (10): plot_control_chart(), plot_decomposition_stack(), plot_prediction_band(), plot_trend_parameters(), One panel per additive component, on a shared clock. The panels are stacked…, Observed against expected, with the prediction interval drawn behind them.…, A control statistic against its limits, with alarming episodes shaded.…, The trend on covered time above, the rate of each segment below; changepoints… (+2 more)

### Community 97 - "neuralprophet_backtest"
Cohesion: 0.15
Nodes (14): _analysis_freq(), backtest_specifications(), _drop_singleton_segments(), _long_predictions(), neuralprophet_backtest(), neuralprophet_predict(), _quantile_column(), Predict new same-time rows with an already fitted NeuralProphet model.… (+6 more)

### Community 99 - "Study 5 · Grey-box expectation and monitoring of the station 02 inclination"
Cohesion: 0.29
Nodes (6): Input, Outputs, Rebuilding the report, Reproducing, Study 5 · Grey-box expectation and monitoring of the station 02 inclination, Tests

### Community 100 - "TestNativeDelayScan"
Cohesion: 0.43
Nodes (3): Step 7.5's native-resolution delay: the pairs it produces, and its sign., A response that lags a diurnal driver by `delay_steps` twenty-minute slots, and…, TestNativeDelayScan

### Community 103 - "baseline_predictions"
Cohesion: 0.50
Nodes (4): baseline_predictions(), Lag within each segment, never across segment IDs., Long zero, persistence and seasonal-naive predictions for fold test rows.…, _segment_lag()

### Community 104 - "_model_frame"
Cohesion: 0.13
Nodes (16): compare_daily_terms(), component_variance_shares(), decompose_components(), decomposition_columns(), _frame_columns(), _model_frame(), Every column a model frame needs: the requested regressors, plus whatever the…, The additive parts NeuralProphet fitted, aligned to the study's index.… (+8 more)

### Community 105 - "contiguous_segments"
Cohesion: 0.50
Nodes (4): contiguous_segments(), How many training windows survive contiguous segmentation, per configuration.…, Complete rows grouped into deterministic contiguous segment IDs. Rows with…, segment_survival()

### Community 107 - "_scan"
Cohesion: 0.38
Nodes (7): _as_series(), paired_shift_scan(), Scan timestamp shifts for one sensor-reference pair. Positive shifts assign the…, _scan(), _shift_delta(), _shifted_pair_series(), _shifted_reference()

### Community 111 - "run_experiment"
Cohesion: 0.22
Nodes (10): harmonise(), Join the loaded sources onto one index under one naming scheme. An outer join,…, _choice_rows(), _fisher_z(), _json_safe(), _Pair, Run the complete temporal-alignment protocol. Parameters ---------- archive_csv…, run_experiment() (+2 more)

### Community 115 - "clock_check"
Cohesion: 0.33
Nodes (6): best_lag(), lag_scan(), Correlation between a response and a driver at every lag in a range. The whole…, The winning lag of a scan, under the stated objective. Parameters ----------…, clock_check(), Two independent tests of the clock each source keeps, run on one quantity. A…

### Community 116 - "seasonal_weights"
Cohesion: 0.33
Nodes (6): ladder_frame(), Condition columns for the smoothly weighted daily seasonality (spec D7). Two…, Model-ready frames for Model A, one per regressor set (spec D7). Each regressor…, The current-era window plus the wall probe and the on-structure pyranometer,…, regressor_set_frames(), seasonal_weights()

### Community 122 - "Q: I need help to figure out if one idea is good for neuralprophet-shm/studies/05_greybox_monitoring. Can you check the content to gain some context, then I tell you my idea?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: I need help to figure out if one idea is good for neuralprophet-shm/studies/05_greybox_monitoring. Can you check the content to gain some context, then I tell you my idea?, Source Nodes

### Community 123 - "Q: Can radiation be transformed so nighttime cooling is represented as a negative effect and the regressor resembles compensated inclination more closely?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Can radiation be transformed so nighttime cooling is represented as a negative effect and the regressor resembles compensated inclination more closely?, Source Nodes

### Community 126 - "setup_sidequest.py"
Cohesion: 0.50
Nodes (3): Regenerate ``test_24_changepoints.py`` from the parent study script. Run this…, Replace, in place, the single line in ``lines`` that starts with…, substitute_line()

### Community 128 - "fold_stability"
Cohesion: 0.50
Nodes (4): fold_stability(), Rows of ``block`` whose index appears in ``ds``, a fold boundary column from…, Component stability across NeuralProphet's own chronological folds (spec §4.1).…, _select_by_ds()

### Community 129 - "_score_group"
Cohesion: 0.50
Nodes (4): Compute scalar scores for one already selected group., Score observed and predicted values, optionally by group. The input must…, _score_group(), score_predictions()

### Community 130 - "build_regressor_sets"
Cohesion: 0.50
Nodes (4): build_regressor_sets(), fill_short_gaps(), Fill short regressor dropouts by linear interpolation, and say where. A…, Assemble the regressor sets of one model specification from a joined record.…

### Community 131 - "reference_shift_scan"
Cohesion: 0.50
Nodes (4): Scan timestamp shifts for one sensor-reference pair, displacing the reference.…, Tag and concatenate :func:`reference_shift_scan` over pairs, periods and record…, reference_shift_scan(), scan_reference_pairs()

## Knowledge Gaps
- **61 isolated node(s):** `Input`, `Outputs`, `Reproducing`, `Input`, `Outputs` (+56 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 708 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **63 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TestFiguresStudy05` connect `TestFiguresStudy05` to `figures.py`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Why does `TestShiftSummary` connect `TestShiftSummary` to `figures.py`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **What connects `Input`, `Outputs`, `Reproducing` to the rest of the system?**
  _61 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `_diurnal` be split into smaller, more focused modules?**
  _Cohesion score 0.05868118572292801 - nodes in this community are weakly interconnected._
- **Should `calibrate_against_sensor` be split into smaller, more focused modules?**
  _Cohesion score 0.07586206896551724 - nodes in this community are weakly interconnected._
- **Should `_quiet` be split into smaller, more focused modules?**
  _Cohesion score 0.05387205387205387 - nodes in this community are weakly interconnected._
- **Should `tables.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09333333333333334 - nodes in this community are weakly interconnected._