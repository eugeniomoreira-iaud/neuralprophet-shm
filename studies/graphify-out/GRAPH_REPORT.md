# Graph Report - studies  (2026-08-22)

## Corpus Check
- 32 files · ~1,175,518 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 822 nodes · 1130 edges · 116 communities (50 shown, 66 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `bf5aadf0`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- _diurnal
- calibrate_against_sensor
- Departure from the 24-hour rolling median, summer 2026 anomaly window
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
- adc.py
- plot_coupling_scatter
- WallTemperatureFilterTests
- TestDrawCycle
- finish
- TestMeteo
- TestTables
- TestViz
- TestSensorLoading
- season_of
- proxies.py
- test_prediction.py
- phase_chain
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
- plot_source_panels
- test_shmlib_study02.py
- TestCircularHandling
- TestInventories
- anomaly_by_channel
- clock_cell
- solar_night_mask
- coverage_by_period
- export_columns
- filter_impulsive_segment
- sentinel_census
- Station Locations Map
- Data exploration report review notes
- TestRestrictToDays
- Diurnal Cycles (st02)
- T-Wall Filtered Chart
- Study 4 · NeuralProphet inclination prediction
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
- oikolab_weather.csv (ERA5 reanalysis via Oikolab)
- scan_raw_text
- circular_std
- wind_components
- driver_coverage
- lag_ranges
- r2_ceiling
- shortlist
- certified_sr_window
- sr_day_census
- verdict_table
- viz.py
- Study 3 README
- Inclination Cleaned (st02)
- Battery (st02)
- Current Era Coverage (st02)
- Twall Raw (st02)
- Study 2 must not re-decide Study 1's rejections
- DE F04 ST02 inclination time series
- ST02 inclination summer 2026 anomaly versus historical average
- Summer 2026 filtered measurements against historical average
- Full compensated, offset-removed, and cleaned inclination series
- Average diurnal cycle of inclination by era and season
- Average diurnal cycle of air temperature by era and season
- Average diurnal cycle of the relative humidity, by era and season
- st02: supply voltage
- Daily availability of the current-era channels
- st02 current-era wall temperature as written
- st02 current-era wall temperature after filtering
- Final wall-temperature segment, before filtering
- T-Wall Final Segment Filtered Chart
- Final wall-temperature segment, filtered
- Average diurnal cycle of the wall temperature, current era
- SR Chart
- st02, current era: solar radiation as written
- SR Failure Chart
- st02: the radiation failure, 2025-07-27 to 2026-06-17
- Average diurnal cycle of the solar radiation, unaffected days only
- SR Diurnal Cycles Chart

## God Nodes (most connected - your core abstractions)
1. `finish()` - 31 edges
2. `format_spines()` - 29 edges
3. `_hourly()` - 24 edges
4. `_diurnal()` - 24 edges
5. `apply_report_style()` - 15 edges
6. `figsize()` - 13 edges
7. `TestMeteo` - 12 edges
8. `Study 2 — Proxy Forcing Characterization (README)` - 11 edges
9. `TestViz` - 10 edges
10. `TestTables` - 10 edges

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
- **Summer diurnal cycle variables** — 01_data_exploration_outputs_de_f24_st02_phase_chain_solar_radiation, 01_data_exploration_outputs_de_f24_st02_phase_chain_air_temperature, 01_data_exploration_outputs_de_f24_st02_phase_chain_wall_temperature, 01_data_exploration_outputs_de_f24_st02_phase_chain_compensated_inclination [EXTRACTED 1.00]
- **Site orientation and monitoring context** — studies_01_data_exploration_outputs_de_f02_st02_sketch_monitoring_station_02, studies_01_data_exploration_outputs_de_f02_st02_sketch_gate_of_sant_ubaldo, studies_01_data_exploration_outputs_de_f02_st02_sketch_valley, studies_01_data_exploration_outputs_de_f02_st02_sketch_mountain [EXTRACTED 1.00]
- **Monitoring chain from solar forcing to wall response** — 01_data_exploration_report_data_exploration_report_solar_radiation, 01_data_exploration_report_data_exploration_report_wall_temperature, 01_data_exploration_report_data_exploration_report_inclinometer [INFERRED 0.85]
- **st02 Diurnal Cycles Group** — 01_data_exploration_outputs_de_f08_st02_diurnal_cycles, 01_data_exploration_outputs_de_f10_st02_tair_diurnal_cycles, 01_data_exploration_outputs_de_f12_st02_rh_diurnal_cycles [INFERRED 0.95]

## Communities (116 total, 66 thin omitted)

### Community 0 - "_diurnal"
Cohesion: 0.05
Nodes (29): _diurnal(), _hourly(), Unit tests for the decisions study 3 relies on, which live in ``shmlib``. Run…, The second parameter: inertia, which a pure delay cannot express., The grid, and what it says that a single lag curve cannot., A negative coupling is invisible to a search that maximises ``r``., The filter must remove the drift and leave the daily cycle alone., The slope, and the standard error an hourly series actually deserves. (+21 more)

### Community 1 - "calibrate_against_sensor"
Cohesion: 0.09
Nodes (26): anomaly_and_reference(), The anomaly window, and the same calendar window averaged over earlier years.…, _agreement_scores(), agreement_stability(), calibrate_against_sensor(), daily_amplitude_phase(), diurnal_profile(), diurnal_table() (+18 more)

### Community 2 - "Departure from the 24-hour rolling median, summer 2026 anomaly window"
Cohesion: 0.10
Nodes (22): Daily Inclinometer Availability by Station, Station Coverage, Air temperature, Compensated inclination, Summer diurnal cycles phase-chain chart, Solar radiation, Wall temperature, Gate of Sant'Ubaldo (+14 more)

### Community 3 - "tables.py"
Cohesion: 0.12
Nodes (19): basename(), _cell(), date_cell(), latex_escape(), Module: shmlib.tables Writing the LaTeX table bodies that the study reports…, Format a table into LaTeX rows, one string per row of ``frame``. Parameters…, Write LaTeX rows to a table body file. The rows are joined by the row separator…, Format a table and write it in one call. Convenience wrapper over… (+11 more)

### Community 4 - "apply_report_style"
Cohesion: 0.16
Nodes (20): plot_channel_series(), plot_current_era_series(), plot_segment(), plot_sr_failure(), plot_sr_raw(), plot_twall_raw(), One full-archive plot of a single channel, in that channel's colour. Parameters…, One plot of a current-era channel over the era it exists in. Unlike… (+12 more)

### Community 5 - "couple"
Cohesion: 0.12
Nodes (20): best_operator(), couple(), diurnal_band(), gain_at_lag(), gain_stability(), operator_scan(), _paired_correlation(), A driver put through the two things that can delay a response, in order. They… (+12 more)

### Community 6 - "figures.py"
Cohesion: 0.18
Nodes (13): # TODO: write one row per quantity per source once the numbers above have been…, Study 2 — Proxy Forcing Characterization (README), Usable radiation window (n_sr_ok night-zero correction intersected with sr_suspect's 161-day exclusion), proxy_forcing_report.tex, Module: shmlib.compare How a source behaves on its own, and how two sources…, Module: shmlib.coupling How one series moves with another, and after how long.…, Module: shmlib.figures The multi-source figures a proxy-forcing comparison…, Package: shmlib The library shared by the studies under ``studies/``, and the… (+5 more)

### Community 7 - "de_lib.py"
Cohesion: 0.09
Nodes (21): assemble_wide(), coalesce_target(), coincidence(), condemned_by_month(), diurnal_response(), export_column_doc(), extreme_excursions(), Module: de_lib.py Support library for the data-exploration study… (+13 more)

### Community 8 - "format_spines"
Cohesion: 0.15
Nodes (18): plot_agreement_stability(), plot_certified_window(), plot_diurnal_comparison(), plot_pair_bias_grid(), plot_pair_scatter_grid(), plot_source_scatter(), plot_three_source_series(), What became of every day of the on-structure radiation channel's life. A day-… (+10 more)

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
Cohesion: 0.17
Nodes (8): `figures.plot_diurnal_season_grid`, the channel-by-season diurnal grid., Five hourly days each of summer and winter, on two channels of very different…, A temperature channel and a wind-direction channel, five hourly days each…, The returned figure carries exactly `len(columns) * len(seasons)` axes, laid…, The two seasonal panels of one row must return the same `get_ylim()`, since a…, The season name appears once, capitalised, at the start of the top-row panel…, The row of a channel declared circular is drawn on a 0-360 degree compass axis…, TestPlotDiurnalSeasonGrid

### Community 13 - "adc.py"
Cohesion: 0.27
Nodes (9): _parse(), parse_file(), parse_legacy_file(), Module: shmlib.adc The raw ``.adc`` archive: its file format, the constants of…, Parse one numeric field, accepting either decimal separator. The archive mixes…, Parse the records of one era out of one ``.adc`` file. Shared body of…, Parse the current-era records of one ``.adc`` file. Handles the mixed decimal…, Parse the legacy-era records of one ``.adc`` file. Differs from… (+1 more)

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
Cohesion: 0.17
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

### Community 23 - "proxies.py"
Cohesion: 0.11
Nodes (19): channel_inventory(), harmonise(), join_eras(), load_response(), load_sensor_forcings(), mask_implausible(), Module: shmlib.proxies The external forcings that drive the wall, in every…, Load the on-structure forcings from study 1's archive onto the analysis grid.… (+11 more)

### Community 24 - "test_prediction.py"
Cohesion: 0.05
Nodes (25): Tests for Study 04 prediction helpers. Run directly, with no test runner…, Routes must reflect what information exists for each prediction task., Prediction scores use paired observed and predicted values only., Bootstrap skill must keep parent and child errors timestamp-aligned., A tiny real fit protects the wrapper contract at the library boundary., Notebook orchestration must keep fold boundaries explicit., First differences must not bridge records the model may not learn across., Baselines must never borrow values across segment boundaries. (+17 more)

### Community 25 - "phase_chain"
Cohesion: 0.25
Nodes (8): cycle_extreme(), diurnal_cycle_pair(), phase_chain(), Position of a cycle's extreme, or nothing when the extreme is a plateau. A mean…, Average diurnal cycle of a current-era channel, by season. The current-era…, The four quantities of the thermal chain, on one clock and one scale. The sun…, channel_name(), The channel's name without its unit, for a title or a legend entry. Parameters…

### Community 26 - "solar_elevation"
Cohesion: 0.18
Nodes (12): flag_sr_day_quality(), Name the single state each day of the failure window belongs to. The window is…, The radiation that survived rejection, before the night correction. Both…, Condemn the days on which the radiation channel has no diurnal cycle. A working…, Solar radiation grouped by solar elevation, condemned days against the rest.…, sr_by_elevation(), sr_day_states(), sr_measurements() (+4 more)

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
Nodes (39): _analysis_freq(), _as_series(), availability_route(), backtest_specifications(), baseline_predictions(), contiguous_segments(), execution_folds(), expanding_segment_folds() (+31 more)

### Community 41 - "plot_source_panels"
Cohesion: 0.12
Nodes (16): _clip_note(), plot_source_panels(), The note a clipped axis carries, naming what it leaves outside. Parameters…, The complete record of one source, one panel per channel, on one shared clock.…, circular_resample(), Resample a Series of angles onto a new frequency by the circular mean.…, load_era5(), load_ground_station() (+8 more)

### Community 42 - "test_shmlib_study02.py"
Cohesion: 0.40
Nodes (3): Unit tests for the decisions study 2 relies on, now that they live in…, Season definitions match study 1's, and unnamed months stay unnamed., TestSeasons

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

### Community 52 - "Station Locations Map"
Cohesion: 0.50
Nodes (4): Station 01, Station 02, Station 03, Station Locations Map

### Community 53 - "Data exploration report review notes"
Cohesion: 0.50
Nodes (4): Data exploration report review notes, Academic report tone, Legacy-current instrument changeover, Three-station wall monitoring

### Community 55 - "Diurnal Cycles (st02)"
Cohesion: 0.67
Nodes (3): Diurnal Cycles (st02), Tair Diurnal Cycles (st02), RH Diurnal Cycles (st02)

### Community 56 - "T-Wall Filtered Chart"
Cohesion: 0.67
Nodes (3): T-Wall Filtered Chart, T-Wall Anomalies Chart, T-Wall Diurnal Cycles Chart

### Community 57 - "Study 4 · NeuralProphet inclination prediction"
Cohesion: 0.40
Nodes (4): Contents, Input and model designs, Reproducing, Study 4 · NeuralProphet inclination prediction

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

### Community 90 - "viz.py"
Cohesion: 0.33
Nodes (3): Module: shmlib.viz Figure scaffolding shared by every study: the colour scheme,…, Set the seaborn theme and the canvas scale for the figures that follow.…, set_context()

## Knowledge Gaps
- **66 isolated node(s):** `Input and model designs`, `Contents`, `Reproducing`, `Answer`, `Outcome` (+61 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **66 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TestPlotDiurnalSeasonGrid` connect `TestPlotDiurnalSeasonGrid` to `figures.py`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Why does `TestDrawCycle` connect `TestDrawCycle` to `figures.py`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Why does `TestMeteo` connect `TestMeteo` to `figures.py`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **What connects `Input and model designs`, `Contents`, `Reproducing` to the rest of the system?**
  _66 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `_diurnal` be split into smaller, more focused modules?**
  _Cohesion score 0.050724637681159424 - nodes in this community are weakly interconnected._
- **Should `calibrate_against_sensor` be split into smaller, more focused modules?**
  _Cohesion score 0.08923076923076922 - nodes in this community are weakly interconnected._
- **Should `Departure from the 24-hour rolling median, summer 2026 anomaly window` be split into smaller, more focused modules?**
  _Cohesion score 0.09956709956709957 - nodes in this community are weakly interconnected._