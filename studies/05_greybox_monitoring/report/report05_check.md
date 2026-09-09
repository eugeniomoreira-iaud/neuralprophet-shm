# 1 Introduction

# 2 The record and the three regressor sets

- [ ] **Acronyms:** Ensure that the acronyms for the sources (GS, STR, and ERA5) are explicitly identified at their first appearance in the text.
- [ ] **Table 1:** Transform the table into a stacked bar chart with a double Y-axis (one for raw quantities, another for percentages, since the maximum total slots is the same for all of them). The bars should be placed vertically and grouped by source. Put `inclination` as the first bar in the chart.
- [ ] **Clarity:** The following paragraph is confusing and should be rewritten for clarity: *"Table 1 gives what each set actually holds over the window. The inclination returns an accepted value in 67.7 % of the slots. The on-structure air temperature and humidity reach 71.1 %, with about two per cent of their accepted slots filled across short dropouts; the station’s channels reach 91.7 to 94.7 %, with under one per cent filled; ERA5 covers 99.2 % of the window with nothing to fill. The radiation of the on-structure and station sets is the same column and has the same coverage."*
- [ ] **Compliance Warning:** Ensure that the text surrounding the new chart is updated to remain compliant with the fact that Table 1 has been transformed into a chart.
- [ ] **Data-Quality Report:** Update `docs/data-quality-report-2026-08-10.md` to include information about the gaps coming from the cleaning procedure, not just the information about outages.
- [ ] **Self-Contained Text:** Remove any explicit references to the "data-quality report" in the text (e.g., *"longer than the outages the data-quality report lists"*). Because the quality report is an internal document that won't be accessed by the reader, the text should provide a full, self-contained description of the gaps and outages.
- [ ] **Table 2 (Gap Inventory):** Replace Table 2 with a log-log histogram. Use the exact number of missing 20-minute datapoints to define the bins on the X-axis rather than the broad categories currently in the table. Set both the X-axis (gap duration) and Y-axis (gap count) to a logarithmic scale so both the high frequency of short gaps and the few extremely long gaps are visible. Crucially, ensure the tick labels on both axes display the *real values* (e.g., actual counts and durations) rather than scientific notation or log values, so the reader has an immediate physical reference.
- [x] **Methodological Doubt (Timezone & DST), resolved 2026-09-06:** The 1-hour lag of the on-structure pyranometer against ERA5 is not a logger clock error. Three facts settle it. First, all three time bases are UTC before any comparison: the archive through `site.to_utc` (Italian civil time with daylight saving, measured by Study 01 against solar noon), the ground station converted by its retrieval script, ERA5 natively. Second, the 1-hour figure is the end-of-hour stamp of ERA5's accumulated radiation in the hourly clock-check frame; once `to_native_grid` places the accumulation at its interval midpoint, `sr_str` and `sr_era5` align at 0 min on the 20-minute grid. Third, the same logger gives two different answers: `sr_str` sits at 0 min against ERA5 while `tair_str` runs 40 to 80 min ahead of both ERA5 and the station. A clock error moves every channel by the same amount; what moves one channel and not the other is the sensor, here the air probe in its sun-exposed housing. A minute-resolution displacement scan of every structure channel against the proxies gives per-pair optima from −60 min (inclination against ERA5 radiation) to +100 min (inclination against air temperature), which is response, not clock; the scan now lives in Study 02 for the sensor-versus-proxy pairs and in Study 03 for the inclination pairs. **Action item:** Rewrite the paragraph that raises the doubt as a short time-base statement citing Study 02's harmonisation table (`PF_16_shift_summary`) and the same-logger split. No shift is applied to any structure channel.
- [ ] **Ground-station stamping (notebook):** The regridding cell passes `accumulations=()` for the station and `load_ground_station` receives no stamp offset, so the station's values, which are interval means stamped at the interval end, sit 20 to 40 min late against both ERA5 and the wall's own pyranometer. Add `GROUND_STAMP_OFFSET` to the parameter cell with the value Study 02 settles on (15 min under the end-stamped-mean reading), pass it to `load_ground_station`, document it in the Parameter Tuning Guidance, and rerun from Movement 0. This matters because the "str" regressor set borrows `sr_gs` as its radiation driver.

# 3 Method

## 3.1 D1 · The whole record, 2018 to the archive end

## 3.2 D2 · Three regressor sets through one model specification

## 3.3 D3 · Study 03's operator is applied outside the model; radiation is global horizontal in every set

- [ ] **Radiation delay:** `RADIATION_DELAY_H = 1` and its guidance text claim "the diurnal-band optimum measured in Study 03 for all three radiation sources". Study 03's table `TR_02_coupling_all.csv` gives 1 h for `sr_str` only; `sr_gs` and `sr_era5` sit at 0 h, and the report itself says "zero to one hour, not distinguishable". On the 20-minute grid the inclination lags radiation by about 40 min against ERA5 and the wall pyranometer, whose curves coincide, and by about 20 min against the station before its stamping offset. Replace the hourly constant by the delay Study 03's new native-resolution step reports, declared in minutes and converted to slots in the notebook, one value per source if they differ after the station offset. Correct the guidance text so it quotes Study 03's table faithfully. Rerun from Movement 0; every gain, interval and threshold downstream changes.

## 3.4 D4 · Wall temperature and on-structure radiation, tested as a ladder on the current era

## 3.5 D5 · The trend stays on in the monitoring fit

## 3.6 D6 · Harmonic diagnostics before modelling

## 3.7 D7 · Model A, an additive grey-box without autoregression

## 3.8 D8 · Uncertainty the library's way, calibrated on a rolling window

## 3.9 D9 · Model B, a learned impulse response on the on-structure set

## 3.10 D10 · Three charts, one per damage mechanism and time scale

## 3.11 D11 · Detectability by mechanism, with injections sized by the wall's daily response

## 3.12 D12 · Outages as hypotheses, never as data

## 3.13 D13 · The target is never filled; regressor dust is

## 3.14 D14 · Every native plot type, redrawn in the project's graphical language

# 4 Trend

# 5 Seasonality

## 5.1 The harmonic diagnostic

# 6 Regressors and gains

- [ ] **What `tair_str` measures:** The on-structure air temperature leads free air by 40 to 80 min in both eras (Study 02, `PF_16_shift_summary`), because its probe sits in a sun-exposed housing and heats with radiation; the station and ERA5 air temperatures agree with each other to within one slot. The text must not describe the "str" set's air-temperature gain as the wall's response to air temperature; it is a mixed air-and-radiation response. Either say so where the gains are compared across sets, or move the air temperature of the "str" set to the station column the way radiation already is, and state the choice in D2.

# 7 Impulse response

- [ ] **State the expectation before the figure:** Against radiation the wall responds with a delay of about 40 min, two slots, and against air temperature the wall leads by 1 to 1.5 h, both measured on the 20-minute grid before Model B was fitted. Write these two numbers as the expectation Model B's learned weights are checked against, then compare: a radiation kernel peaking near two slots, and an air-temperature kernel whose weight sits at or before lag zero, confirm that the wall follows the sun and not the air. A radiation kernel peaking at zero would mean the delay was absorbed elsewhere, and the text must say where.
- [ ] **Radiation as a filtered thermal state (idea, 2026-09-07):** The pyranometer's curve has a floor at zero every night; the inclination and the air temperature do not. What the wall feels is not the radiation but the heat stored in the stone, which rises while the sun is up and decays continuously, without a floor, once it sets. In a lumped heat balance that stored state obeys `tau * dx/dt = a * R(t) - x(t)`, whose solution is a causal exponential filter of the radiation, an exponentially weighted moving average with one time constant `tau`; the same `tau` governs heating and cooling because loss to the air is linear in the excess temperature. This is the filter half of Study 03's delay-and-filter operator, which the ladder already applies to the wall temperature with `tau = 4 h`; D3 imposes the delay half only for radiation. On the daily harmonic a 1 h delay and a filter with `tau` of about 1 h give the same phase lag and cannot be told apart, which is why Study 03's diurnal-band scan chose between them by neither; they differ in shape, the delay keeping the floor and the filter drawing the night-time tail. Model B's radiation weights, read as noise in Section 7, are also what a filter with `tau` well beyond twelve hours looks like when truncated at twelve: no peak, a cumulative weight climbing almost linearly across the window. **Action item:** Add a decision D15 to the design document and a sweep to the notebook, `TAU_SWEEP_H` over about 1, 2, 4, 8, 12, 24 and 48 hours applied to the station radiation with the delay set to zero once the filter is on, the filter state reset after every target gap longer than the D13 fill limit and a warm-up of about `3 * tau` flagged. Report per `tau`, in a new table `GM_10b` and one figure beside Model B: held-out MAE with the paired block-bootstrap increment over the delay-only fit, the learned radiation gain beside Study 03's −0.035 mdeg per W m⁻², and the residual's daily-sideband power. Rule stated in advance: the filter counts as an improvement only if the bootstrap bounds on its skill exclude zero. This runs as a diagnostic under D9's own rule, Model A's main line stays on the delay operator and nothing in Sections 4 to 8 is re-run; if the sweep says yes with margin, promotion to the main line is Study 06's job, or the paper's model choice, because it changes every attribution number. Expect a modest MAE gain at best, since the wall probe measures this state directly and the ladder found it buys nothing; the likelier wins are the sign of the on-structure radiation gain, currently wrong, and the annual sidebands of the daily cycle left in the residual, which filtered radiation modulates six-fold across the year and air temperature does not. Cautions: filtered radiation is a smooth afternoon-peaking wave close to air temperature in shape, so the two gains will compete and the held-out skill, not the gain, is the judge. Out of scope for the sweep and a candidate for Study 06: night cooling by longwave loss to a clear sky, which the pyranometer cannot see and ERA5's surface net thermal radiation could supply, and asymmetric heating and cooling constants.

# 8 Uncertainty and validation

# 9 What the wall temperature and the pyranometer buy

# 10 The monitor

# 11 Outages as hypotheses

# 12 Verdict

# 13 Limitations

- [ ] **Undocumented geometry:** Neither the azimuth of the monitored wall face nor the mounting orientation of the on-structure pyranometer is recorded anywhere in the project. The 20-minute phase between the wall pyranometer and horizontal radiation, and the choice to treat the wall's own radiation as the physical forcing with the proxies as stand-ins, both rest on that geometry. State it as a limitation and as the one field measurement that would close it.
- [ ] **Station stamping is inferred, not declared:** The ground-station offset applied at input is inferred from the displacement scan and from the 30-minute cadence of the export; the vendor documents no stamping convention. Say so, and give the residual after correction as the bound on the remaining time-base uncertainty.

# 14 Run metadata
