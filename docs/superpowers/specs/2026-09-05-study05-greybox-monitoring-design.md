# Study 05 · Design — Grey-box expectation and monitoring of the station 02 inclination

Date: 2026-09-05
Status: approved 2026-09-05
Folder: `studies/05_greybox_monitoring/`, artefact prefix `GM_`
Supersedes the approach of `studies/04_neuralprophet_inclination_prediction/`; Study 04 stays in place as the record of what was learned.

---

## 0 · Working rules for every step of this study

- **Caveman ultra, always.** Every session and every spawned subagent runs under `/caveman:caveman ultra` from the first message: chat, checkpoints, commit discussion, code review. The only exception is prose that reaches a produced document — the study report, the README, this spec, docstrings, comments and commit messages — which is written in full academic or technical prose.
- **Delegate down.** The orchestrator (Fable or Opus) plans, decides, reviews checkpoints and writes the report. Everything else that a lower-tier model can do goes to Sonnet subagents spawned with the Agent tool, each prompt starting with "respond in caveman ultra mode": folder scaffolding, `shmlib` functions under written instruction, tests, notebook cells, table and figure export, LaTeX plumbing, graph updates, bulk reads. The report prose is never delegated.
- **Delivery sequence.** Study folder, then this spec, then the user's approval, then an implementation plan written with the writing-plans skill, then implementation phase by phase with a checkpoint approved before each next phase.

---

## 1 · The question this study answers

A reading arrives every twenty minutes: inclination, air temperature, relative humidity, and, since February 2025, wall temperature and solar radiation. The operator's question is the one Study 04 posed and did not fully answer:

> **Given the environment measured at this instant, is this inclination the one the wall was expected to show — and if not, which kind of departure is it?**

Study 04 answered it on the post-outage window (June 2023 onward) and found its own weak points honestly: an annual term and a trend that never converged on 3.2 broken cycles; a conformal interval covering 67.7 % against a nominal 90 %; a residual with lag-1 autocorrelation 0.995 that forced the control limit to 14.75σ; and a detector that consequently sees only a doubling of the daily swing. Its phase-change injection was also scaled by the leftover daily seasonal term (2.1 mdeg) rather than by the wall's actual daily response (median amplitude 10.9 mdeg), so that blind spot was overstated by roughly five times.

This study keeps what Study 04 established and changes what it got wrong. It asks five questions:

1. **What is the record made of, over eight years?** Trend, annual cycle, daily cycle, and the response to each measured driver, with the share of variance each carries and the gain each learns, on three regressor sets.
2. **Does the wall answer its drivers with a delay or an inertia the 20-minute grid can resolve?** Learned impulse responses on the on-structure set, confronted with Study 03's delay-and-time-constant scan.
3. **Is this reading the expected one?** A rolling expectation with a conformal interval whose coverage is measured as a function of refit staleness.
4. **What departure does each chart catch, at a stated false-alarm budget?** Three charts, three damage mechanisms, injections sized by the wall's measured daily response.
5. **What happened across each outage?** The expected level at resumption, from proxies that kept recording, against the level observed.

Forecasting is out of scope. Study 04 answered it: skill comes from the response's own memory and the environment adds nothing distinguishable from zero beyond six hours.

---

## 2 · What is already established, and therefore not re-litigated

### 2.1 From Study 01 — the record

| Fact | Value | Source |
|---|---|---|
| Analysis product | `inc_comp_cleaned` — compensated, anchored once across the whole record, spike-cleaned | `studies/01_data_exploration/de_lib.py` |
| Interpolation flag | `inc_spike` — `True` marks a value the cleaning invented | same |
| Native cadence and extent | 20 minutes, 2018-07-26 to the archive end | archive manifest |
| Compensation | Manufacturer's linear correction, 5 mdeg/°C on the unit's own air temperature, one reference for both eras | Study 01 report §Thermal compensation |
| The two instruments at station 02 are continuous across 2025-02-21 | +1.58 mdeg on 30-day window means | `docs/raw-data-format.md` §3.3 |
| The summer 2026 excursions are an instrument or power artefact, not a structural or environmental event | inclination crosses 5σ on 21 % of samples, air temperature and humidity on under 1 % | Study 01 report §The summer 2026 anomaly across the channels |

**The compensation is taken as given.** It is the manufacturer's procedure and the user's numerical model of the wall has validated it. Every gain this study learns is a post-compensation wall response and is labelled as such. No raw-channel fit is made.

### 2.2 From Study 02 — the external sources

| Quantity | Source | Bias | MAE | r | Phase lag | Coverage |
|---|---|---|---|---|---|---|
| Air temperature | town station | +0.37 °C | 1.30 | 0.983 | 1.8 h | 96.6 % on its half-hour grid, from 2018-10-17, one 10-day outage (November 2021) |
| Air temperature | ERA5 | −0.56 °C | 1.70 | 0.970 | 2.3 h | 99.5 %, hourly, from 2018-01-01 |
| Solar radiation (certified window) | town station | −88 W/m² | 118 | 0.925 | 0.6 h | as above |
| Solar radiation | ERA5 | −56 W/m² | 125 | 0.892 | 0.9 h | as above |

Source: `studies/02_proxy_forcing_characterization/outputs/PF_10_tair_agreement.csv` and `PF_11_sr_agreement.csv`, on-structure channel as reference, hourly UTC. The station is the better proxy; ERA5 is the better transferable regressor. ERA5 radiation is `surface_solar_radiation`, global horizontal, the same quantity the station pyranometer measures. The file's `skin_temperature` channel is not used: an ERA5 cell of several kilometres of mixed land cover says nothing about the temperature of a stone face (D3).

### 2.3 From Study 03 — the couplings

| Driver | Delay | Gain on `inc_comp_cleaned` (diurnal band) | Note |
|---|---|---|---|
| On-structure air temperature | 0 h | −2.79 mdeg/°C, r = −0.957 | Strongest driver; gain stable over 35 months (median −2.48) |
| Station air temperature | 0 h | −2.23 mdeg/°C, r = −0.807 | |
| ERA5 air temperature | 0 h | −2.04 mdeg/°C, r = −0.787 | |
| Station radiation | 0–1 h | −0.035 mdeg per W/m², r = −0.879 | Strongest external driver |
| ERA5 radiation | 0 h | −0.026 mdeg per W/m², r = −0.873 | |
| On-structure radiation | 1 h | −0.026 mdeg per W/m², r = −0.867 | Current era, certified days only |
| Relative humidity | 0 h | positive, r ≈ +0.7 to +0.8 | Inverse of the temperature cycle, not a mechanism |
| Wall temperature | −2 h (leads), τ = 4 h on the levels | −1.75 mdeg/°C diurnal | Internal state variable; lead must be clamped to zero before predictive use |

Binding consequences: external drivers enter instantaneously or with the diurnal-band delay; no level-band time constant is carried; the battery control is a conservative floor, not a null.

### 2.4 From Study 04 — what to keep and what to change

Kept: no autoregression in the decomposition (it absorbs the diurnal structure); changepoints on covered time; the level is decomposed and monitored, never scored as a forecast; contemporaneous regressors in the expectation are not leakage; no value of the target is imputed; NeuralProphet's own gap filling is disabled (`impute_missing=False`, `drop_missing=False`).

Changed, with the reason in §3: the window (D1); the trend in the monitoring fit (D5); the seasonal orders and condition weights, now measured (D6); the calibration window of the interval (D8); the single residual chart (D10); the phase-injection scale (D11).

### 2.5 Measured for this design (2026-09-05)

The diurnal band of `inc_comp_cleaned` (series minus its centred 24-hour rolling mean, Study 03's definition) over 2023-06-21 to 2025-09-01, on 589 days with at least 60 of 72 slots:

| Quantity | Value |
|---|---|
| Median daily peak-to-peak of the inclination | 21.7 mdeg (q25 13.8, q75 34.2; summer median 39.4) |
| Median daily peak-to-peak of the on-structure air temperature | 8.3 °C (summer 12.5) |
| Residual amplitude a 2-hour timing shift of the full daily response implies | 5.6 mdeg (Study 04 injected 1.07) |
| Residual amplitude a 1-hour shift implies | 2.8 mdeg |

Station record on its native half-hour grid: air temperature 96.6 % coverage, 1,319 gaps of which 915 are one hour or shorter, one outage longer than seven days (2021-11-21 to 2021-11-30).

---

## 3 · Design decisions, and the evidence for each

### D1 · The whole station 02 record, 2018-07-26 to the archive end

Study 04's annual term was not identifiable on 3.2 cycles with a 103-day hole, and its trend disagreed with itself in sign across thirds (+38.6, +4.8, +45.5 mdeg/yr against −2.77 for the whole window). The whole record holds five usable annual cycles (2019, 2020 with 11 days missing, 2021, 2024, 2025), compensated and anchored once by Study 01. The instrument change of 2025-02-21 is not modelled: the recorded channel is continuous across it to +1.58 mdeg and the compensated gain is stable across both eras.

### D2 · Three regressor sets through one model specification

One specification, three regressor roles, three sources. The on-structure set gives the site result; the station set the best external proxy; the ERA5 set the transferable arm. The paper narrates two and tables all three.

| Role | On-structure set (`str`) | Station set (`gs`) | ERA5 set (`era5`) |
|---|---|---|---|
| Air temperature | `tair` joined across eras | station `Temp` | ERA5 `temperature` |
| Relative humidity | `rh` joined across eras | station `Umid` | ERA5 `relative_humidity` |
| Solar radiation | station `Rad.Sol.`, labelled as borrowed | station `Rad.Sol.` | ERA5 `surface_solar_radiation` |

The on-structure set borrows the station's radiation because the wall's own pyranometer does not exist before 2025-02-21 and Study 01 condemned 161 of its days after that. The borrowing is stated in every table that reports the set. Wall temperature and on-structure radiation stay out of the main line (17.6 % and 18.8 % coverage) and are tested on their own window in D4.

### D3 · Study 03's operator is applied outside the model; radiation is global horizontal in every set

Air temperature enters instantaneous. Radiation enters delayed by one hour, the diurnal-band optimum for all three sources. Both go through `shmlib.coupling.thermal_operator`, the one function that applies a delay and a one-pole filter. A driver is never given a time constant from the level band; Study 03 showed those pin against the grid.

**Which radiation variable.** Global horizontal irradiance in every set: the station's pyranometer and ERA5's `surface_solar_radiation` measure the same quantity, so the radiation gain is comparable across sets, and it is the variable Study 03 screened (r = −0.87 to −0.88 in all three sources). No other radiation variable is used.

**Skin temperature is dropped.** ERA5's `skin_temperature` is the radiative temperature of a grid cell several kilometres across, averaged over fields, roads and roofs; it carries no information about one stone face that the air temperature and radiation do not already carry, and its resolution cannot be argued in a paper. It is not loaded.

### D4 · Wall temperature and on-structure radiation are tested on the current era, as a ladder

The probe and the pyranometer were installed to find out whether they carry information the other drivers do not. On the window 2025-02-21 onward — the two `twall`-valid blocks (143 and 54 days) and the certified radiation days — the same specification is refitted rung by rung:

1. the on-structure set as in the main line;
2. on-structure radiation replacing the station's, on the same certified days;
3. plus `twall` at zero delay with τ = 4 h, the deployable form;
4. `twall` at its measured lead of −2 h, as a diagnostic row only, never for the expectation.

Each rung reports held-out MAE with a paired block-bootstrap increment, variance share, learned gain beside Study 03's −1.75 mdeg/°C and −0.026 mdeg per W/m², conformal coverage and width, and residual lag-1 autocorrelation. The verdict is one sentence per channel on whether a deployment should keep it. This is the input to the manuscript's sensor recommendation.

### D5 · The trend stays on in the monitoring fit; drift is found by the slow chart

Study 04 dropped the trend from its monitoring fit so that drift would "stay in the residual where a monitor can see it". The consequence was a residual with lag-1 autocorrelation 0.995 at 20 minutes and 0.974 at 24 hours, a control limit swept out to 14.75σ to meet the false-alarm budget, a 288-day alarm episode that said only "the wall has moved since calibration", and a detector blind to every slow mechanism. Leaving drift in the residual is what made drift undetectable.

Here the expectation comes from a rolling refit every 30 days with the trend on. Over 30 days the trend's extrapolation error is a fraction of a millidegree at the measured rates, and the residual it leaves is stationary enough to chart. Drift is detected by the slow chart of D10 on daily means, and by the sequence of refit trend slopes. The attribution fit is a single fit over the full training window, for component shares, gains and the parameter figures.

### D6 · Harmonic diagnostics before modelling: the seasonal orders and the condition weights are measured, not guessed

NeuralProphet's seasonal terms are Fourier series whose orders are the user's choice, and the conditional daily term of D7 needs a weight curve. Both are fixed by measurement in Phase 1b, before any model is fitted, on two series: the target itself, which gives the physics picture, and the residual of a plain least-squares regression of the target on the regressors of the on-structure set, which is what the seasonal terms will actually have to explain.

1. **Spectral scan.** `prediction.period_scan` (Lomb–Scargle, which takes the gaps as they are and needs no filling) run from 12 hours to 900 days on both series. The peaks it certifies decide the orders: a semi-annual peak means `YEARLY_ORDER ≥ 2`, a 12-hour peak means `DAILY_ORDER ≥ 2`. The orders are then confirmed on held-out folds in Phase 2, never raised beyond what the scan supports.
2. **The daily cycle through the year.** For every day with at least 60 of 72 slots, a 24-hour harmonic (with its 12-hour companion) is fitted to the diurnal band of each series, giving a daily amplitude and phase over eight years (`monitoring.daily_harmonic`, the same function the daily chart of D10 uses). Each is then fitted against day of year with an annual Fourier series of order 1 or 2 (`coupling.annual_modulation`), the order chosen on held-out years. The amplitude fit, normalised to 0..1, is the weight curve `w(doy)` that D7 consumes (`prediction.seasonal_weights`); its phase fit says whether the timing of the daily cycle moves through the year, which is what justifies two weighted shapes rather than one scalar envelope.
3. **Rank of the daily-by-annual surface.** The diurnal band binned by day of year and hour of day is a surface; its singular value decomposition (`coupling.cycle_surface_rank`) says how many weighted daily shapes it takes to reproduce it. Two components carrying most of the variance license D7's two conditions; a third would mean the design needs three weights, and that is reported before Model A is fitted rather than discovered after.

Table `GM_04` and figure `GM_F02` carry the results: the certified periods per series, the annual fits of daily amplitude and phase with their held-out order, and the surface's singular values. Every order and weight used afterwards traces to this table.

### D7 · Model A — additive grey-box without autoregression

```python
NeuralProphet(
    growth='linear',
    n_changepoints=N_CHANGEPOINTS,        # placed on covered time: prediction.covered_changepoints
    changepoints_range=0.95,
    trend_reg=TREND_REG,                  # swept in Phase 2; tutorial 02 and the sub-daily guide
    yearly_seasonality=YEARLY_ORDER,      # Fourier order, set by the harmonic diagnostic of D6
    daily_seasonality=DAILY_ORDER,        # likewise
    weekly_seasonality=False,             # nature does not follow the week
    n_lags=0,
    quantiles=[0.05, 0.95],
    impute_missing=False, drop_missing=False,
    learning_rate=LEARNING_RATE, epochs=EPOCHS,
)
set_random_seed(SEED)
m.add_future_regressor(role) for each role in the set
```

Every value in capitals is declared in the notebook's parameter cell with its Parameter Tuning Guidance, and passed in as an argument. `YEARLY_ORDER`, `DAILY_ORDER` and the coefficients of the weight curve are declared there too, with the values Phase 1b measured and a pointer to `GM_04`, so that the notebook states what it was run on without a reader having to open the diagnostic. Minimum training history is two full annual cycles, which the whole record allows.

**Conditional daily seasonality, smoothly weighted.** The shape of the daily cycle changes with day length and sun elevation, which move continuously through the year rather than in seasonal boxes. Following NeuralProphet's conditional-seasonality guide, which accepts floats in 0..1 as conditions, the daily cycle is fitted as two series, one for a midsummer form and one for a midwinter form, blended by two weights that are functions of the day of year alone:

```
summer_w = w(doy)          # the annual modulation measured in D6, scaled to 0..1
winter_w = 1 − summer_w
m.add_seasonality(name='daily_summer', period=1, fourier_order=DAILY_ORDER, condition_name='summer_w')
m.add_seasonality(name='daily_winter', period=1, fourier_order=DAILY_ORDER, condition_name='winter_w')
```

The weight curve `w(doy)` is not assumed: it is the low-order annual Fourier fit of the measured daily amplitude that D6 produces, normalised to 0..1. A pure cosine with its maximum near mid-July, `½(1 − cos(2π(doy − 15)/365))`, is the fallback if the measured curve is not better on held-out folds. The two weights sum to one at every timestamp, so no row is left without a daily term, and there is no boundary day at which the shape steps. The result is an annual modulation of the daily cycle's amplitude and phase, which is the interaction the physics implies, at the cost of one extra Fourier series. The report draws the fitted daily cycle at the two solstices and the two equinoxes, so the four-season picture appears without four parameter sets. Four boolean conditions on Study 01's meteorological seasons (DJF, MAM, JJA, SON) are the stated fallback if the smooth weights fail to fit.

The daily term is expected to be small either way, since air temperature carries most of the diurnal variance (0.03 % in Study 04). The conditional version is therefore tested against the plain daily term and kept only if it improves held-out MAE or carries more than 1 % of variance; the comparison is reported either way.

### D8 · Uncertainty the NeuralProphet way, calibrated on a rolling window

Quantile regression inside the model at 0.05 and 0.95, then split conformal prediction through `m.conformal_predict(df, calibration_df, alpha=0.10, method='cqr')`. Study 04 calibrated once, on a stretch that ended eleven months before the record it was then asked to cover, and got 67.7 % coverage. Here each refit calibrates on the most recent six months of its own out-of-sample residuals, so the interval is never calibrated on a quieter era than the one it covers. Coverage, width, Winkler score and pinball loss are reported per set and per days since refit.

### D9 · Model B — learned impulse response on the on-structure set

The same specification with `n_lags=0` on the target and two drivers, air temperature and the station radiation, entering through `add_lagged_regressor(['tair', 'sr_gs'], n_lags=36, regularization=LAGGED_REG)` at 20 minutes: twelve hours of history per driver, at the resolution Study 03 could not reach. Humidity stays contemporaneous. The learned weight per lag is the impulse response; from it an effective delay (first moment) and a time constant (one-pole fit to the cumulative response) are read and set beside Study 03's operator cell for each driver. Agreement validates the operator imposed in Model A. Disagreement is reported as a finding; Model A is not re-tuned to hide it. Phase 0 verifies that NeuralProphet 0.8.0 accepts lagged regressors without target autoregression; if not, Model B runs hourly with `n_lags=12` and the sub-hour claim is dropped.

### D10 · Three charts, one per damage mechanism and time scale

Reference statistics come from a fixed window, 2019-01-01 to 2021-12-31: three complete years, no outage longer than eleven days, before the 2022 outages and before the instrument change. The monitored record runs from 2022-01-01. All three charts run on Model A's rolling residual for each set; the on-structure set is narrated, the others tabled.

| Chart | Statistic | Cadence | Mechanism | Budget |
|---|---|---|---|---|
| Fast | AR(1)-prewhitened innovations `e(t) = r(t) − φ·r(t−1)`, φ from the reference window; EWMA and CUSUM with the joint alarm | 20 min | Steps, spikes, instrument faults | 90 watched days per false alarm |
| Daily | Amplitude and phase of a 24-hour harmonic fitted to each day's residual (days with at least 60 of 72 slots); EWMA and CUSUM on each | daily | Amplitude growth (lost composite action between the leaves); phase change (changed thermal path) | 90 days |
| Slow | Daily-mean residual, CUSUM with a small reference value; beside it the sequence of refit trend slopes | daily | Drift (mortar creep, thermal ratcheting, settlement) | 365 days |

Budgets are fixed before any sweep and stated in the parameter cell. Achieved run lengths are reported with the number of episodes they rest on. Every fast alarm is cross-checked against the air-temperature, humidity and battery channels in the same slot — the test Study 01 ran by hand on summer 2026 — so the episode table carries an attribution column: environment, instrument, or unattributed. The summer 2026 artefact is expected to be caught and attributed to the instrument; that demonstrates the attribution step and is not a sensitivity claim.

### D11 · Detectability by mechanism, with injections sized by the wall's daily response

The three injection shapes of Study 04 (amplitude growth, phase change, drift) are kept, with two corrections. The phase injection is sized by the wall's fitted daily response on the injection date — the air-temperature component plus the daily seasonal term — not by the leftover daily term alone, and that size is checked against the measured daily amplitude of D6 on the same date; on the numbers of §2.5 a two-hour shift injects about 5.6 mdeg, not 1.07. Each mechanism is scored on the chart built for it, with the other two charts reported as well. Injections are placed at several dates across seasons so the detectability field is monotone rather than an artefact of one injection point. An alarm counts only where the uncontaminated record is silent within the response window.

### D12 · Outages as hypotheses, never as data

For each whole-day outage of seven days or more (2020: 11 d; 2022: 112 d and 271 d; 2024: 40 d and 42 d; 2025: 17 d; 2026: 103 d), the model refitted on data up to the outage predicts the level at resumption from the station and ERA5 sets, which kept recording. The observed mean over the first seven days after resumption, with Study 01's restart transient excluded, is compared with the expected level and its conformal interval. The result is a level shift with an interval and a verdict per outage and per proxy set. Nothing is written into the gap; an imputed value may serve as history or as hypothesis, never as evidence.

### D13 · Gaps: the target is never filled, regressor dust is

Rows with a missing target are left out of every fit. Regressor gaps of at most two hours are filled by linear interpolation and flagged in a provenance column, because a regressor is a measured, smooth driver and not the thing being judged; longer regressor gaps leave the row out of the fit for that set only. Proxies are brought to the 20-minute grid by linear interpolation of the hourly and half-hourly values, with ERA5 radiation, an accumulation over the preceding hour, centred on the half-hour before interpolation. Each source's clock is checked with `shmlib.quality.clock_check` before anything is joined, as Study 02 did.

### D14 · Every NeuralProphet plot type, redrawn in the project's graphical language

The native plots (`plot`, `plot_components`, `plot_parameters`, `conformal_plot`, `plot_latest_forecast`, the fit-metrics curve) run in the notebook as diagnostics with `set_plotting_backend("plotly-static")`. The report shows the same content redrawn by `shmlib.figures` under the binding rules of `instructions-pipeline.md`: Okabe–Ito channel colours, Cividis for scalars, legends below the axes, span highlights black at 5 %, PNG and SVG through `viz.finish`. Each redraw reads its data through NeuralProphet's public methods — `predict(decompose=True)`, `predict_trend`, `predict_seasonal_components`, the lagged-regressor weights, the conformal output columns — through one extractor per plot type in `shmlib.prediction`, and each extractor has a test asserting it reproduces the numbers the native plot draws.

---

## 4 · Metrics — what is measured, and why that metric

### 4.1 Decomposition quality

| Metric | Why it is here |
|---|---|
| Variance share and peak-to-peak per component | What the record is made of — the headline claim |
| Learned gain per driver beside Study 03's | An independent method recovering an independently measured constant is the strongest validation available |
| Component stability across NeuralProphet's chronological folds | A gain or a yearly amplitude that changes sign between folds is not a finding |
| Residual whiteness (Ljung–Box at 1, 72, 216 lags) and `period_scan` | Structure left in the residual means a component is missing, and the monitor would alarm on model error |
| Certified periods, held-out order of the annual modulation, singular values of the daily-by-annual surface (D6) | What the seasonal terms must contain, measured before they are fitted; the evidence behind every Fourier order and weight |

### 4.2 Expectation and interval

| Metric | Why it is here |
|---|---|
| MAE, RMSE, bias in mdeg; MASE against a training-only naive scale | Point accuracy against the 1.3 mdeg noise floor; bias exposes a stale fit |
| PICP of the nominal 90 % interval, per days since refit | Whether the interval means what it claims, and how fast it goes stale |
| Median width, pinball loss at q05 and q95, Winkler score | Coverage can be bought by widening; these price it |

### 4.3 Impulse response

| Metric | Why it is here |
|---|---|
| Weight per lag, cumulative response | The learned operator, drawn |
| Effective delay (first moment) and time constant (one-pole fit) | The two numbers Study 03 measured, read from the learned response |

### 4.4 Monitoring

| Metric | Why it is here |
|---|---|
| Achieved run length per chart, with episode count | The false-alarm rate, stated first; every detection figure is quoted at it |
| Detection delay per mechanism and chart | How long the operator waits |
| Smallest detected departure per mechanism, on its own chart | The headline operational number, stated three times |
| Attribution of each fast alarm | Environment, instrument, or unattributed; a monitor that cannot attribute reports every fault as damage |

### 4.5 Outage bridges

| Metric | Why it is here |
|---|---|
| Expected level at resumption, its interval, the observed level, the shift | Whether the outage hid a movement, per outage and per proxy set |

---

## 5 · Architecture and the `shmlib` contract

Every function lives in `studies/shmlib/`. Nothing is written into the study folder. The notebook declares every path, parameter and choice in its parameter cell and passes them as arguments.

### 5.1 Reused unchanged

`proxies.load_response`, `load_sensor_forcings`, `load_era5`, `load_ground_station`, `join_eras`, `harmonise`; `quality.clock_check`; `coupling.couple`, `thermal_operator`, `thermal_lag_filter`; `prediction.gap_inventory`, `covered_changepoints`, `decompose_components`, `component_variance_shares`, `residual_diagnostics`, `score_predictions`, `conformal_interval`, `paired_mae_skill`; `monitoring.reference_stats`, `ewma_chart`, `cusum_chart`, `joint_alarm`, `alarm_episodes`, `average_run_length`, `inject_anomaly`; `figures.plot_decomposition_stack`, `plot_prediction_band`, `plot_control_chart`, `plot_detectability`, `plot_gap_anatomy`; `viz.*`; `tables.write_table`.

### 5.2 Adapted — additive only; every existing caller keeps its behaviour, and Study 03's and Study 04's tests must pass unchanged

| Function | Change |
|---|---|
| `prediction.neuralprophet_backtest`, `neuralprophet_predict`, `rolling_nowcast` | New optional parameters `lagged_regressors`, `lagged_n_lags`, `lagged_regularization`, `trend_reg`, `yearly_order`, `daily_order`, `conditional_seasonality`; defaults reproduce today's behaviour |
| `prediction.rolling_nowcast` | Optional rolling conformal calibration through `conformal_predict` |
| `monitoring.detectability_curve` | Optional `statistic` argument selecting the fast, daily-amplitude, daily-phase or slow chart |
| `prediction.period_scan` | Called with `min_days=0.5`; the range is already an argument, so no change unless its grid assumes whole days |

### 5.3 New in `shmlib`

```
proxies.to_native_grid(frame, freq='20min', accumulations=('sr',))
proxies.fill_short_gaps(frame, columns, max_gap='2h', flag=True)
prediction.trend_parameters(model, frame)               # trend curve and changepoint deltas
prediction.seasonal_parameters(model, period, conditions=None)
prediction.regressor_gains(model, components, regressors)
prediction.lagged_regressor_weights(model)
prediction.impulse_response_summary(weights, dt_hours)
prediction.outage_bridge(model_factory, frame, outages, regressors, settle_days=1, window_days=7)
monitoring.prewhiten(residuals, phi=None, reference=None)
monitoring.daily_harmonic(series, min_slots=60)
coupling.annual_modulation(daily_series, harmonics=(1, 2), holdout='year')
coupling.cycle_surface_rank(series, doy_bins=52)
prediction.seasonal_weights(index, modulation)
monitoring.channel_coincidence(alarm, channels, scale_window)
figures.plot_harmonic_diagnostics, plot_fit_metrics, plot_trend_parameters, plot_seasonal_parameters, plot_regressor_gains,
figures.plot_impulse_response, plot_daily_harmonic_chart, plot_outage_bridge
```

Each new function carries a NumPy-style docstring and a unit test written before the implementation. Each extractor's test asserts that it reproduces the numbers NeuralProphet's own plot draws for the same fitted model.

### 5.4 Figure conventions

Binding, from `instructions-pipeline.md`: Okabe–Ito for categories, Cividis for scalars, one fixed colour per measured channel (inclination `#0072B2`, air temperature `#E69F00`, relative humidity `#009E73`, wall temperature `#DAA520`, solar radiation `#CC79A7`), Vermilion `#D55E00` reserved for annotations and event markers, span highlights black at 5 % opacity, legends below the axes and never inside them, PNG and SVG both written by `viz.finish`.

---

## 6 · Artefacts

Tables in `outputs/`, LaTeX bodies alongside:

| File | Content |
|---|---|
| `GM_01_window_coverage.csv` | Coverage per role per set over the whole window |
| `GM_02_gap_inventory.csv` | Every target gap over eight years, classified by duration |
| `GM_03_clock_check.csv` | Clock offset per source |
| `GM_04_harmonic_diagnostics.csv` | Certified periods per series, annual fits of daily amplitude and phase with held-out order, singular values of the daily-by-annual surface |
| `GM_05_component_shares.csv` | Variance share and peak-to-peak per component, per set |
| `GM_06_learned_gains.csv` | Learned gain per driver beside Study 03's, per set |
| `GM_07_component_stability.csv` | Gain, yearly amplitude, trend slope per fold, per set |
| `GM_08_residual_diagnostics.csv` | Ljung–Box, autocorrelation, scale, period scan, per set |
| `GM_09_nowcast_metrics.csv` | Expectation and interval metrics, per set and per days since refit |
| `GM_10_impulse_response.csv` | Effective delay and time constant per driver beside Study 03's operator |
| `GM_11_alarm_episodes.csv` | Episodes per chart with attribution |
| `GM_12_run_lengths.csv` | Achieved run length per chart with episode count |
| `GM_13_detectability.csv` | Smallest detected departure by mechanism, chart and persistence |
| `GM_14_outage_bridges.csv` | Expected, observed, shift, interval and verdict per outage and proxy set |
| `GM_15_run_metadata.csv` | Every parameter, seed and library version |
| `GM_16_current_era_ladder.csv` | What `twall` and the pyranometer buy, rung by rung |

Figures, each written as PNG and SVG:

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

---

## 7 · Report structure

Study 03's shape — every decision explained where it is made, before the result that depends on it — ordered as the NeuralProphet tutorials are. Target length 16 to 20 pages.

1. **Introduction** — the operational question; what is inherited from Studies 01 to 04 and not re-litigated.
2. **The record and the three regressor sets** — window, coverage, clock check, `GM_F01`.
3. **Method** — one decision per subsection, D1 to D14.
4. **Trend** — changepoints on covered time, rate changes, `GM_F04`.
5. **Seasonality** — the harmonic diagnostic that set the orders and the weights (`GM_F02`), the yearly and daily terms, the conditional-seasonality test, `GM_F05`.
6. **Regressors and gains** — shares, gains confronted with Study 03, per set, `GM_F06`.
7. **Impulse response** — the learned operator against the imposed one, `GM_F07`.
8. **Uncertainty and validation** — folds, component stability, rolling conformal coverage by staleness, `GM_F03`, `GM_F08`.
9. **What the wall temperature and the pyranometer buy** — the current-era ladder, `GM_F14`.
10. **The monitor** — reference window, prewhitening, the three charts, attribution, detectability, `GM_F09` to `GM_F11`, `GM_F13`.
11. **Outages as hypotheses** — `GM_F12`.
12. **Verdict** — one paragraph per question, each with its qualification.
13. **Limitations** — compensation not independent of the fitted term; borrowed radiation in the on-structure set; a fixed reference window three years before the instrument change; one sensor, one site.
14. **Run metadata.**

Every number in the prose traces to a named `GM_` artefact. A folder-honesty test, on the model of Study 04's `test_folder_honesty.py`, guards the folder against any claim without an artefact behind it.

---

## 8 · Phases and checkpoints

Orchestrator (O) plans, reviews and writes prose; Sonnet subagents (S) implement under written instruction. Each checkpoint is shown to the user and approved before the next phase starts.

### Phase 0 · Smoke tests and housekeeping — S, O reviews
Confirm on NeuralProphet 0.8.0: lagged regressors with `n_lags=0`; the `conformal_predict` signature and output columns; `predict_trend` and `predict_seasonal_components`; the matplotlib backend. Run `graphify --update` on `studies/`, since the graph predates Study 04's work. Reconcile Study 02's status line (index says "In progress", README says "skeleton", report prose is prospective) so Study 05 cites it correctly.
> **Checkpoint 0:** each capability confirmed or its fallback chosen; graph current.

### Phase 1 · Data and regressor sets — O plans, S implements
Loaders, clock check, upsampling, regressor-dust filling. `GM_01` to `GM_03`, `GM_F01`.
> **Checkpoint 1:** coverage per set; clock check clean.

### Phase 1b · Harmonic diagnostics — S runs, O interprets
Spectral scan of the target and of the post-regressor residual; daily amplitude and phase over eight years with their annual fits; rank of the daily-by-annual surface. `GM_04`, `GM_F02`.
> **Checkpoint 1b:** `YEARLY_ORDER`, `DAILY_ORDER` and the weight curve fixed from measurement, each traced to `GM_04`; the number of weighted daily shapes the surface needs stated.

### Phase 2 · Model A attribution on three sets — O
Shares, gains, trend and seasonal parameters, the conditional-seasonality test, fold stability, residual diagnostics. `GM_05` to `GM_08`, `GM_F04` to `GM_F06`.
> **Checkpoint 2:** the on-structure air-temperature gain agrees with Study 03's within its interval; residual diagnostics named. **If the gain disagrees, the study stops here and the disagreement becomes the finding.**

### Phase 2b · Current-era ladder — S runs, O interprets
`GM_16`, `GM_F14`.
> **Checkpoint 2b:** a verdict per channel.

### Phase 3 · Expectation and interval — S runs, O reviews
Rolling refit with rolling CQR calibration; metrics by staleness. `GM_09`, `GM_F03`, `GM_F08`.
> **Checkpoint 3:** coverage within a stated tolerance of 90 %, or the shortfall explained by staleness.

### Phase 4 · Model B impulse response — S runs, O interprets
`GM_10`, `GM_F07`.
> **Checkpoint 4:** learned delay and time constant beside Study 03's operator, agreement or finding stated.

### Phase 5 · The monitor — O designs, S runs the sweeps
Reference statistics, prewhitening, the three charts, attribution, detectability with re-sized injections across seasons. `GM_11` to `GM_13`, `GM_F09` to `GM_F11`, `GM_F13`.
> **Checkpoint 5:** achieved run length per chart; smallest departure found per mechanism on its own chart; blind spots named.

### Phase 6 · Outage bridges — S
`GM_14`, `GM_F12`.
> **Checkpoint 6:** a verdict per outage and proxy set.

### Phase 7 · The report — O only, never delegated
Written to §7's structure, from the artefacts, in full academic prose.
> **Checkpoint 7:** every number traces to a named file; the study `README.md` matches the folder; the report builds twice cleanly; the folder-honesty test passes.

---

## 9 · Risks and kill criteria

| Risk | Detection | Response |
|---|---|---|
| Lagged regressors need `n_lags > 0` in 0.8.0 | Phase 0 | Model B on the hourly grid; sub-hour claim dropped |
| The daily-by-annual surface needs more than two weighted shapes, or the measured weight curve is no better than the cosine | Checkpoint 1b | Three weights, or the plain daily term, or the cosine, as the measurement says; stated before Model A is fitted |
| Learned gain disagrees with Study 03 | Checkpoint 2 | Stop; the disagreement is the result, not a bug to tune away |
| Rolling refits at 20 minutes over eight years too slow | Phase 3 | Refit monthly on the last N years rather than all history; state N |
| Prewhitened residual still autocorrelated | Phase 5 | AR(2), or chart hourly innovations; state which |
| Daily harmonic fit unstable on gappy days | Phase 5 | Raise the minimum slot count; report the days lost |
| Reference window not in control | Phase 5 | Shift the window; never widen the limit to compensate |
| Detector alarms constantly on one chart | Phase 5 | Re-tune to the fixed budget; never report a chart without its run length |

---

## 10 · Decisions taken

Approved in conversation and in written review on 2026-09-05.

1. **Scope** — a new Study 05; Studies 01 to 03 untouched; Study 04 kept as the record of what was learned.
2. **Window** — the whole station 02 record, 2018-07-26 to the archive end.
3. **Regressor sets** — on-structure (with borrowed station radiation), station, ERA5. Radiation is global horizontal (`surface_solar_radiation`) in every set; no other radiation variable. Skin temperature is dropped: ERA5's cell is too coarse to say anything about a wall face.
4. **Wall temperature and on-structure radiation** — tested on the current era as a ladder, not carried in the main line.
5. **Compensation** — taken as given.
6. **Imputation** — the target is never filled; regressor dust up to two hours is filled and flagged; outages are bridged as hypotheses only.
7. **Architecture** — additive grey-box without autoregression (Model A) as the core; lagged-regressor impulse response on air temperature and station radiation (Model B) as one section; forecasting out of scope.
8. **Figures** — every NeuralProphet plot type, redrawn under the project's graphical rules; native plots in the notebook only.
9. **Monitor** — three charts, one per mechanism; fixed reference window 2019-01-01 to 2021-12-31; budgets 90, 90 and 365 watched days per false alarm.
10. **Language** — English, matching Studies 03 and 04.
11. **Harmonic diagnostics** — the seasonal Fourier orders and the daily-cycle weight curve are measured in Phase 1b before any model is fitted; a mid-July cosine is the fallback weight.
