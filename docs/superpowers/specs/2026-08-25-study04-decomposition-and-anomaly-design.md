# Study 04 · Design — Decomposition, Nowcast Expectation, and Anomaly Judgement

Date: 2026-08-25
Status: approved 2026-08-25
Folder: `studies/04_neuralprophet_inclination_prediction/`
Supersedes: `.agents/ORIGINAL_REQUEST.md` (2026-08-22)

---

## 1 · The question this study answers

A reading arrives every twenty minutes: air temperature, relative humidity, and inclination.
The operator's question is not "what will the inclination be next week" but:

> **Given the environment measured at this instant, is this inclination the one the wall was
> expected to show — or is it a departure?**

Answering it requires three things that must not be confused with one another:

1. **A decomposition.** An explicit statement of what the inclination record is made of — a slow
   trend, a daily thermal cycle, a response to the measured environment, and a remainder. Without
   this there is no *expected* value to compare a new reading against, only a black-box number.
2. **An expectation with an interval.** A predicted value carrying a calibrated uncertainty, so
   that "departure" means "outside what the model said, at a stated confidence" rather than
   "larger than a number someone chose".
3. **A forecast, and its honest limit.** How far ahead the record can be projected at all, and at
   which horizon the model stops beating a trivial baseline.

The study also inherits an unanswered question from Study 01, which recorded the omission
explicitly: the anatomy of the missing data. That is not a preliminary here — it decides what the
model can be, and it is treated as a result in its own right.

---

## 2 · What is already established, and therefore not re-litigated

Every fact below comes from an earlier study or from the binding documentation. The design rests
on them; it does not re-derive them.

### 2.1 From Study 01 — the record

| Fact | Value | Source |
|---|---|---|
| Analysis product | `inc_comp_cleaned` — compensated, anchored once across the whole record, spike-cleaned | `de_lib.py:1233` |
| Interpolation flag | `inc_spike` — `True` marks a value the cleaning invented | `de_lib.py:2340` |
| Era label | `era` ∈ {`legacy`, `current`} | `de_lib.py:442` |
| Native cadence | 20 minutes, 212,257 rows, 2018-07-26 → 2026-08-21 | archive header |
| A real event exists | ST02 summer-2026 inclination anomaly, flagged at 5σ against a 24-hour rolling median | `DE_F05`, `DE_F14`, `de_lib.anomaly_by_channel` |

The last row is why the reference window stops in mid-2025: a window used to calibrate a detector
must be in control, and Study 01 flagged that stretch. It is excluded as a precaution and nothing
more. The event is far too large to say anything about sensitivity — a detector that finds a 5σ
excursion has shown only that it is not broken — so this study states no result from it and measures
its sensitivity by injection instead (D12).

### 2.2 From Study 03 — the couplings

| Driver | Delay | Gain on `inc_comp_cleaned` | Survives a change of proxy source? |
|---|---|---|---|
| Air temperature | 0 h | **−2.79 mdeg/°C**, r = −0.957 (diurnal band) | Yes — ground station −2.23, ERA5 −2.04 |
| Solar radiation | 1 h | −0.026 mdeg/(W·m⁻²), r = −0.867 | Yes |
| Relative humidity | 0 h | +0.686, r = 0.79 | Reported, **not causal** — inverse of the diurnal temperature cycle |
| Wall temperature | −2 h (leads), τ = 4 h at levels | −1.75 mdeg/°C diurnal | On-structure only; lead must be clamped to zero before predictive use |

Binding consequences:

- **No level-band time constant may be carried into a model.** Fifteen of twenty-two candidate
  time constants pinned at the 72-hour scan boundary, which is a seasonal slide and not thermal
  inertia. External drivers act instantaneously or not at all.
- **The battery control is not silent.** `batt_str` reaches r = −0.673 in the diurnal band. It
  remains the right negative control, but it is a *conservative* floor, not a zero.
- **Zero lag is a bound, not a value.** Study 03 states that its hourly resolution cannot resolve
  a delay shorter than one hour, and names a 20-minute-native scan as the way to do better. This
  study runs at 20 minutes and can therefore settle it.
- **Precipitation, wind direction, pressure and dew point were rejected**; wind speed was unstable
  across months. None enters this study.

### 2.3 From the binding documentation

- The **absolute level carries no structural information** — it is set by how the instrument sat
  in its mount (`docs/raw-data-format.md` §7.1).
- The **raw channel measures +1.63 mdeg/°C**, the wrong sign, an unresolved contradiction (§7.5).
  The compensated channel does not carry it: measured here, `corr(inc_comp_cleaned, tair) = −0.865`.
- **Instrument eras are not this study's business.** The two eras are already anchored as a single
  series by Study 01 (§3.3, §7.3). Nothing here reads, labels or offsets them.
- **Compensation has already happened.** `adc.DOCUMENTED_COEFF = 0.005` subtracts a thermal term of
  a size comparable to the true thermal slope, and whether that is the right size is Study 01's
  question. Any coefficient this study fits is what the wall does *after* that correction — which is
  the only quantity a deployed monitoring system ever sees — and is labelled as such.

### 2.4 Measured for this design (2026-08-25, window 2023-06-21 → 2026-08-21)

These numbers were measured directly from the archive to settle design decisions that would
otherwise be taste. They are reproduced by the notebook in Phase 1 and become study artefacts.

**Gap anatomy — 83,305 slots on the 20-minute grid**

| | 20-minute grid | 1-hour grid |
|---|---|---|
| Inclination coverage | 75.6 % | 78.3 % |
| Number of gaps | 1,647 | 391 |
| Missing time inside 4 outages > 7 d | 72.8 % | 82.1 % |
| Gaps ≤ 1 h | 1,282 (8.6 % of missing time) | 205 |
| Median contiguous segment | **1.7 h** | **19 h** |
| Segments surviving 24 h lags + 24 h horizon | 101 → 15,000 windows | 101 → 13,243 windows |

Two findings: the missing time is **four outages plus a dust of short dropouts**, where the dust is
8.6 % of the missing time but 78 % of the gap *count* and is what destroys contiguity. And
**`tair` and `rh` cost nothing** — `inc` alone and `inc + tair + rh` produce identical segment
counts, because the missingness is perfectly nested. Adding `twall` and `sr` collapses coverage
from 75.6 % to 17.0 %.

**Signal and noise**

```
level:             lag-1 autocorrelation  0.99771     range −45.2 … +168.1 mdeg
change @ 20 min:   lag-1 autocorrelation −0.0581      std 2.57   MAD 0.78 mdeg
change @ 1 h:      lag-1 autocorrelation +0.3494      std 3.70   MAD 1.30 mdeg
corr(Δinc, Δtair)  −0.778 @ 20 min        −0.893 @ 1 h
corr(inc,  tair)   −0.865 @ 20 min        −0.867 @ 1 h
```

---

## 3 · Design decisions, and the evidence for each

### D1 · The level is never used as a prediction score, and never abandoned as a modelling target

The level's lag-1 autocorrelation is 0.998. Any error metric computed against it is dominated by
persistence, so a reported MAE would measure the sampling interval rather than the model. But the
level is the only target on which a *trend* exists, and the trend is the structurally interesting
component. The resolution is that the level is decomposed and monitored, while skill is scored on
the change.

### D2 · Forecast skill is scored on the gap-safe hourly change; the anomaly test is not differenced

At 20 minutes the change has lag-1 autocorrelation **−0.058** — the signature of additive white
noise dominating a small true increment. At one hour it is **+0.349**, and the thermal coupling
sharpens from −0.778 to −0.893. Differencing at 20 minutes therefore throws away signal and keeps
noise. The anomaly question does not require differencing at all: it compares a measured level
against an expected level under measured forcing, so it runs at the native 20-minute cadence where
the operator actually needs it.

### D3 · Drift is read from the fitted trend, never from the mean of changes

The mean gap-safe change gives **−65.7 mdeg/yr** at 20 minutes and **+0.97 mdeg/yr** at one hour —
the same record, two cadences, opposite conclusions. The cause is that segment endpoints do not
sample the diurnal cycle uniformly, so the mean increment is a biased drift estimator. This makes
the decomposition a requirement rather than an ornament: only a trend fitted to the level, with
the daily cycle carried by an explicit seasonal term, estimates drift without that bias.

The trend is read **over the training window only**. Past its last changepoint a piecewise-linear
trend is an extrapolation, and on a record whose evaluation stretch opens with a 103-day hole that
extrapolation is the largest error anywhere in the decomposition — measured here, it inflates the
residual's standard deviation from 13.8 mdeg to 41.9 and drives excursions to −194 mdeg. A trend is
evidence where observations constrain it and arithmetic thereafter.

### D4 · Contemporaneous regressors in the nowcast are not leakage

The anomaly model consumes `tair` and `rh` measured at the same instant as the inclination it
judges. This is legitimate because those readings arrive together in the same acquisition record —
the model is asked a *conditional* question ("given this environment, is this inclination
expected?"), not a predictive one. The forecast model is held to the opposite rule and may consume
only past predictor values. The two are never scored on the same table.

### D5 · Two models, because autoregression eats the physics

With `n_lags > 0`, NeuralProphet's AR component absorbs most of the diurnal structure, and the
seasonal component becomes the periodicity *left over after* AR — not the wall's thermal cycle.
Reporting that as physics would be wrong. Model A is fitted **twice**, because attribution and monitoring want opposite things from the
trend and one fit cannot serve both. The attribution fit keeps the trend, since the drift is the
structurally interesting component and does not exist without it. The monitoring fit drops it,
because an extrapolated trend is the largest error in the out-of-sample residual by a factor of
three, and a control chart calibrated on it would spend its alarm budget on the model's own
arithmetic. Dropping it also leaves the drift *inside* the residual, which is where a monitoring
system should meet a drift — in front of the detector, not subtracted away behind it.

| | **Model A — attribution** | **Model A — monitoring** | **Model B — forecast** |
|---|---|---|---|
| Target | `inc_comp_cleaned` level | same | gap-safe hourly change |
| Grid | 20 min | 20 min | 1 h |
| `n_lags` | 0 | 0 | 24 (tuned against segment survival) |
| Regressors | `tair`, `rh` contemporaneous (future regressors) | same | `tair` lagged, past only |
| `growth` | `linear`, changepoints constrained to covered time | `off` | `off` |
| Yearly term | fitted | fitted | not applicable |
| Reads | trend and drift, daily seasonality, regressor contributions, component shares | the residual that becomes the expectation error and the control statistic | skill against baselines, horizon limit |
| Answers | What is the record made of? How much drift is there? | Is this reading expected? Is this a departure? | How far ahead is prediction worth anything? |

Measured over the evaluation stretch, the two Model A fits differ exactly where it matters: residual
σ 41.9 mdeg for the attribution fit against 13.8 for the monitoring fit, and residual MAD 19.7
against 2.7. In-sample they agree closely, which is why the split is invisible until a model is
asked to extrapolate.

### D6 · Ablation is mandatory, or nothing is attributable

Skill from a model carrying both AR memory and `tair` cannot be assigned to either. The study runs
a fixed ladder, and reports each rung's *increment*:

```
seasonal-naive baseline  →  AR only  →  AR + tair  →  AR + tair + rh
                                                   ↘  battery-only negative control
```

Wall temperature and solar radiation are **out of scope for this study** (decision D11).

Study 03's finding that the battery control is not null is inherited: the control marks a
conservative floor, and a driver is credited only when it clears that floor.

### D7 · Changepoints are placed on covered time

NeuralProphet's default places ten changepoints uniformly over the training span, blind to
coverage. This window contains outages of 40, 42, 17 and 103 days. A changepoint inside an outage
is constrained by no data and lets the trend wander. Changepoints are therefore placed at
quantiles of the *observed* timestamps, and trend is never read across a gap.

### D8 · Yearly seasonality is measured, and then kept whatever the measurement says

The window spans 3.2 annual cycles with a 103-day hole in the last one, which is not obviously
enough to identify an annual term. It is therefore fitted both ways and the comparison is reported —
but the annual term is carried forward regardless of which way the held-out error falls.

The two criteria genuinely disagree on this record: the annual term costs about 12 mdeg of held-out
MAE. Keeping it anyway follows from what Model A is for. It is read as an attribution of variation
to named causes, and a cause left out of a model does not leave the record: it moves into the
residual. Every control chart in Phase 5 is calibrated on that residual, so an unmodelled annual
cycle — measured here at roughly 75 mdeg peak to trough — would be read as a structural departure,
spend the entire false-alarm budget on the seasons, and bury the millidegree-scale movements the
detector exists to find. A forecasting study would choose the other way; this one is not a
forecasting study.

### D9 · Instrument eras are out of scope

Study 01's cleaned, compensated series is this study's raw material, and the changeover of
21 February 2025 is a fact about parsing the `.adc` archive — Study 01's subject, not this one's. No
era term is fitted, no era offset is estimated, and no era label is read: `cadence_evidence` and
`hourly_change` are called with `era=None`. The two units are already anchored once as a single
series, and re-deriving that anchoring here could only damage it.

### D10 · Compensation is taken as given

Whether the documented coefficient is correctly sized, and what the raw channel would have shown,
are Study 01's questions. This study is blind to them: it models `inc_comp_cleaned` as delivered,
fits no raw channel, and makes no claim about the §7.5 sign contradiction. The gain it learns is a
post-compensation gain, which is what a deployed system sees, and it is confronted with Study 03's
independently measured −2.79 mdeg/°C — a comparison that does not depend on the compensation at all.

---

### D11 · Wall temperature and solar radiation are excluded

Requiring `twall` and `sr` collapses the window from 75.6 % to 17.0 % coverage — 14,122 complete
rows in 271 segments, of which only 23 segments and 3,802 training windows survive a 24-hour lag
and a 24-hour horizon. Study 03 established that both are real drivers, so their exclusion is a
scope decision and not a claim that they do not matter: the channels exist for less than half the
window, and carrying them would force every result to be reported twice, on two incomparable
windows. They are named in the limitations as the first candidate for an extension study.

### D12 · Sensitivity is stated from injections shaped like damage

The record holds one flagged event, and it is useless as a sensitivity test: a 5σ excursion found by
a 24-hour rolling median is found by anything, so detecting it measures nothing but the detector's
pulse. Sensitivity is therefore established by injecting departures of known size, length and
**shape** into the residual, and an injected departure earns its place only if its shape corresponds
to a mechanism a three-leaf stone wall can actually produce — two masonry leaves either side of a
weaker rubble-and-mortar core:

| Kind | Injected shape | Mechanism it stands for |
|---|---|---|
| **Amplitude growth** | A daily harmonic whose amplitude grows to the stated size over the stated duration, then holds | Progressive loss of composite action between the leaves — delamination at the leaf-to-core interface, or loss of through-stones. A less stiff section bends further under the same daily heating, so the diurnal swing grows while its timing and its mean stay put. |
| **Phase change** | A daily harmonic in quadrature, of the amplitude a stated timing shift implies | A changed thermal path rather than a changed stiffness — water ingress raising the core's heat capacity, or a crack re-routing conduction. The wall answers the same forcing later or earlier. |
| **Drift** | A linear accumulation at a stated rate, in mdeg per year, running to the end of the record | Creep of the lime-mortar core under sustained load, thermal ratcheting of the outer leaf, or foundation settlement. Slow, monotone, invisible in any single day. |

Magnitudes stay in millidegrees, the unit the instrument reports; a phase change is quoted by the
timing shift in hours that produced it, converted through the daily amplitude the decomposition
measures. The generic step, ramp and pulse injections remain in the library as building blocks, but
the study reports the three mechanisms above. Every detectability figure is quoted at the stated
false-alarm budget, and an alarm counts only where the uncontaminated record is silent.

---

## 4 · Metrics — what is measured, and why that metric

### 4.1 Point accuracy

| Metric | Why it is here |
|---|---|
| **MAE** | Primary. Robust, in mdeg, directly interpretable against the 1.3 mdeg noise floor. |
| **RMSE** | Reported beside MAE; the RMSE/MAE ratio diagnoses whether error lives in the tail. |
| **Bias** | A decomposition with a drifting mean residual is misspecified, and bias is how that shows. |
| **MASE** | Scale-free, so horizons and cadences are comparable on one axis. *(new)* |
| **R²** | Reported for the change only. On the level it is an artefact of persistence. |

### 4.2 Skill, relative to baselines

Absolute error answers nothing without a reference. Baselines: **zero-change**, **persistence**,
and **seasonal-naive at 24 h**. Skill is `1 − MAE_model / MAE_baseline`, computed **paired** and
with a **block bootstrap** (24-hour blocks, horizon-adaptive), because residuals are autocorrelated
and an unpaired comparison would overstate significance. A horizon counts as skilful only when the
bootstrap interval excludes zero.

### 4.3 Uncertainty calibration

An interval that is drawn but never checked is decoration.

| Metric | Why it is here |
|---|---|
| **PICP** — empirical coverage of the nominal 90 % interval | The single number that says whether the interval means what it claims. |
| **MPIW** — median interval width | Coverage is trivially achievable by widening; width is the price. |
| **Pinball loss** at q05 / q95 | Scores the quantiles themselves, not just whether they bracketed. *(new)* |
| **Winkler interval score** | Combines width and violation penalty into one comparable number. *(new)* |

Calibration is reported **per horizon**, since it typically degrades with distance.

### 4.4 Decomposition quality

| Metric | Why it is here |
|---|---|
| **Variance share per component** | Says what the record is actually made of — the study's headline claim. |
| **Residual whiteness (Ljung–Box)** | Structure left in the residual means a component is missing; the anomaly detector would then alarm on model error rather than on the wall. |
| **Component stability across folds** | A trend slope or thermal gain that changes sign between folds is not a finding. |
| **Agreement with Study 03** | The fitted `tair` contribution must reproduce −2.79 mdeg/°C within its uncertainty. An independent method recovering an independently measured constant is the strongest validation available here. |

### 4.5 Anomaly detection

This is where "quality of results" has to be defined carefully, because a detector can be made to
look perfect by never alarming.

| Metric | Why it is here |
|---|---|
| **ARL₀** — in-control average run length on a quiet held-out window | The false-alarm rate, expressed as "one false alarm every N days". Fixed first; everything else is measured at that setting. |
| **Detection delay** per injected mechanism | How long the operator waits before the alarm fires, counting only alarms the uncontaminated record does not raise. |
| **Smallest detected departure** vs persistence duration, per mechanism | The headline operational number, stated three times because the mechanisms are not interchangeable: the smallest growth of the daily swing in mdeg, the smallest timing shift in hours, and the slowest drift in mdeg per year that this system finds — and how long each must last. |
| **Mechanisms undetected across the swept range** | A limit of this instrument and this model, reported as such. A sensitivity claim without its blind spots is a sales figure. |

---

## 5 · Architecture and the `shmlib` contract

Every function lives in `studies/shmlib/`. Nothing is written into the study folder. The notebook
declares all paths, parameters and choices in its parameter cell and passes them as arguments.

### 5.1 Reused unchanged

`proxies.load_response`, `proxies.load_sensor_forcings`, `proxies.join_eras`, `proxies.harmonise`,
`prediction.hourly_change`, `prediction.contiguous_segments`, `prediction.expanding_segment_folds`,
`prediction.execution_folds`, `prediction.baseline_predictions`, `prediction.paired_mae_skill`,
`prediction.gap_closure_summary`, `figures.plot_channel_panels`, `tables.write_table`,
`viz.apply_report_style`, `viz.finish`.

### 5.2 Adapted — additive only, existing callers unaffected

| Function | Change | Guard |
|---|---|---|
| `prediction.neuralprophet_backtest` | new `decompose=False` parameter; new `changepoints=None`, `growth='off'`, `freq=None` parameters | Defaults reproduce today's behaviour exactly; Study 04's existing tests must pass unchanged |
| `prediction.neuralprophet_predict` | same `decompose` parameter | as above |
| `prediction.score_predictions` | adds `mase`, `pinball_q05`, `pinball_q95`, `interval_score` columns | Existing columns keep their names, order and values |

### 5.3 New in `shmlib`

```
prediction.covered_changepoints(index, n_changepoints, observed_mask)
prediction.decompose_components(model, frame, regressors, quantiles)
prediction.component_variance_shares(components, columns)
prediction.residual_diagnostics(residuals, lags=(1, 24, 72))

monitoring.reference_stats(residuals, start, end)              # new module
monitoring.ewma_chart(residuals, mu, sigma, lam, L)
monitoring.cusum_chart(residuals, mu, sigma, k, h)
monitoring.joint_alarm(ewma_alarm, cusum_alarm, window)
monitoring.alarm_episodes(alarm, residuals)
monitoring.inject_anomaly(series, kind, magnitude, start, duration, period)
    # kind: 'step' | 'ramp' | 'pulse' | 'amplitude' | 'phase' | 'drift'  (D12)
monitoring.phase_shift_amplitude(daily_amplitude, shift_hours, period_hours)
monitoring.detectability_curve(residuals, magnitudes, durations, kind, chart_params)
monitoring.average_run_length(alarm, freq)

figures.plot_decomposition_stack(components, ...)
figures.plot_prediction_band(observed, expected, lower, upper, ...)
figures.plot_control_chart(statistic, limits, alarms, ...)
figures.plot_metric_vs_horizon(metrics, ...)
figures.plot_detectability(curve, ...)
```

`heritageshm/monitoring.py` already contains working EWMA, CUSUM, joint-alarm and alarm-summary
implementations. They are **prior art to adapt from, not to import**: the studies' rule is that
`shmlib` is the only library a study has. The port is mechanical and is delegated.

### 5.4 Figure conventions

Binding, from `instructions-pipeline.md`: Okabe–Ito for categories, Cividis for scalars, one fixed
colour per measured channel (inclination `#0072B2`, air temperature `#E69F00`, relative humidity
`#009E73`, wall temperature `#DAA520`, solar radiation `#CC79A7`), Vermilion `#D55E00` reserved for
annotations and event markers, span highlights black at 5 % opacity, legends below the axes and
never inside them, PNG and SVG both written by `viz.finish`.

---

## 6 · Artefacts

Tables (`outputs/`, LaTeX bodies alongside):

| File | Content |
|---|---|
| `NP_01_window_coverage.csv` | Accepted coverage per channel over the window |
| `NP_02_gap_inventory.csv` | Every gap: start, end, duration, class |
| `NP_03_segment_survival.csv` | Segments and training windows surviving each `n_lags` choice |
| `NP_04_cadence_evidence.csv` | Autocorrelation and coupling at both cadences — the D2 decision |
| `NP_05_component_shares.csv` | Variance share of trend, daily, `tair`, `rh`, residual |
| `NP_06_learned_gains.csv` | Fitted gain against Study 03's three independent measurements |
| `NP_07_nowcast_metrics.csv` | MAE, RMSE, bias, R², PICP, MPIW, pinball, interval score |
| `NP_08_residual_diagnostics.csv` | Ljung–Box, residual scale, stability across folds |
| `NP_09_alarm_episodes.csv` | Alarm start, end, duration, peak standardised residual |
| `NP_10_detectability.csv` | Smallest detected departure by mechanism and duration, at fixed ARL₀ |
| `NP_11_forecast_metrics.csv` | All metrics by horizon and model rung |
| `NP_12_skill_vs_baseline.csv` | Paired block-bootstrap skill with intervals |
| `NP_13_ablation.csv` | Increment attributable to each predictor |
| `NP_14_gap_closure.csv` | Accumulated reconstruction check, with explicit verdict |
| `NP_15_run_metadata.csv` | Executed lags, horizons, epochs, seed, refit policy, versions |
| `NP_16_component_stability.csv` | Trend slope and thermal gain re-estimated on each third of the training period |

Figures: `NP_F01` on-structure record (exists) · `NP_F02` gap anatomy · `NP_F03` segment survival ·
`NP_F04` cadence evidence · `NP_F05` decomposition stack · `NP_F06` daily cycle and regressor
response · `NP_F07` observed vs expected with band · `NP_F08` control chart ·
`NP_F09_detectability_{amplitude,phase,drift}` one curve per damage mechanism · `NP_F10` skill vs
horizon · `NP_F11` ablation · `NP_F12` gap closure.

---

## 7 · Report structure

The report follows Study 03's proven shape — every decision explained where it is made, before the
result that depends on it. Target length 12–16 pages.

1. **Introduction** — the operational question; what is inherited and not re-litigated. *(exists,
   minus the fabricated paragraph)*
2. **The record this study works on** — window, coverage, `NP_F01`.
3. **The anatomy of what is missing** — gap classes, why the dust matters more than the outages,
   what segmentation costs. `NP_F02`, `NP_F03`, `NP_01`–`NP_03`.
4. **Method**
   - 4.1 Why the level is decomposed but never scored — the 0.998 autocorrelation.
   - 4.2 Why forecasting is done on the hourly change — the −0.058 versus +0.349 evidence.
   - 4.3 Why the anomaly test is not differenced, and why contemporaneous regressors are not
     leakage.
   - 4.4 Why two models — what autoregression does to the seasonal component.
   - 4.5 How gaps are honoured — segment IDs, no imputation, and the fail-loud configuration.
   - 4.6 Changepoints on covered time; why eras are out of scope.
   - 4.7 The metrics, and what each is for.
5. **What the record is made of** — the decomposition; component shares; the learned thermal gain
   set against Study 03's −2.79 mdeg/°C. `NP_F05`, `NP_F06`.
6. **Is this reading expected?** — nowcast accuracy, interval calibration, `NP_F07`.
7. **Judging a departure** — control charts, ARL₀, and detectability per damage mechanism: the
   smallest amplitude growth, timing shift and drift the system finds, with what each stands for in
   a three-leaf wall. `NP_F08`, `NP_F09`.
8. **How far ahead is prediction worth anything?** — horizons, baselines, skill, ablation.
   `NP_F10`, `NP_F11`.
9. **Can the model fill gaps?** — gap-closure verdict. `NP_F12`.
10. **Verdict** — one paragraph per question, each with its qualification.
11. **Limitations** — compensation is not independent; `twall` and `sr` exist for 17 % of the
    window; the battery control is not null; three annual cycles cannot identify a yearly term;
    one sensor, one site.

Every number in the prose traces to a named artefact. A claim with no artefact behind it does not
enter the report — the standing text is being rebuilt precisely because that rule was broken.

---

## 8 · Phases and checkpoints

Each phase ends at a checkpoint that is **shown to the user and approved before the next phase
starts**. Delegation column: **O** = orchestrator, **S** = Sonnet subagent under written
instruction.

### Phase 0 · Truth restoration — *O*
Remove `clean_up.py` and `replace_report.py` from the study folder. Delete the fabricated results
paragraph from the report. Rewrite the study `README.md` and the study-04 row of `studies/README.md`
to state the real status. Decide whether `studies/` goes under version control before any further
work.
> **Checkpoint 0:** no claim anywhere in the folder is unsupported by an artefact on disk; no code
> outside `shmlib`; user has ruled on version control.

### Phase 1 · The record and its gaps — *O plans, S implements*
Notebook movements 1–2. Coverage, gap inventory, segment survival at both cadences.
> **Checkpoint 1:** `NP_01`–`NP_03` and `NP_F02`–`NP_F03` reproduce §2.4 exactly. Every parameter
> visible in the parameter cell. No inline logic longer than ten lines.

### Phase 2 · Cadence and target, settled on evidence — *O*
Autocorrelation and coupling at both cadences, written as a decision table rather than a
preference.
> **Checkpoint 2:** `NP_04` supports D2 and D3, or the design changes to follow the evidence.

### Phase 3 · `shmlib` additions — *S under precise instruction, O reviews*
The adapted parameters and the new decomposition, monitoring, metric and figure functions, each
with a NumPy-style docstring and unit tests.
> **Checkpoint 3:** `python studies/shmlib/tests/test_shmlib.py` and every study's test file pass,
> including Study 03's, proving the adaptations changed nothing for existing callers. New functions
> have tests that fail before the implementation exists.

### Phase 4 · Model A — decomposition and expectation — *O*
Fit on the level at 20 minutes, `n_lags=0`, contemporaneous `tair` and `rh`, constrained
changepoints; the yearly-seasonality comparison; the raw-channel sensitivity fit.
> **Checkpoint 4:** the learned `tair` contribution agrees with Study 03's −2.79 mdeg/°C within
> uncertainty; residuals pass Ljung–Box or the missing component is named; interval coverage is
> within a stated tolerance of 90 %. **If the coefficient disagrees, the study stops here and the
> disagreement becomes the finding.**

### Phase 5 · Anomaly judgement — *O designs, S runs the sweeps*
Reference window, control-chart tuning to a fixed ARL₀, and the injected sweeps over the three
damage mechanisms of D12.
> **Checkpoint 5:** ARL₀ measured and stated in days; a detectability curve exists for each
> mechanism; the smallest departure found is stated per mechanism and persistence, in the unit that
> makes it physical, and any mechanism the sweep never detects is reported as a blind spot.

### Phase 6 · Model B — forecast and its limit — *S runs, O interprets*
The ablation ladder over horizons 1…168 h against three baselines, with paired block-bootstrap
skill.
> **Checkpoint 6:** the horizon at which skill ceases to exclude zero is stated with its interval,
> and the increment attributable to `tair` beyond AR memory is reported — even if it is small.

### Phase 7 · Gap reconstruction — *S*
`gap_closure_summary` over prior-only estimates.
> **Checkpoint 7:** an explicit safe/unsafe verdict on using model output to fill gaps.

### Phase 8 · The report — *O only, never delegated*
Written to §7's structure, from the artefacts, in full academic prose.
> **Checkpoint 8:** every number traces to a named file; the study `README.md` matches what the
> folder contains; the report builds twice cleanly.

---

## 9 · Risks and kill criteria

| Risk | Detection | Response |
|---|---|---|
| Segmentation leaves too little data at 20 min | Phase 1 `NP_03` | Lower `n_lags`, or model A at 20 min and B at 1 h as designed |
| Learned gain contradicts Study 03 | Checkpoint 4 | Stop; the contradiction is the result, not a bug to tune away |
| Residuals are not white | Checkpoint 4 | Name the missing component before any alarm is trusted |
| Detector alarms constantly | Phase 5 ARL₀ | Re-tune to a fixed false-alarm budget; never report a detector without ARL₀ |
| `tair` adds no skill beyond AR | Checkpoint 6 | Report it. Study 03 measured a real coupling; a small *predictive* increment is a legitimate and interesting finding |
| Trend wanders inside the 103-day outage | Visual, Phase 4 | Constrained changepoints (D7); never read trend across a gap |

---

## 10 · Decisions taken

Approved 2026-08-25.

1. **Version control** — `studies/` is committed as of `555da8c`. All later work is committed as it
   is made.
2. **`twall` and `sr`** — excluded, per D11. An extension study may add them later on their own
   matched window; this one does not report them.
3. **Study identity** — rebuilt **in place as study 04**, keeping the `NP_` artefact prefix. Study 03
   already cross-references study 04, and the standing introduction and `NP_F01` are sound and
   reused. The question the study answers is widened from prediction alone to decomposition,
   expectation and anomaly judgement, and its `README.md` is rewritten to say so.
4. **Language** — English, matching study 03.
