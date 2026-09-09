# Study 06 · Design — Moisture and heat exchange as drivers of the station 02 inclination

Date: 2026-09-07
Status: draft, awaiting approval. Revised the same day, after the measurement of §2.6, to add decisions D11 and D12, Phase 3b, artefacts `HT_13`–`HT_15` and `HT_F09`–`HT_F10`, report section 6, and open questions 15 to 17.
Folder: `studies/06_hydro_thermal_drivers/`, artefact prefix `HT_`
Builds on Study 05 and on the radiation time-constant sweep it records as decision D15 (Study 05 `report05_check.md`, item of 2026-09-07). Study 05 stays as it is; nothing in it is re-run by this study.

---

## 0 · Working rules for every step of this study

- **Caveman ultra, always.** Every session and every spawned subagent runs under `/caveman:caveman ultra` from the first message. The only exception is prose that reaches a produced document — the study report, the README, this spec, docstrings, comments and commit messages — which is written in full academic or technical prose.
- **Delegate down.** The orchestrator plans, decides, reviews checkpoints and writes the report. Everything a lower-tier model can do goes to subagents spawned with the Agent tool, each prompt starting with "respond in caveman ultra mode": folder scaffolding, `shmlib` functions under written instruction, tests, notebook cells, table and figure export, LaTeX plumbing, graph updates, bulk reads. The report prose is never delegated.
- **Delivery sequence.** Study folder, then this spec, then the user's approval, then an implementation plan written with the writing-plans skill, then implementation phase by phase with a checkpoint approved before each next phase.
- **Entry condition.** Study 05 has run its D15 sweep and recorded which radiation operator, delay-only or filtered, it hands on. Study 06 starts from that operator and does not re-decide it.

---

## 1 · The question this study answers

Study 05 decomposed the eight-year record into a trend, an annual term, a daily term, three instantaneous drivers and a residual. Its decomposition leaves two things without a driver. The trend holds 60 per cent of the fitted variance and moves by more than a hundred millidegrees over six years, not as a drift but as a slow reversing process: rising through 2019 and 2020 at up to 104 mdeg per year, falling through 2021, falling again to a minimum in March 2024, rising through the rest of 2024. The yearly term holds another 12 per cent, peaks in late March and troughs in late September, in quadrature with the air temperature, which Study 05 reads as a delayed thermal response. Neither reading has been tested against a measured driver, because Study 05 carried none on those time scales.

Two of the wall's environmental exchanges act on those scales and were never given the form in which they could be seen. Rainfall does not push the wall; it changes the moisture of the masonry and of the embankment the wall retains, over days to weeks, and drains over weeks to months. Wind does not push the wall either; it changes how fast the stone exchanges heat with the air, which is a modulation of a time constant rather than a term added to the level. Longwave loss to a clear sky cools the stone at night by a flux the pyranometer cannot see. Study 03 screened rain, wind speed, wind direction, pressure and dew point as instantaneous additive drivers and found what such a screen must find: precipitation, wind direction, pressure and dew point did not clear the negative control, and wind speed cleared it with a gain that changed sign between sources and swung by an order of magnitude between months. That verdict stands for the raw readings. This study asks whether the same measurements, transformed into the physical states through which they act, explain what Study 05 could not.

> **Is the slow, reversing movement of the wall, and the annual term in quadrature with air temperature, the response to a driver the record measures but no study has yet carried in its physical form: moisture from rainfall, and the wall's heat exchange with the air and the sky?**

Three questions, each with a null stated before any fit:

1. **Moisture.** Does an antecedent-precipitation state, or ERA5-Land soil moisture where it can be obtained, explain the trend's reversals and the yearly term's phase, on held-out whole years? Null: the trend is what it is in Study 05, a driverless slow component, and the yearly term is a thermal delay whose amplitude repeats every year.
2. **Heat exchange.** Starting from the radiation operator Study 05's D15 hands on, do longwave sky loss and wind-modulated exchange improve the expectation, and do they restore the sign of the radiation gain, which Study 05 learned positive on the on-structure set against Study 03's negative? Null: air temperature already carries the wall's thermal state and the two additions buy nothing, as the wall-temperature probe bought nothing in Study 05's ladder.
3. **Consequence for the monitor.** If a slow driver is found, what share of Study 05's trend becomes an explained component, and what does the slow chart of Study 05's D10 then watch? Null: nothing moves and the monitor is unchanged.

A fourth question is carried alongside these three. It is about the method rather than about the wall, and it is answered last, after the three above, so that it cannot change how they are judged: **is the additive model with one linear gain per driver, which Studies 04 and 05 adopted and this study inherits, the thing that limits what can be explained?** Two things are measured to answer it — whether letting the air-temperature gain vary through the year buys anything the physical states did not (D11), and how much skill an unconstrained model finds that the additive linear one does not (D12). Both are diagnostics. Neither changes the main line, and both exist because the interpretability of the grey box is an argument this project makes without having yet measured its price.

Forecasting is out of scope, as in Study 05. Re-running Study 05's monitor is out of scope: this study reports what would change and leaves the change to Study 05's successor or to the paper's model choice.

---

## 2 · What is already established, and therefore not re-litigated

### 2.1 From Study 03 — the screen of the raw readings

| Driver | Diurnal-band result | Verdict carried here |
|---|---|---|
| ERA5 and station precipitation | Did not clear the negative control | Never enters additively as a reading |
| Wind direction components | Did not clear the control | Not carried; direction enters only through the wind-speed modulation of D6 if at all |
| Station pressure, ERA5 dew point | Did not clear the control | Not carried |
| ERA5 and station wind speed | Cleared the control; monthly gain from −0.84 to −14.6 mdeg per m/s on ERA5 and +0.09 to +3.02 on the station | Never enters additively; enters only as a modulator of a time constant |
| External drivers' level-band time constants | Pin against the grid's bound | No time constant is taken from Study 03; every time constant here is swept and scored |

Source: `studies/03_thermomechanical_response/report/thermomechanical_response_report.tex`, sections on stability and on the control, and `TR_T06_verdict`.

### 2.2 From Study 05 — what is left without a driver

| Fact | Value | Source |
|---|---|---|
| Trend share of fitted variance, on-structure set | 60.1 %, peak-to-peak 116.4 mdeg | `GM_05_component_shares` |
| Trend segment rates | +17 to +104 mdeg/yr through 2019–2020; −29 to −91 through 2021; −89 and −49 from mid-2023 to March 2024; +28 through 2024; +3 in early 2025 | `GM_04d_trend_rates` |
| Trend maximum and minimum | changepoints of 2020-11-19 and 2024-03-31 | same |
| Yearly term | order one, peak-to-peak 44.4 mdeg, 12.2 % of variance, maximum +22.2 mdeg on day 80, minimum −22.2 on day 263 | `GM_05d` |
| Radiation gain, on-structure set | +0.006 mdeg per W m⁻², opposite in sign to Study 03's −0.035 | `GM_06_learned_gains` |
| Model B radiation kernel | No peak; cumulative weight climbing almost linearly over twelve hours | `GM_10_impulse_response` |
| Residual, long band | Semi-annual peak on every set, 181 to 183 days, power 0.17 to 0.24 | `GM_08b_residual_periods` |
| Residual, short band | Annual sidebands of the daily cycle at 0.9972 and 1.0027 days | same |
| Wall-temperature probe, deployable form | Skill 0.000 over the rung below, bounds −0.3 to +0.4 % | `GM_16_current_era_ladder` |
| Mean trend rate across chronological folds | Changes sign between folds; not a finding | `GM_07_component_stability` |

Binding consequences: the trend is a slow reversing process and not a drift; the yearly term's phase alone cannot separate a thermal delay from a moisture cycle, since both give quadrature; a driver measured directly as the wall's thermal state added nothing to air temperature, so the heat-exchange hypothesis starts with a low prior and needs a stated margin.

### 2.3 From Study 05 D15 — the radiation operator handed on

Recorded in `report05_check.md` on 2026-09-07 and executed inside Study 05: a sweep of the one-pole time constant applied to the station radiation, judged by held-out skill with bootstrap bounds. This study takes the operator that sweep hands on, delay-only or filtered at the chosen τ, as its base for every fit of D6, and states it in its parameter cell with a pointer to `GM_10b`.

One expectation about that operator is corrected here before it can be inherited as an assumption. The D15 note motivates the filter partly by the annual sidebands of the daily cycle that Study 05 leaves in its residual, on the ground that filtered radiation modulates through the year where air temperature does not. The modulation argument is sound and the attribution to the filter is not: a one-pole filter is linear and time-invariant, so it attenuates the 24-hour harmonic by the same factor in every season and leaves the ratio between the summer and winter daily amplitudes exactly as it found it (§2.6). What the filter changes is the waveform — the night-time tail that the pyranometer's hard floor at zero cannot produce — and the phase. The annual envelope that might reduce the sidebands is already present in the raw radiation. This study therefore expects the filter to act on the radiation gain's sign and on the residual's waveform, and does not expect it, on its own, to remove the sidebands.

### 2.4 The sources this study can read today

| Source | Channels relevant here | Cadence and extent | Loader |
|---|---|---|---|
| ERA5 (`data/raw/proxies/oikolab_weather.csv`) | `total_precipitation`, `snowfall`, `wind_speed`, `wind_direction`, `surface_thermal_radiation`, `dewpoint_temperature` | hourly, 2018-01-01 onward, 99.2 % coverage | `shmlib.proxies.load_era5`, channel map already carries `rain`, `wspd`, `wdir`, `tdew` |
| Town station (`data/raw/proxies/meteosystem_gubbio.csv`) | `Pioggia` (rain), `Int.Pio.` (rain rate), `Vento` (wind speed), `Dir`, `Raffica` (gust), `Press` | half-hourly, 2018-10-17 onward, 96.6 % coverage | `shmlib.proxies.load_ground_station`, channel map already carries `rain`, `wspd`, `wdir`, `pres` |
| ERA5-Land volumetric soil water, layers 1 to 4 | not yet downloaded | hourly, obtainable from the same provider | new download, Phase 0 |

Two facts about these channels are settled at Phase 0 before anything is joined. First, whether the file's `surface_thermal_radiation` is the downward longwave flux or the net flux: its values of 310 to 340 W m⁻² on a January night read as downward, in which case the net loss is computed as the downward flux minus the wall's own emission, from air temperature with a stated emissivity, and from the wall temperature where the probe exists. Second, whether the station's rain gauge is heated, since an unheated gauge under-reads snow and the winter half of the moisture signal would then come from ERA5 alone. Study 02's documented corrupt rainfall value is masked by the existing loader.

### 2.5 Measured for this design (Phase 0)

The design commits to no number it has not measured. Phase 0 produces the table this section will carry: the annual precipitation total per hydrological year from each source, the annual precipitation anomaly against the eight-year mean, and the day of year on which the exponentially filtered precipitation peaks and troughs for τ of 30, 60 and 90 days. The last row is the one the moisture hypothesis rests on: if the filtered precipitation peaks near day 80 and troughs near day 263, the yearly term of Study 05 has a moisture reading as well as a thermal one; if it peaks two months away from that, the moisture hypothesis is weaker before any model is fitted, and the report says so first.

### 2.6 · Measured in conversation, 2026-09-07 — the annual envelope of each driver's daily cycle

The size of a driver's day-to-night swing is not the same quantity as its level, and the two behave differently through the year. Fitting a 24-hour harmonic to every calendar day of the ERA5 record from 2018-07-26 to 2026-08-12 and averaging that daily amplitude by month gives the annual modulation of each driver's daily cycle:

| Channel | December | July | Ratio |
|---|---|---|---|
| Air temperature | 2.40 °C | 5.72 °C | 2.4 |
| Radiation, raw | 101.8 W m⁻² | 425.3 W m⁻² | 4.2 |
| Radiation, one-pole filtered, τ = 1, 4, 12 h | 98.8, 70.5, 31.1 W m⁻² | 412.5, 294.7, 129.7 W m⁻² | 4.2 |
| Target inclination (Study 05 `GM_04`) | 2.8 mdeg (day 342) | 17.1 mdeg (day 200) | 6.1 |

Three consequences, each of which this study inherits as a stated expectation rather than a hope.

First, the ratio is unchanged by the filter at every τ, which is the linearity argument of §2.3 in measured form.

Second, no single driver in the record reproduces the wall's 6.1. Radiation at 4.2 is much the closer of the two, so replacing or supplementing the temperature channel with a radiation state should shrink the annual sidebands of the daily cycle; it should not be expected to remove them, and a report that claims removal without showing the sideband power has not measured what it asserts. The sideband power per rung is therefore already among the metrics of D6.

Third, and this is what motivates D11 below, a fixed linear gain applied to a driver that modulates 2.4 cannot produce a response that modulates 6.1, but a *seasonally varying* gain applied to the same driver can. Study 03 measured the air-temperature gain over 35 months and found it ranging from −3.63 to −1.88 mdeg per °C, a factor of 1.9, which is close to the factor 2.5 that would be needed. A seasonally varying temperature response is therefore a live third explanation for variance this study is about to attribute to moisture and to heat exchange, and it is not separable from either by inspection. It is given its own test, before the joint fit, in D11.

The measurement needs no new code: `compare.daily_amplitude_phase` already fits the daily harmonic day by day, and a monthly mean of its amplitude column is the table above. Phase 0 reproduces it into `HT_13_driver_envelope.csv` from the study's own loaded frames, and adds the on-structure and station channels, which the conversational measurement did not cover.

---

## 3 · Design decisions, and the evidence for each

### D1 · The whole record, and whole years as the unit of validation

The moisture question is interannual. A thermal delay produces an annual term of the same amplitude every year; a moisture cycle produces one whose amplitude follows the wet and dry years. Separating them needs every usable annual cycle the record has, 2019, 2020, 2021, 2024 and 2025, and a validation scheme that holds out whole years rather than a trailing fifth. The window is Study 05's, 2018-07-26 to the archive end, on the twenty-minute grid, with the same target product and the same gap rules. Validation is leave-one-year-out over the five cycles, with Study 05's chronological folds run beside it for comparability with `GM_07`.

### D2 · Two hypotheses, tested separately before any joint fit

Moisture acts over weeks and competes with the trend; heat exchange acts over hours and competes with air temperature. A joint model fitted first would let each hypothesis borrow the other's variance and neither would be falsified cleanly. The moisture line (D4, D5) and the heat-exchange line (D6) are therefore run as two independent ladders on the same base model, each with its own pre-stated rule, and only a hypothesis that survives its own ladder is carried into the joint fit of D7.

### D3 · Every new driver is a state with a time constant, never a reading

Study 03's verdict on the raw readings is inherited: no rain, wind, pressure or dew-point reading enters a model additively. What enters is a state through which the reading acts, each a one-pole filter of the reading with a time constant declared in the parameter cell and swept:

| State | Reading | Filter | Time constants swept | Enters as |
|---|---|---|---|---|
| Antecedent precipitation index, API(τ) | ERA5 total precipitation plus snowfall as water equivalent; station rain as the second source | causal exponential, τ in days | 3, 7, 14, 30, 60, 90, 180 d | additive future regressor |
| Soil moisture, where obtained | ERA5-Land volumetric soil water, layer by layer | none, the reanalysis has integrated already | — | additive future regressor, one layer at a time |
| Filtered radiation | station radiation | Study 05 D15's operator | fixed by D15 | additive future regressor, as in Study 05 |
| Net longwave state | ERA5 downward longwave minus computed emission | same τ as the radiation state | tied to D15's τ | additive future regressor |
| Wind-modulated exchange | ERA5 or station wind speed | modulates the τ of the radiation state, τ(w) = τ₀ / (1 + k·w) | k swept over 0, 0.05, 0.1, 0.2, 0.5 s m⁻¹ | multiplicative, through the filter |
| Wind × temperature difference | wind speed and (air − wall) temperature | none | — | additive, current era only, diagnostic rung |

The API is computed on the proxy's own continuous record, which has almost no gaps, before the join to the target; a target gap therefore never resets a moisture state, and the warm-up problem of Study 05's D15 does not arise for τ in days. The heat-exchange states at τ in hours are handled as D15 handles them, reset after regressor gaps longer than the fill limit with a flagged warm-up of about 3τ. The filter is `shmlib.coupling.thermal_operator`, the one function Study 05 already uses; the API is the same operator with a daily unit.

### D4 · The moisture sources, and what each is for

ERA5 precipitation covers the whole window from 2018-01-01 and is the primary source; the station's gauge is the second, closer source and is scored against ERA5 on the overlap before either is used, as Study 02 scored air temperature and radiation. ERA5-Land soil moisture is the direct measurement of the state the API approximates, and if Phase 0 obtains it the study reports both and says which explains more; if it does not, the API stands alone and the limitation is stated. Snowfall is folded into precipitation as water equivalent, since the embankment receives the melt whichever form the water fell in; a variant that delays snowfall by a fixed melt period is reported only if the winter totals are large enough to matter, which Phase 0 measures.

### D5 · The trend is the competitor, and the test is run three ways

Study 05's trend, with twelve changepoints on covered time, already holds every slow movement in the record. Added to that model, a moisture state would find its variance taken and would score nothing, whether or not the wall responds to it. The moisture hypothesis is therefore tested against the trend rather than beside it:

1. **The trend's own rates.** Study 05's segment rates in `GM_04d` are regressed on the change of API(τ) over the same segments, for each τ. A slow driver that explains the reversals gives a slope of consistent sign across segments; the segment across the 2022–2023 gap is excluded, since its rate is a straight line and not a measurement. This is descriptive and comes first, before any model.
2. **Changepoints traded for a driver.** Model A is refitted with API(τ) as a future regressor and the number of changepoints swept over 0, 2, 4, 8 and 12. The question is whether a model with fewer changepoints and a moisture state reaches the held-out error of Study 05's model with twelve and none, on leave-one-year-out folds. The comparison is the paired block-bootstrap skill on the shared held-out rows, block length no shorter than τ.
3. **The yearly term's amplitude year by year.** With the moisture state in the model, the yearly term's peak-to-peak and the moisture component's annual swing are read per held-out year. A thermal delay gives the same yearly amplitude in every year; a moisture cycle gives an amplitude that follows the year's rainfall. Whichever picture the folds show is the finding.

**The expected sign is stated before the fit.** The wall retains an embankment on the mountain side and the instrument's convention counts tipping towards the mountain as negative. Wetting the embankment raises the earth pressure on the wall, which pushes it towards the valley, so the earth-pressure mechanism predicts a positive gain on the moisture state. Moisture swelling of the masonry has no sign the design can predict. A learned gain of the predicted sign is a coupling recovered; a gain of the other sign is reported as a finding about mechanism and is not tuned away.

**The rule, stated in advance.** The moisture state counts as a driver only if all three hold: the leave-one-year-out skill over Study 05's model has bootstrap bounds that exclude zero; the gain keeps its sign in every fold; and the yearly term's share of variance falls when the state is present. Failing any one of the three, the trend stays a driverless component and the report says which condition failed.

### D6 · Heat exchange, as a ladder on the base of D15

The base is Study 05's Model A on the on-structure set with the radiation operator D15 handed on. The ladder adds one exchange at a time, on the same rows, so that each rung differs from the one below in one channel only:

1. the base;
2. plus the net longwave state, at D15's τ, the night-time cooling the pyranometer cannot see;
3. plus wind modulation of the radiation state's τ, k swept, the best k kept and the sweep tabled;
4. on the current era only, where the wall probe exists, plus the additive term wind × (air − wall) temperature, as a diagnostic rung on the model of Study 05's fourth rung.

Each rung reports held-out error with the paired block-bootstrap increment over the rung below and over the base, the learned radiation gain beside Study 03's −0.035 and its sign, the residual's lag-one autocorrelation and the power of its annual sidebands of the daily cycle. **The rule, stated in advance:** an exchange counts as an improvement only if its skill bounds exclude zero; the sign of the radiation gain is reported at every rung whatever the skill says, because a gain of the right sign at zero skill is itself a finding about what the level can separate. The prior is low: Study 05's ladder found that the wall's own temperature, the integrated state these exchanges approximate, bought nothing over air temperature.

### D7 · The joint fit, only for what survived

Whatever passes D5 and D6 is fitted together, once, on the on-structure set, with the leave-one-year-out folds, and its component shares, gains and residual diagnostics are tabled beside Study 05's `GM_05`, `GM_06` and `GM_08`. If nothing passes, D7 is one sentence in the verdict. The joint fit is then repeated on the station and ERA5 sets for the transferable arm, since every driver this study adds is available from ERA5 alone.

### D8 · Model, folds and everything else as in Study 05

The model is Study 05's Model A specification, unchanged except for the regressor set: linear trend with changepoints on covered time, yearly order one and daily order two as Study 05's harmonic diagnostic fixed them, plain daily term, quantile regression at 0.05 and 0.95, no autoregression, imputation off, the same seed, learning rate and epochs. Every one of these is declared in the parameter cell with a pointer to the Study 05 artefact that fixed it. Gap rules are Study 05's D13. The clock check of Study 05's D13 is inherited, not repeated, except for the two new ERA5 channels, whose stamping is checked once against the radiation channel they share a file with.

### D9 · What the slow chart would then watch

If D5 finds a moisture driver, the residual of the joint model over Study 05's monitored period, from 2022-01-01, is passed through Study 05's slow chart with the same reference window, 2019 to 2021, and the same budget, and the episodes are set beside Study 05's `GM_11`. The point is not to re-run the monitor but to show what changes: which of Study 05's slow episodes were the wall answering the rain, and which remain. Nothing else of the monitor is re-run.

### D10 · Figures under the project's rules

Every figure is drawn by `shmlib.figures` under the binding rules of `instructions-pipeline.md`. Two identity colours are already assigned to the channels this study introduces, precipitation and wind speed, in `shmlib.viz`; the longwave and soil-moisture states take unspent Okabe–Ito colours assigned once and recorded there.

### D11 · A seasonally varying temperature gain is the third hypothesis, and it is tested before the joint fit

Studies 04 and 05 fitted the air temperature as a single additive future regressor with one gain for the whole record. That is a modelling choice, and §2.6 shows it is a consequential one: a fixed gain on a driver whose daily swing modulates 2.4 through the year cannot produce a response that modulates 6.1, whereas a gain that is itself about 2.5 times larger in summer can. Study 03's month-by-month range of −3.63 to −1.88 mdeg per °C is consistent with such a variation and does not establish it, because the same months that are warmer are also sunnier and wetter.

This matters to both of the study's own hypotheses rather than being a side question. A seasonally varying temperature gain produces a component at the annual period, which is the period the moisture state of D5 is being asked to explain; and it produces exactly the summer-weighted daily response that the radiation states of D6 are being asked to supply. Left untested, it is a confound that could take credit under either heading, and a model given more freedom in the temperature response before the physical states are added will absorb the physical signal into that freedom and report a clean fit that means nothing.

The order is therefore fixed, and it is the reverse of the order a modeller reaching for capacity would choose. **The physical states are added first, at fixed linear gain, exactly as D5 and D6 specify. Only afterwards, and only on the model those ladders hand on, is the temperature gain allowed to vary.** Three specifications are compared on the leave-one-year-out folds of D1:

1. the surviving model of D5 and D6, fixed linear gains, which is the base;
2. the same with air temperature entering as two conditional regressors weighted by the annual curve Study 05 measured for its conditional daily term, `GM_04`'s order-two fit normalised to the unit interval — the same mechanism Study 05 built and rejected for the daily *shape*, applied here to the driver's *gain*;
3. the same with `future_regressors_model='neural_nets'`, which replaces every additive regressor's linear coefficient with a small multilayer perceptron and is the library's own answer to a nonlinear response.

Each reports leave-one-year-out MAE with the paired block-bootstrap skill over the base, the summer and winter gains where they are separable, the share of variance held by the moisture and exchange states in each specification, and the annual-sideband power of the residual.

**The rule, stated in advance.** A varying gain counts as a finding only if its skill bounds over the base exclude zero *and* the moisture and exchange states keep the share and the sign they held in the base. If skill improves while a physical state's share collapses, the reading is that the varying gain has absorbed that state, the state's own verdict from D5 or D6 stands as reported, and the report says which state was absorbed and by how much. The main line of the study remains the fixed-gain model in every case; a nonlinear specification is a diagnostic, never the model whose components Section 6 of the report tables, for the same reason D7 of Study 05 kept autoregression out of the decomposition.

### D12 · The capacity control: how much skill the additive linear grey box leaves on the table

Every model in this study and in Study 05 is additive, and every gain but D11's is linear. Nothing in either study measures what that costs. A reader is entitled to ask whether the grey box's interpretability is bought with skill, and neither study can currently answer, which is a gap in the argument rather than in the results.

One unconstrained model is therefore fitted, on the same rows and scored on the same leave-one-year-out folds, purely to bound the answer: gradient-boosted trees on the lagged driver matrix, the machinery Notebook 03 of the production pipeline already carries, with lags to twelve hours per driver, day of year and time of day as features, and no structure imposed. It is a ceiling, not a candidate. It has no components, no gains, no interval this study would trust and no place in the monitor.

**What the number means, stated before it is measured.** If the ceiling beats the base by a margin whose bootstrap bounds include zero, or by a few per cent, the additive grey box is not leaving meaningful skill on the table and that is a result the paper should report, because it converts an assumption into a measurement. If the ceiling beats it by a large margin, something structural is missing from the specification, the study says so, and identifying it is the next study's question rather than this one's. Either outcome is reportable; neither changes the main line, and the ceiling is never fitted before D5, D6 and D11 have run, so that it cannot influence a decision it is meant to bound.

The same table carries a second row for context: Study 05's Model A on this study's folds, so that the ceiling is read against the model this study started from as well as against the model it ends with.

---

## 4 · Metrics — what is measured, and why that metric

| Metric | Why it is here |
|---|---|
| Annual precipitation per hydrological year and its anomaly, per source | The interannual variation the moisture hypothesis needs; if the years do not differ, the test has no power |
| Day of year of the filtered precipitation's peak and trough, per τ | The phase test against Study 05's yearly term, before any model |
| Slope of trend-segment rate on API change, per τ | Whether the reversals follow the rain |
| Leave-one-year-out MAE and paired block-bootstrap skill, per rung and per changepoint count | The judge of every decision; bounds excluding zero is the rule |
| Learned gain per state with its sign, per fold | Sign stability is the finding-or-not rule inherited from Study 05 |
| Yearly term peak-to-peak per held-out year, with and without the moisture state | Thermal delay repeats; moisture does not |
| Variance share per component beside Study 05's | How much of the trend became explained |
| Radiation gain and sign per rung of the exchange ladder | The one number Study 05 got wrong, tracked at every step |
| Residual lag-one autocorrelation and annual-sideband power per rung | Whether the exchange states absorb the seasonally modulated daily cycle |
| Slow-chart episodes on the joint residual beside Study 05's | What the monitor would watch |
| Annual modulation of each driver's daily amplitude, December against July | Whether any driver in the record carries the wall's 6.1 (D11, §2.6) |
| Leave-one-year-out skill of the two varying-gain specifications over the base | Whether the fixed linear gain is the limiting assumption (D11) |
| Moisture and exchange share and sign under each varying-gain specification | Whether a varying gain absorbs a physical state rather than adding to it (D11) |
| Leave-one-year-out MAE of the unconstrained model, against the base and against Study 05's Model A | The ceiling: how much skill the additive linear specification leaves on the table (D12) |

---

## 5 · Architecture and the `shmlib` contract

Every function lives in `studies/shmlib/`. Nothing is written into the study folder. The notebook declares every path, parameter and choice in its parameter cell and passes them as arguments. What exists and what must be written is confirmed at Phase 0 by a `/graphify` query, not from memory.

### 5.1 Reused unchanged

`proxies.load_era5`, `load_ground_station`, `load_response`, `load_sensor_forcings`, `join_eras`, `harmonise`, `to_native_grid`, `fill_short_gaps`; `coupling.thermal_operator`, `couple`, `annual_modulation`; `prediction.covered_changepoints`, `decompose_components`, `component_variance_shares`, `residual_diagnostics`, `score_predictions`, `paired_mae_skill`, `period_scan`, `regressor_gains`, `trend_parameters`, `seasonal_parameters`; `monitoring.reference_stats`, `cusum_chart`, `alarm_episodes`; `figures.plot_decomposition_stack`, `plot_regressor_gains`, `plot_control_chart`, `plot_ladder`; `viz.*`; `tables.write_table`.

### 5.2 Adapted — additive only; every existing caller keeps its behaviour, and the tests of Studies 03 to 05 must pass unchanged

| Function | Change |
|---|---|
| `proxies.load_era5` | Channel map extended with `lw_down` (`surface_thermal_radiation`), `snow` (`snowfall`); default column set unchanged |
| `coupling.thermal_operator` | Optional `modulation` argument, a series that scales the time constant slot by slot; default `None` reproduces today's behaviour |
| `prediction.neuralprophet_backtest`, `rolling_nowcast` | Optional `folds='year'` selecting leave-one-year-out; default unchanged |
| `prediction.neuralprophet_backtest` | Optional `future_regressors_model='linear'`, `future_regressors_d_hidden=4`, `future_regressors_num_hidden_layers=2`, `lagged_reg_layers=()`, passed through to the `NeuralProphet` constructor, which today builds a fixed dict that exposes none of them (`prediction.py:1104`). The defaults are the library's own, so every existing caller fits the identical model it fits now; the check is that Studies 03 to 05 reproduce their tables bit for bit. Needed by D11 (D12 does not use NeuralProphet) |

### 5.3 New in `shmlib`

```
coupling.antecedent_index(precip, tau_days, freq)            # causal exponential filter of precipitation on its own grid
coupling.net_longwave(lw_down, temperature, emissivity=0.9)  # downward minus computed emission, W/m²
coupling.segment_rate_regression(rates, state, segments)     # slope of trend-segment rate on state change, per τ
coupling.amplitude_envelope(series, freq='M')                # monthly mean of the daily harmonic amplitude, and its ratio (D11, §2.6)
prediction.year_folds(frame, years)                          # leave-one-year-out fold indices on covered years
prediction.yearly_amplitude_by_fold(models, folds)           # yearly term peak-to-peak per held-out year
prediction.gain_variation_ladder(frame, base, specs, folds)  # the three specifications of D11, scored on one fold set
prediction.capacity_ceiling(frame, folds, lags, seed)        # the unconstrained gradient-boosted fit of D12
proxies.load_soil_moisture(path, layers)                     # ERA5-Land volumetric soil water, if obtained
figures.plot_moisture_state, plot_trend_rate_regression, plot_yearly_by_year, plot_exchange_ladder
figures.plot_driver_envelope, plot_capacity_ceiling
```

`coupling.amplitude_envelope` is a thin wrapper over `compare.daily_amplitude_phase`, which already fits the daily harmonic day by day; it exists so that the same reduction is not written twice, once in Phase 0 and once in Phase 3b. `prediction.capacity_ceiling` is the only function in this study that imports `xgboost`, which the environment already carries for the production pipeline's Notebook 03; it must not import NeuralProphet, so that a failure of one cannot be read as a failure of the other.

Each new function carries a NumPy-style docstring and a unit test written before the implementation, one test file per `shmlib` area under `tests/`, plus a folder-honesty test on the model of Study 04's.

---

## 6 · Artefacts

Tables in `outputs/`, LaTeX bodies alongside:

| File | Content |
|---|---|
| `HT_01_source_agreement.csv` | Station rain, wind against ERA5 on the overlap: bias, MAE, r, phase |
| `HT_02_annual_water.csv` | Precipitation per hydrological year and anomaly, per source; snow share |
| `HT_03_state_phase.csv` | Day of year of peak and trough of API(τ) and of soil moisture, per τ and layer, beside Study 05's yearly term |
| `HT_04_trend_rate_regression.csv` | Slope, interval and sign consistency of segment rate on API change, per τ |
| `HT_05_moisture_ladder.csv` | Leave-one-year-out MAE and skill per τ and changepoint count; gain and sign per fold |
| `HT_06_yearly_by_year.csv` | Yearly term peak-to-peak per held-out year, with and without the moisture state |
| `HT_07_exchange_ladder.csv` | The heat-exchange ladder, rung by rung, with the radiation gain and sign |
| `HT_08_wind_modulation_sweep.csv` | Skill per k |
| `HT_09_joint_shares.csv` | Component shares and gains of the joint fit beside Study 05's, per set |
| `HT_10_joint_residual.csv` | Residual diagnostics of the joint fit beside Study 05's |
| `HT_11_slow_chart_episodes.csv` | Slow-chart episodes on the joint residual beside Study 05's `GM_11` |
| `HT_12_run_metadata.csv` | Every parameter, seed and library version, with the Study 05 artefact each inherited value points to |
| `HT_13_driver_envelope.csv` | Monthly mean daily amplitude per driver and per source, with the December-to-July ratio, beside the target's (D11, §2.6) |
| `HT_14_gain_variation.csv` | The three specifications of D11: leave-one-year-out MAE, skill over the base with bounds, summer and winter gains, moisture and exchange shares, residual sideband power |
| `HT_15_capacity_ceiling.csv` | The unconstrained model of D12 against the base and against Study 05's Model A, on the same folds |

Figures, each written as PNG and SVG:

| Figure | Content |
|---|---|
| `HT_F01` | Precipitation, API at three τ, soil moisture if obtained, and Study 05's trend on one clock |
| `HT_F02` | Phase of the filtered precipitation through the year beside Study 05's yearly term |
| `HT_F03` | Trend-segment rate against API change, per τ |
| `HT_F04` | The moisture ladder: held-out MAE by changepoint count, with and without the state |
| `HT_F05` | Yearly term per held-out year, with and without the moisture state |
| `HT_F06` | The heat-exchange ladder, rung by rung, with the radiation gain beside it |
| `HT_F07` | The joint decomposition stack beside Study 05's |
| `HT_F08` | The slow chart on the joint residual with Study 05's episodes marked |
| `HT_F09` | The annual envelope: monthly mean daily amplitude of each driver and of the target, normalised to its own December value, on one axes |
| `HT_F10` | The capacity ladder: leave-one-year-out MAE of Study 05's Model A, this study's base, the two varying-gain specifications and the unconstrained ceiling, with bootstrap bounds |

---

## 7 · Report structure

Study 05's shape: every decision explained where it is made, before the result that depends on it. Target length 12 to 16 pages.

1. **Introduction** — what Study 05 left without a driver; what Study 03 ruled out and why that ruling does not cover the states.
2. **The sources and the states** — the channels, their agreement on the overlap, the annual water budget, the phase of the filtered precipitation against the yearly term, `HT_F01`, `HT_F02`.
3. **Method** — one decision per subsection, D1 to D10.
4. **Moisture against the trend** — the three tests of D5, `HT_F03` to `HT_F05`.
5. **Heat exchange** — the ladder of D6, `HT_F06`.
6. **What the linear gain assumes, and what it costs** — the envelope of §2.6 and `HT_F09`; the three specifications of D11 and `HT_14`; the ceiling of D12 and `HT_F10`. Placed after both ladders and before the joint fit, in the order the study runs them, so that a reader sees the physical states judged at fixed gain before any model is given more freedom.
7. **The joint decomposition** — `HT_F07`, beside Study 05's.
8. **What the slow chart would watch** — `HT_F08`.
9. **Verdict** — one paragraph per question, each with its qualification.
10. **Limitations** — one gauge and one reanalysis cell for the rain; the embankment's own moisture never measured; five annual cycles; the emissivity assumed; the wall's azimuth still unrecorded; and the ceiling of D12 measured with one model family on one feature set, which bounds the skill left on the table from below and not from above.
11. **Run metadata.**

Every number in the prose traces to a named `HT_` artefact.

---

## 8 · Phases and checkpoints

Orchestrator (O) plans, reviews and writes prose; subagents (S) implement under written instruction. Each checkpoint is shown to the user and approved before the next phase starts.

### Phase 0 · Sources, semantics and the design table — S runs, O decides
Confirm Study 05's D15 result and the operator handed on. Settle the meaning of `surface_thermal_radiation` against `docs/proxy-data-dictionary.md` and the provider's documentation. Attempt the ERA5-Land soil-moisture download and record the outcome. Establish whether the station gauge is heated. Score station rain and wind against ERA5 on the overlap. Produce the table of §2.5. Run `/graphify` on `studies/` to confirm §5's inventory.
> **Checkpoint 0:** the base operator named; each channel's semantics settled; soil moisture obtained or its absence recorded; the phase of the filtered precipitation stated beside day 80 and day 263. **If the annual precipitation totals do not differ between years by more than their own uncertainty, the moisture question cannot be answered on this record, and the study is reduced to its heat-exchange half before any model is fitted.**

### Phase 1 · The moisture states and the trend, descriptively — S runs, O interprets
API at every τ, soil moisture if obtained, the annual water budget, the trend-rate regression. `HT_01` to `HT_04`, `HT_F01` to `HT_F03`.
> **Checkpoint 1:** the sign and consistency of the rate regression per τ; the τ range worth carrying into the model, or none.

### Phase 2 · Moisture in the model — O designs, S runs the sweeps
The changepoint-by-τ sweep on leave-one-year-out folds; gain and sign per fold; yearly amplitude per held-out year. `HT_05`, `HT_06`, `HT_F04`, `HT_F05`.
> **Checkpoint 2:** the three-part rule of D5 applied and its outcome stated condition by condition.

### Phase 3 · The heat-exchange ladder — S runs, O interprets
`HT_07`, `HT_08`, `HT_F06`.
> **Checkpoint 3:** a verdict per exchange; the radiation gain's sign at every rung.

### Phase 3b · The gain assumption and the capacity ceiling — S runs, O interprets
Runs only after Phases 2 and 3 have delivered their verdicts, so that no physical state is judged against a model with more freedom than the one its own rule was written for. The three specifications of D11 and the unconstrained fit of D12, on the folds of D1. `HT_13`, `HT_14`, `HT_15`, `HT_F09`, `HT_F10`.
> **Checkpoint 3b:** whether a varying temperature gain earns its bounds; whether any physical state's share collapsed when it was allowed, and which; the size of the gap between the base and the ceiling, stated as a percentage with its bounds. **If a varying gain absorbs a state that Phase 2 or Phase 3 had passed, that state's verdict is not revised — the absorption is reported as the finding, and the report carries both readings.**

### Phase 4 · The joint fit and the slow chart — S runs, O interprets
Only for what survived Phases 2 and 3, at the fixed linear gain of the main line whatever Phase 3b found. `HT_09` to `HT_11`, `HT_F07`, `HT_F08`.
> **Checkpoint 4:** shares beside Study 05's; what the slow chart would watch.

### Phase 5 · The report — O only, never delegated
> **Checkpoint 5:** every number traces to a named file; the README matches the folder; the report builds twice cleanly; the folder-honesty test passes; `studies/README.md` carries the study's line.

---

## 9 · Risks and kill criteria

| Risk | Detection | Response |
|---|---|---|
| Annual rainfall does not vary enough between the five years to separate moisture from thermal delay | Checkpoint 0 | Moisture half dropped before any fit; stated as the record's limitation, not the wall's |
| ERA5-Land soil moisture cannot be obtained | Phase 0 | API stands alone; the limitation is stated; the download is listed as the one step that would close it |
| `surface_thermal_radiation` turns out to be net rather than downward | Phase 0 | The emission computation is skipped and the channel used as it is; the parameter cell records which |
| The moisture state and the trend are collinear at every changepoint count | Checkpoint 2 | The rate regression of D5(1) and the yearly-amplitude test of D5(3) are the evidence; the model comparison is reported as inconclusive |
| The moisture gain has the sign opposite to the earth-pressure prediction | Checkpoint 2 | Reported as a finding about mechanism; never tuned away |
| Wind modulation makes the filter unstable at high k | Phase 3 | k capped where τ(w) falls below one slot; the cap stated |
| The exchange ladder shows zero skill at every rung, as the wall probe did | Checkpoint 3 | Expected outcome; the report states it and keeps the radiation-gain sign as the finding |
| Leave-one-year-out folds leave too little training history for the 2019 fold | Phase 2 | The 2019 fold is dropped and the study says the test rests on four years |
| A varying temperature gain absorbs the moisture or exchange state that had just passed its rule | Checkpoint 3b | Expected and provided for: the state's verdict stands as Phase 2 or 3 reported it, the absorption is the finding, and the report carries both readings rather than choosing between them |
| The neural-net regressor specification fails to converge, or its held-out error is wildly unstable across folds on five annual cycles | Phase 3b | Reported as a negative result about the data budget rather than about the wall; the conditional two-gain specification is the one D11 rests on, and the neural-net row is dropped with its reason stated |
| The unconstrained ceiling beats the base by a large margin | Checkpoint 3b | Reported, not chased. The gap is stated with its bounds and the identification of what is missing is named as the next study's question; the main line does not change |
| The ceiling is fitted before the ladders and influences a decision it exists to bound | Phase ordering | Forbidden by D12 and by the phase order; the notebook raises if `HT_15` is written before `HT_05` and `HT_07` exist |

---

## 10 · Decisions taken and decisions open

Taken in conversation on 2026-09-07:

1. **Scope** — a new Study 06; Study 05 untouched and not re-run; Study 05's D15 sweep is the entry condition.
2. **Form of the drivers** — states with time constants, never readings; Study 03's verdict on the raw readings inherited.
3. **Two independent ladders** — moisture against the trend, heat exchange against air temperature; a joint fit only for what survives.
4. **Validation** — leave-one-year-out over whole years, beside Study 05's chronological folds.
5. **Pre-stated rules** — bootstrap bounds excluding zero; sign stable across folds; the expected sign of the moisture gain positive by the earth-pressure mechanism, stated before the fit.
6. **The monitor** — not re-run; the slow chart's view of the joint residual reported as a consequence.

Added on 2026-09-07 after the conversational measurement of §2.6:

7. **The linear gain is a choice, and the study measures it rather than assuming it** (D11). NeuralProphet 0.8.0 is not restricted to linear regressor gains — `future_regressors_model`, `future_regressors_d_hidden`, `future_regressors_num_hidden_layers`, `lagged_reg_layers` and `ar_layers` all exist and are unused by Studies 04 and 05, which chose linear deliberately and did not record that it was a choice. This study records it and tests it.
8. **Order is fixed: physical states first at fixed gain, freedom afterwards** (D11). A model given more freedom before the states are added absorbs the physical signal into that freedom.
9. **The main line stays fixed-gain and additive whatever Phase 3b finds** (D11), on the precedent of Study 05's D7 keeping autoregression out of the decomposition. A nonlinear specification is a diagnostic.
10. **The capacity ceiling is measured once, as a bound, and is never a candidate** (D12).

Open, for approval:

11. **Folder name and prefix** — `06_hydro_thermal_drivers/` with prefix `HT_` is proposed.
12. **Soil moisture download** — whether to spend the provider request on ERA5-Land layers now or run the API alone first.
13. **Emissivity** — 0.9 is proposed for limestone masonry; a different value, or a sweep, is the user's call.
14. **Regressor sets** — the on-structure set for every ladder and the station and ERA5 sets for the joint fit only, as proposed; or all three sets throughout at three times the compute.
15. **Scope of Phase 3b** — both specifications of D11, the conditional two-gain and the neural-net regressors, as proposed; or the conditional one alone, which is cheaper, keeps the gains readable and is the specification the rule actually rests on, with the neural-net row dropped.
16. **The ceiling's model family** (D12) — gradient-boosted trees on lagged features, as proposed, reusing the production pipeline's Notebook 03 machinery; or a second family beside it, which would bound the answer better and costs another implementation.
17. **Whether Phase 3b belongs in this study at all** — it answers a question about method rather than about the wall, and it would sit equally well in the paper's model-choice section or in a short note of its own. Keeping it here is proposed because the folds, the base model and the artefacts already exist at that point in the sequence, and splitting it would mean building all three again.
