# Design · Proxy-to-target correlation and the solar-radiation comparison

**Study:** `studies/proxy_comparison/`
**Date:** 14 August 2026
**Status:** design approved, implementation not started

---

## 1. What this design changes, and why it is needed

The proxy-comparison study already compares the three environmental sources against one
another. It also already measures each source against the calibrated inclination
`inc_comp` in levels and in first differences, through `pc.target_relationship`, and it
writes the result to `outputs/P_08_target_relationship.csv`. That table carries all three
canonical quantities — air temperature, relative humidity and solar radiation — in both
domains. Only the air-temperature rows reach the report; the humidity and radiation rows
are computed and then discarded.

Surfacing those rows is not sufficient, because the way they are computed is not the way
the sibling studies compute the same quantity, and the two methods disagree by a wide
margin.

`pc.target_relationship` is applied to the whole joined frame. That frame spans 26 July
2018 to 13 August 2026 and contains both instrument eras: the legacy package, and the
package physically reinstalled at station `st02` on 21 February 2025. It is also ragged,
with the on-structure channels present on roughly two thirds of the grid. The sibling
studies `inclination_prediction` and `inclination_prediction_legacy` never correlate
against the target on a span of that kind. They cut contiguous blocks of near-complete
coverage, group those blocks into tiers according to which channels the tier requires, and
report every correlation, slope and operator within a tier.

The consequence is large. For the same target at the same station, against the same
on-structure air temperature:

| Method | levels *r* | first-difference *r* |
|---|---|---|
| `inclination_prediction`, Tier 1 blocks, current era | −0.956 | −0.886 |
| `proxy_comparison`, pooled 2018–2026 span | −0.524 | −0.155 |

For solar radiation the pooled computation does not merely attenuate the relationship, it
reverses its sign: `sr_str` returns *r* = +0.087 in first differences, while
`inclination_prediction` returns −0.624 on its Tier 1 blocks and both proxies return
approximately −0.09 on the pooled span.

Section 5 of the current report therefore states that the differenced relationship is weak
for every source, with *R²* between 0.007 and 0.024, and attributes that weakness to
physics. The comparison above indicates the weakness is at least partly a property of the
window rather than of the wall. This design replaces the pooled computation with the
block-and-tier treatment the sibling studies use, and extends the whole target-relationship
analysis to solar radiation on the same footing as air temperature.

### 1.1 One assumption this design must verify before it interprets anything

`inclination_prediction` builds its target through `ip.add_target`, applying the
compensation coefficient inside its own current-era loader. The proxy-comparison study
takes `inc_comp` from `ud.load_unified`. The two targets are believed to be identical, but
the discrepancy in section 1 is attributed to windowing only if they are. The first
implementation step is a direct numerical comparison of the two series over their common
index. If they differ, the difference is characterised and reported before any conclusion
in this design is drawn.

---

## 2. Decisions taken

**The target is `inc_comp` and nothing else.** The compensated inclination is the ground
truth of this project. No alternative target is constructed, the compensation coefficient
is not re-estimated, and the report carries no discussion of the on-structure temperature
appearing inside the calibrated reading. This preserves the standing rule of the study.

**Windows are contiguous blocks, grouped into tiers.** No statistic in the new analysis is
computed on the pooled ragged span, and no window crosses the era boundary of 21 February
2025.

**Block parameters are the sibling defaults**, `min_days=20` and `max_gap_hours=6`, so
block inventories are directly comparable with those in `inclination_prediction`.

**Daylight is defined by solar geometry.** The mask is solar elevation above the horizon,
computed from the site coordinates. It is independent of any of the three sources being
compared, it is defined at every timestamp including those where a source is missing, and
the identical mask applies to all three channels.

**Existing artefacts are not renumbered.** New tables take `P_13` onward and new figures
`P_F06` onward.

---

## 3. Site coordinates and their provenance

The site coordinates are already present in `pc_lib` as `SITE_LATITUDE = 43.35343` and
`SITE_LONGITUDE = 12.582047`. They originate in `auxiliary/oiko.py`, lines 84 and 85,
where they are the `LAT` and `LON` arguments of the Oikolab API request that produced
`data/raw/proxies/oikolab_weather.csv`. The two pairs of values are identical; the
provenance was simply never written down.

Because the daylight mask makes these constants load-bearing for the first time, the
provenance is recorded in two places:

1. A comment on the constants in `pc_lib`, naming `auxiliary/oiko.py` as the origin and
   stating that they define the ERA5 extraction point as well as the solar geometry, so
   the two can never be changed independently.
2. A **Site** block emitted into `docs/proxy-data-dictionary.md` by
   `pc.write_data_dictionary`, giving latitude, longitude, and the origin. The data
   dictionary is generated rather than hand-written, so this record cannot drift away from
   the code.

---

## 4. Tiers

The era boundary is 21 February 2025. Blocks are found with
`ip.contiguous_blocks(df, cols, min_days=20, max_gap_hours=6)`, and the analysis window of
a tier is the union of that tier's blocks.

| Tier | Columns required simultaneously | Era |
|---|---|---|
| **P0** | `inc_comp`, `tair_str`, `sr_str`, `tair_gs`, `sr_gs`, `tair_era5`, `sr_era5` | current |
| **P1** | `inc_comp`, `tair_gs`, `sr_gs`, `tair_era5`, `sr_era5` | current |
| **P2** | `inc_comp`, `tair_gs`, `sr_gs`, `tair_era5`, `sr_era5` | legacy |

This is the sibling studies' Tier 1 / Tier 2 construction applied to the binding constraint
of this study. In `inclination_prediction` the binding channel is the wall probe; here it
is the on-structure pyranometer.

The radiation record settles the shape of the tiers by itself. `sr_str` begins on
21 February 2025 — the era boundary — and covers 11.8 % of the joined span, against 88.7 %
for `sr_gs` and 99.5 % for `sr_era5`. The three-source radiation comparison therefore
lives entirely inside the current era, and the two proxies additionally reach into a legacy
era the on-structure channel can never enter. Tier P0 is where the three sources compete;
tiers P1 and P2 are where the proxies are measured over the long windows only they possess.

Tiers P0 and P1 overlap by construction, so agreement between them is weaker evidence than
between independent samples. This is stated in the report, following the same caution
`inclination_prediction` applies to its own overlapping tiers.

### 4.1 What a tier does and does not determine

A tier is defined by air temperature and solar radiation only, because those are the two
quantities this design analyses against the target and the radiation channel is what
constrains the windows.

**Relative humidity does not define any tier.** It is reported inside a tier's window
wherever it is present, and its absence never splits a block. The study's standing
conclusion that humidity must not be substituted between sources is unchanged by this work.

**The on-structure channels are absent from tiers P1 and P2 by construction.** Every
per-source table therefore carries three rows in tier P0 and two in tiers P1 and P2. Every
correlation matrix, operator table and partial fit below is built from whichever source
channels the tier's window actually contains, and no row is fabricated for a source the
tier excludes.

**Guard.** Every tier must contain at least one block of at least `MIN_OVERLAP_DAYS`
(30 days). A tier that fails is reported as unavailable rather than being computed on
insufficient data.

**Artefact.** `P_13_block_inventory.csv` — one row per block, with tier, era, start, end,
days and coverage percentage.

---

## 5. Correlation structure

For each tier, the correlation matrix over `inc_comp` and the six source channels
(`tair` and `sr` for `str`, `gs` and `era5`), computed twice with
`tc.correlation_matrix(df, cols, differenced=False)` and `differenced=True`.

Both matrices are reported side by side, with the reading rule carried over from
`inclination_prediction`: correlations between two trending series are inflated by the
shared trend, so the differenced matrix is the honest one for a signal with drift, and the
pair is shown so the gap is visible.

**Artefacts.** `P_14_corr_levels.csv`, `P_15_corr_differences.csv`, and figure
`P_F06_correlations.png` / `.svg` via `tc.plot_correlation_heatmaps`, one heatmap pair per
tier.

---

## 6. Relationship with the target, per tier

`pc.target_relationship` gains an optional `blocks` argument restricting the computation to
a tier's blocks. The default is `None`, which preserves the current behaviour exactly, so
the existing pooled call and its artefact `P_08` continue to work unchanged.

The argument is blocks rather than a plain index because **a first difference must never be
taken across a block boundary.** Two rows on either side of a boundary are both valid and
both present, but they are not adjacent in time, and differencing them manufactures a step
out of a gap. Level statistics take the union of a tier's blocks; differenced statistics are
computed inside each block and concatenated afterwards. The same constraint applies to
`tc.correlation_matrix`, which calls `df[cols].diff()` unguarded and therefore must be fed a
frame that has already been differenced block by block.

The function is then called once per tier, for each canonical quantity in
`['tair', 'rh', 'sr']`, retaining the existing outputs: *n*, overlap in days, *r*, slope in
mdeg per unit, Newey–West standard error at 24 lags, *t*, and *R²*.

The pooled `P_08` is retained and shown once in the report, as the contrast that shows what
pooling the eras and the ragged span costs. Every other number in the section comes from a
tier.

**Artefacts.** `P_16_target_relationship_tiers.csv`, and figure `P_F07_relationship_tiers`
extending `pc.plot_target_relationship` with one facet per tier, error bars from the
Newey–West standard errors.

---

## 7. Operators, on two bands

For each tier, `ip.operator_table` is run over the six source channels as drivers against
`inc_comp`, with the study's existing grid: delays `0…12` hours, time constants `TAUS`,
hourly step.

**The scan runs on the tier's longest single block, not on the union.** A transport delay is
applied with `.shift()` and a time constant with a recursive filter, and the band-limit step
uses a one-week rolling mean; every one of those operations reads across adjacent rows, so a
union of blocks would let a filter draw values from the far side of a multi-month
interruption. The identity of the block used, and its length, are recorded in the output
table so the window behind every operator is visible.

It is run twice, following `inclination_prediction`:

- **full band**, `detrend_hours=None`;
- **diurnal band**, `detrend_hours=168`, a one-week centred rolling mean removed.

Both are reported and the diurnal-band operators are the ones carried forward. The sibling
study's result on the on-structure channels predicts what to expect: on the full band,
solar radiation's optimum pins at the last time constant in the grid, τ = 168 hours,
lifting explained variance from 0.229 to 0.662, which is a filter converting a daily
radiation cycle into a seasonal envelope rather than a thermal property of masonry. On the
diurnal band the same driver optimises at zero delay and a one-hour time constant. Whether
the proxies reproduce this is one of the questions the new section answers.

The twelve-hour delay bound is unchanged and remains justified by
`docs/raw-data-format.md` §7.5: both air temperature and solar radiation are external
forcings, and beyond half a diurnal cycle a delay is indistinguishable from a lead.

**Artefacts.** `P_17_operators_fullband.csv`, `P_18_operators_diurnal.csv`, and figure
`P_F08_operators` via `ip.plot_operator_heatmaps`.

These supersede the existing `P_09` and `P_F04`, which cover air temperature on one band
only. `P_09` and `P_F04` are removed from the notebook once `P_17`/`P_18` are in place, and
the artefact inventory in the report is updated accordingly.

---

## 8. Signed-delay control on solar radiation

`tc.cross_correlation` followed by `tc.peak_lags`, over signed delays from −12 to +12 hours,
on the diurnal band, for `sr` from each source against `inc_comp` within tier P0.

Solar radiation is an external forcing. It must precede the response it causes, so a
negative optimum is not a physical lag but a signature of a defect in the channel. This is
the correct test of the radiation anomaly the study has already documented from two other
directions — the non-zero night-time floor and the impossible solar-noon phase — and it is
run on the one tier where all three sources coexist, which removes the window as an
explanation.

**Guard.** The diurnal-band optimum for `sr_gs` and `sr_era5` must be at a non-negative
delay. A violation on a proxy indicates a problem in the harmonisation or the clock
correction rather than in the wall, and the notebook asserts it.

**Artefact.** `P_19_signed_delay_sr.csv`.

---

## 9. What solar radiation adds over air temperature

Air temperature and solar radiation are strongly collinear, so a marginal correlation
between radiation and the target largely re-measures the temperature relationship. Two
steps separate them, within each tier and for each source:

1. `ip.vif_table` on the pair `[tair_<source>, sr_<source>]`, quantifying the duplication.
2. `tc.multivariate_fit(df, 'inc_comp', [tair_<source>, sr_<source>], hac_lags=24)`, giving
   the partial slopes with Newey–West standard errors, and the increment in *R²* over the
   temperature-only fit.

This answers the question the downstream imputation study inherits: whether a radiation
channel is worth carrying as a regressor once temperature is present.

**Artefact.** `P_20_radiation_contribution.csv`.

---

## 10. Solar radiation: availability and the daylight variant

### 10.1 Availability

The availability computation currently applied to air temperature — coverage of the archive
span, and presence during the hours the inclinometer is missing — is extended to the
radiation channels. The existing temperature result is 0.0 % for the on-structure sensor,
94.3 % for the ground station and 99.6 % for ERA5. The radiation equivalents are new and
are required before any radiation channel can enter the feature specification.

### 10.2 The daylight mask

Roughly half of every radiation record is a structural night-time zero. Those zeros are
real measurements and are never removed, but they distort every agreement statistic:
correlations are inflated by a block of samples on which all sources trivially agree,
Bland–Altman limits are computed over a bimodal distribution, and the trough hour of the
mean diurnal cycle is meaningless — `P_06` currently reports 23, 1 and 2 for the three
sources, which are arbitrary points inside the night.

A new function `pc.daylight_mask(index)` returns a boolean mask for solar elevation above
the horizon, computed from `SITE_LATITUDE` and `SITE_LONGITUDE` by extending the solar
geometry already present in `pc_lib` behind `solar_noon_utc`. Implementation is the standard
solar position calculation: day angle, solar declination, the equation of time, hour angle,
and elevation from the spherical law of cosines. The mask is source-independent and defined
on every timestamp.

Three statistics are then recomputed under the mask and reported beside their all-hours
counterparts, never in place of them: pairwise agreement (`pc.pairwise_agreement`),
first-difference agreement (`pc.difference_agreement`), and diurnal amplitude
(`pc.diurnal_comparison`).

**Artefacts.** `P_21_sr_daylight_agreement.csv` and `P_22_sr_availability.csv`.

---

## 11. Report structure

The report is `report/proxy_comparison_report.tex`. Changes, in document order:

**Section 3, Absolute agreement.** The solar-radiation block of the agreement table gains a
daylight-only counterpart, so the night-zero inflation is visible next to the all-hours
figures the report already carries (`str`–`gs` *r* = 0.744, `str`–`era5` *r* = 0.738,
`gs`–`era5` *r* = 0.930 at slope 0.998).

**Section 4, Dynamic agreement.** The diurnal table gains the solar-radiation rows that
already exist in `P_06` and are currently omitted: amplitudes 261.0, 243.0 and 282.6 W/m²,
ratios 1.00, 0.93 and 1.08, all three peaking at hour 12. The daylight-only amplitudes are
reported beside them, and the meaningless all-hours trough hours are dropped from the
radiation rows with a sentence saying why.

**Section 5, Relationship with the inclination.** Rewritten, in this order:

1. Blocks and tiers — the inventory table, and why the radiation record forces this shape.
2. Correlation structure — levels and differences, both matrices, one figure.
3. Relationship with the target — every quantity, every source, both domains, per tier.
4. Operators on two bands — the full-band boundary optimum and the diurnal-band result.
5. The signed-delay control on radiation.
6. What radiation adds over temperature — collinearity and the partial fit.
7. What the pooled span cost — the single contrast against `P_08`.

**Section 6, Verdict.** The availability table, the which-source-for-what table and the
feature specification each gain their radiation rows.

**Section 7, Caveats.** Three additions: tier P2 measures a different physical instrument
from tiers P0 and P1 and its results do not transfer across the reinstallation; the
daylight mask is geometric and takes no account of cloud or of the local horizon formed by
surrounding terrain; tiers P0 and P1 overlap and are not independent samples.

**Artefact inventory.** Extended with `P_13`–`P_22` and `P_F06`–`P_F08`, with `P_09` and
`P_F04` removed.

The abstract and the README summary are revised last, once the numbers exist. Both
currently quote pooled-span figures for the target relationship and will be wrong until
then.

---

## 12. Notebook structure

New and changed steps in `proxy_comparison_study.py`, which is the file edited; the paired
`.ipynb` is generated by `jupytext` and is never edited directly.

| Step | Content |
|---|---|
| 4a (new) | Daylight mask and the recomputed agreement statistics; `P_21` |
| 5 (existing) | Retained, producing the pooled `P_08` as the contrast case only |
| 5a (new) | Target verification against `ip.add_target`; era split; tier definition; blocks; `P_13` |
| 5b (new) | Correlation matrices; `P_14`, `P_15`, `P_F06` |
| 5c (new) | Target relationship per tier; `P_16`, `P_F07` |
| 5d (new) | Operator scans, both bands; `P_17`, `P_18`, `P_F08`; removes the `P_09`/`P_F04` cells |
| 5e (new) | Signed-delay control; `P_19` |
| 5f (new) | Collinearity and partial contribution; `P_20` |
| 6 (existing) | Extended: radiation availability `P_22`, radiation rows in the verdict and in `P_12` |

Each new step follows the study's existing convention: a Markdown header cell describing
what the step does and documenting its parameters, then code cells that call library
functions rather than implementing logic inline.

---

## 13. New code in `pc_lib.py`

| Function | Purpose |
|---|---|
| `solar_elevation(index, latitude, longitude)` | Solar elevation angle per timestamp |
| `daylight_mask(index, ...)` | Boolean mask, elevation above the horizon |
| `tier_blocks(df, tiers, era_boundary, ...)` | Blocks per tier, stacked into one inventory |
| `tier_index(blocks, tier)` | The union index of a tier's blocks |

Changed:

| Function | Change |
|---|---|
| `target_relationship` | New optional `index` argument; default `None` preserves current behaviour |
| `write_data_dictionary` | Emits the Site block described in section 3 |

Everything else composes functions that already exist in `ip_lib` and `tc_lib`, both of
which `pc_lib` already imports. No new statistical machinery is written.

---

## 14. Out of scope

- The plaintext Oikolab API key committed in `auxiliary/oiko.py`. It is a real credential
  exposure, it is reported separately, and it is not touched by this work.
- Relative humidity beyond surfacing the rows that already exist in `P_08` and `P_16`. The
  study's standing conclusion that humidity must not be substituted between sources is
  unchanged.
- Any station other than `st02`.
- Any change to the compensation coefficient, to the target, or to the raw archive.

---

## 15. Verification

The notebook carries assertions, in the study's existing style, for each of:

- the two constructions of `inc_comp` agree, or their disagreement is characterised;
- every tier holds at least one block of at least 30 days;
- no tier window crosses 21 February 2025;
- the diurnal-band delay optimum for `sr_gs` and `sr_era5` is non-negative;
- `inc_comp` is present and unmodified, as the existing target guard already checks.

The report is verified statically, as the project requires: figure files present in
`outputs/`, no undefined labels or references, and every number in the prose traceable to a
named artefact. The user compiles the PDF.
