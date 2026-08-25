# Raw Sensor File Format (`.adc`) — Gubbio Medieval Walls

Specification of the raw acquisition files produced by the Gubbio monitoring system. This
document is the authoritative reference for any code that parses `.adc` files. It is
site-specific by nature and therefore belongs to Notebook 00 (`00_Sensor_Preprocessing`), which
is the only stage of the pipeline allowed to encode site detail; Notebooks 01–04 must never read
these files directly.

Last verified against the live archive on 2026-08-10. The empirical evidence behind every claim
below is recorded in [`data-quality-report-2026-08-10.md`](data-quality-report-2026-08-10.md).

**Revised 2026-08-21.** Sections 3.3, 6, 7.2 and 7.3 were corrected after `studies/01_data_exploration/`
measured the changeover on the recorded channel. Two claims that this document previously made are
now known to be wrong: that the two station 02 instruments share no baseline, and, by implication,
that the eras require an offset treatment before they can be joined. The step that motivated both
was produced by anchoring each instrument separately, not by the hardware. Section 6 also now
states explicitly that the inclinometer's calibrated range is not a rejection criterion.

---

## 1. Location of the archive

The raw files are **not** stored in this repository and are never committed. They are synchronised
from the acquisition system into a Google Drive folder that sits outside the repository tree:

```
~/Library/CloudStorage/GoogleDrive-<account>/My Drive/_UNIPG/__Mura-realtime/
```

The folder is flat: one file per calendar day, named `GUBBIO_YYYYMMDD.adc`, with no
subdirectories. Agent read access to this path is granted through
`additionalDirectories` in the parent folder's `.claude/settings.local.json`, together with an
explicit `deny` rule on `Write`, `Edit` and `NotebookEdit` for the same path. The archive is
strictly read-only: it is the acquisition system's output, and nothing in this project may modify
it.

Because the folder is served by Google Drive streaming, sequential reads run at roughly ten files
per minute. Any analysis touching more than a few dozen days must first copy the relevant files to
local disk; reading them in place from Drive is not practical.

The repository's own `data/raw/sensor/` directory (gitignored, created locally) holds whatever
subset of the archive a given analysis run works from.

---

## 2. Physical layout

| Property | Value |
|---|---|
| Encoding | Plain text, one record per line |
| Field separator | Tab (`\t`) |
| Header row | None |
| Field 1 | Date, `DD/MM/YY` |
| Field 2 | Time, `HH:MM:SS` |
| Fields 3… | Numeric measurements, see Section 3 |
| Nominal sampling interval | 20 minutes (72 records per day) |
| Decimal separator | **Both `.` and `,` occur** — see Section 4 |

A file therefore contains 72 records on a complete day. Files with fewer records are common and
indicate intra-day acquisition dropouts; see Section 5.

---

## 3. Column layout and its two eras

The measurement columns are organised as fixed-width blocks, one block per acquisition unit. Each
legacy unit contributes four channels in the order `Batt, Tair, RH, I`. The number of columns
changed exactly once, on **21 February 2025**, when a new instrument package was installed at
station 02.

### 3.1 Legacy era — 2018-07-26 through 2025-02-20 — 14 columns

```
 1  date        DD/MM/YY
 2  time        HH:MM:SS
 3  b1_Batt     [V]
 4  b1_Tair     [°C]
 5  b1_RH       [%]
 6  b1_I        [mdeg]
 7  b2_Batt     [V]
 8  b2_Tair     [°C]
 9  b2_RH       [%]
10  b2_I        [mdeg]
11  b3_Batt     [V]
12  b3_Tair     [°C]
13  b3_RH       [%]
14  b3_I        [mdeg]
```

### 3.2 Current era — 2025-02-21 onwards — 20 columns

The three legacy blocks are retained in their original positions but are **written as constant
zero on every record**; no legacy station reports data after the changeover. The new station 02
package occupies the six trailing columns and adds two channels that the legacy units did not
have, incident solar radiation and wall surface temperature.

```
 1  date        DD/MM/YY
 2  time        HH:MM:SS
 3–14           legacy blocks b1, b2, b3 — all identically zero in this era
15  n_Batt      [V]
16  n_Tair      [°C]
17  n_RH        [%]
18  n_I         [mdeg]
19  n_SR        [W/m²]
20  n_Twall     [°C]
```

### 3.3 Mapping blocks to station identifiers

The blocks appear in station order. This mapping is established from the installation record and
is not inferable from the file itself:

| Block | Columns (legacy era) | Station |
|---|---|---|
| b1 | 3–6 | **station 01** |
| b2 | 7–10 | **station 02** |
| b3 | 11–14 | **station 03** |

The new six-column package installed on 2025-02-21 is at **station 02**.

Observed channel ranges per block over the 2018-07-26 to 2019-12-11 sample, useful as a sanity
check when validating a loader:

| Block | Station | `Batt` [V] | `Tair` [°C] | `RH` [%] | `I` [mdeg] |
|---|---|---|---|---|---|
| b1 | st01 | 3.26 – 4.73 | −4.03 – 36.76 | 14.29 – 100.02 | 1779.25 – 2005.50 |
| b2 | st02 | 3.28 – 4.74 | −3.29 – 37.60 | 19.10 – 99.96 | 2104.63 – 2251.00 |
| b3 | st03 | 3.29 – 4.73 | −3.85 – 34.67 | 18.53 – 100.14 | 2056.25 – 2372.13 |

Note that the current-era station 02 inclinometer is a **different instrument** from legacy block
b2, and the two are kept in separate columns everywhere in this project. Nothing guarantees that a
replacement instrument's arbitrary zero matches its predecessor's, so the separation is the safe
default and costs nothing.

Whether the two *in fact* disagree on a level is a separate, measurable question, and the answer is
that they do not. Measured on the recorded channel with no compensation and no anchor applied
(`studies/01_data_exploration/`, 2026-08-21), the 30-day window means either side of the changeover are
**−335.90 mdeg and −334.32 mdeg**, a step of **+1.58 mdeg**; the last legacy and first current-era
conditioner readings, twelve hours apart, differ by **7 mV**. The two instruments are continuous
across the replacement to within a small fraction of the ordinary daily swing. Why they agree so
closely is not recoverable from the archive — a reused mount, or a new unit zeroed to match, would
both produce it.

The practical rule this yields is the opposite of an offset treatment, and is stated in
Section 7.3: the station 02 record spans two instruments and must be compensated and anchored
**once, as a single series**. Concatenating two separately anchored series is what creates a step
here, not the hardware.

---

## 4. Value conventions and sentinels

### 4.1 Zero is a missing-data sentinel — except for solar radiation

For `Batt`, `Tair`, `RH` and `I`, the value `0.000` is written when the channel is unavailable. It
is not a measurement: a battery at 0 V, an inclinometer at 0 mdeg and a relative humidity of 0 %
are all physically implausible for this installation, and the zeros appear in whole-block runs
that coincide with known unit outages. **Every `0.000` in these channels must be converted to
`NaN` at read time.** Treating them as measurements injects fabricated flatlines into the record,
and the return to normal values at the end of an outage then presents itself as a step change.

`n_SR` (solar radiation) is the exception. Zero is a legitimate night-time reading for that
channel and must be preserved.

### 4.2 `-55.0` is the wall-temperature failure sentinel

`n_Twall` reports exactly `-55.0` when the wall temperature probe is not responding. This is the
bottom of the sensor's range, emitted as an open-circuit indicator rather than as a measurement.
It must be converted to `NaN`. Its extent is large and is documented in the quality report.

### 4.3 `8191.875` is a solar radiation wrap-around

`n_SR` reads `8191.875 W/m²` — that is, 2¹³ − 0.125, the full scale of the field — on records that
are otherwise unremarkable and that frequently fall at night. The value is an unsigned wrap-around
of a near-zero or slightly negative reading, not a measurement. Any `n_SR` above roughly
1400 W/m² is unphysical at this latitude and must be rejected.

### 4.4 Decimal separator is mixed

Both `3.500` and `3,500` occur. In the earliest files the comma form dominates, and in later files
the dot form dominates, but the two also **coexist inside a single file** in 2019. A parser must
normalise the separator per field, not per file and not per era. Detecting the separator once from
the first few lines and applying that decision to the whole file will silently corrupt values.

---

## 5. Structural defects a parser must handle

These are properties of the archive as delivered, not hypothetical edge cases. Each is quantified
in the quality report.

1. **Duplicate timestamps with conflicting payloads.** The same `(date, time)` key can appear
   several times in one file. Some copies are byte-identical; others differ, and the differences
   are not random — one copy typically carries real values in a block where the other copy carries
   zeros. Deduplicating by keeping the first or last occurrence therefore discards real
   measurements. Records must be merged field-wise, preferring the non-sentinel value, and any
   genuine value conflict must be reported rather than silently resolved.
2. **Day-boundary bleed.** A file can contain records belonging to the previous calendar day —
   typically a single `23:40:00` record at the top of the file. The date must be taken from the
   record's own date field, never from the filename.
3. **Incomplete days.** Files with fewer than 72 records are common and represent intra-day
   dropouts. A day-level presence count therefore understates the true amount of missing data.
4. **Whole missing days.** Days absent from the archive are absent as files. They occur in long
   contiguous outages, not as scattered singletons.
5. **Restart transients.** The first day of acquisition after a long outage can carry inclinometer
   values far outside the normal band. These are instrument settling artefacts and should be
   screened before use.

---

## 6. Minimum parsing contract

Any loader for these files must, at a minimum:

- read with `sep='\t'`, `header=None`, and no dtype inference on the numeric fields;
- normalise the decimal separator field by field;
- build the timestamp from fields 1 and 2 with `dayfirst=True`, ignoring the filename;
- branch on the record's field count (14 or 20) rather than on its date, so that a file whose
  content disagrees with the expected era is handled rather than mangled;
- map `0.000` to `NaN` in all `Batt`, `Tair`, `RH` and `I` channels, and preserve it in `n_SR`;
- map `-55.0` to `NaN` in `n_Twall`;
- map `n_SR` above 1400 W/m² to `NaN`;
- merge duplicate timestamps field-wise and surface any conflict;
- return a series indexed on a regular 20-minute grid, with absent slots present as `NaN` rather
  than omitted, so that gap statistics downstream are computed against the intended sampling grid.

One test that does **not** belong in this contract, recorded here because it is a natural thing to
add and was carried in this project until 2026-08-21: **do not reject inclinometer readings for
falling outside the calibration certificate.** The certificate covers ±2° about the conditioner's
2500 mV zero, and it is tempting to map `|I − 2500| > 2000` to `NaN` alongside the sentinels. The
recorded deflections leave that interval, so the certificate no longer describes the data it would
be used to judge, and the test discards readings the instrument demonstrably produced rather than
markers the acquisition system wrote. Every rejection above identifies a value written *in place
of* a measurement; none is a judgement about whether a genuine reading is too large. Impulsive
excursions on the inclinometer belong to a filter that compares a sample with its own
neighbourhood — Stage 2 of Section 7.2 — which marks what it replaces instead of silently emptying
it.

---

## 7. The inclinometer signal: cleaning, compensation, and the meaning of zero

### 7.1 What the `I` channel measures, and what its absolute value is worth

`I` is reported in millidegrees, but it is **not** an inclination referred to any physical datum.
It is a raw transducer reading whose absolute level is set by how the instrument happened to sit in
its mount at the moment it was installed, plus whatever offset the electronics carry. A value of
2160 mdeg at station 02 does not mean the wall leans by 2.16°; it means the instrument's arbitrary
zero sits at 2160 counts of its own scale.

The consequence is the single most important interpretive rule for this dataset:

> **The absolute level of the inclinometer series carries no structural information. Only its
> changes do.**

Two operations are therefore legitimate, and one common shortcut is not.

**First-order difference — preferred.** The 20-minute increment
`ΔI(t) = I(t) − I(t − 20 min)` is invariant to every arbitrary constant in the chain: the
instrument's installation offset, the normalisation anchor of Section 7.3, and the thermal
reference temperature of Section 7.2 all cancel exactly under differencing. This is the quantity
that has a physical meaning independent of how the record happens to have been assembled, and it
is what any drift, rate or anomaly statement should ultimately rest on.

**Re-anchoring to a chosen reference — legitimate but arbitrary.** Subtracting the value at some
chosen date, or better the mean over a chosen quiet window, produces a series readable as
"movement since that reference". It is useful for presentation and for comparing stations on one
axis. It is arbitrary by construction, and the arbitrariness has to be stated: a different
reference window gives a different-looking series with identical information content. Prefer a
window mean over a single date, since a single record carries the full measurement noise of that
record into every subsequent value.

**Treating the level as a measurement — wrong.** Any statement of the form "station 02 is at
2160 mdeg" is meaningless, and any model whose output depends on that level rather than on its
increments is reading the mounting geometry, not the structure.

Two practical cautions apply when differencing. First, the difference must be taken on the regular
20-minute grid of Section 6, with absent slots present as `NaN`; taken on a compacted index it
silently spans the gaps, so a single increment can straddle 271 days of outage and appear as an
enormous step. Every difference whose two endpoints are not exactly one sampling interval apart
must be discarded rather than used. Second, differencing amplifies high-frequency noise while
removing the low-frequency content, so it changes what a model of the series is describing: a
trend term fitted to `ΔI` is a rate of drift, not a drift, and its interpretation in the
methodology has to follow accordingly.

### 7.2 The cleaning and compensation pipeline as currently implemented

**Stage 1 — Power-loss removal.** Rows where `charge == 0` are dropped and recorded with
`drop_reason = 'power_loss'`. This is the same zero sentinel described in Section 4.1, seen from
the battery channel. Removing these rows *before* the spike filter is what stops outage zeros from
entering the rolling-median window and dragging the local median toward zero, which would then
cause the genuine readings around the outage to be misclassified as spikes.

**Stage 2 — Rolling-median context spike filter** (`clean_signal_robust`). For each valid sample,
the filter builds a local median from up to `window` valid neighbours on each side, excluding the
sample itself, and drops the sample when

```
|I(k) − median(neighbours)| > spike_threshold
```

Samples with fewer than `min_valid` neighbours are kept unconditionally, so gap edges are never
trimmed for lack of context. A median over both sides is used rather than a first-difference test
specifically so that the *first clean reading after an outage* survives: it is far from its
left-side neighbours but close to its right-side ones, whereas a genuine spike is far from both.
Dropped rows are recorded with `drop_reason = 'spike'`.

**Stage 3 — Linear thermal compensation** (`temp_compensation`). The correction is

```
I_comp(t) = I(t) − (T(t) − T_ref) · comp_coeff · 1000
```

with `T` the station's own on-board air temperature (`temp`) and `T_ref = T` at the **first
retained record**. It is a single-coefficient linear model with no lag term: it assumes the
instrument's thermal response is instantaneous and proportional.

**Stage 4 — Normalisation.** The compensated series is shifted so that its first valid value is
exactly 0, by subtracting that first value.

Stages 3 and 4 are described here per station, which is what Notebook 00 does and is correct as
long as one station means one instrument. Where a station's record spans an instrument
changeover it is not, and Section 7.3 states the rule that applies instead.

Current parameter values in Notebook 00:

| Parameter | Value | Meaning |
|---|---|---|
| `SIGNAL_COL` | `'absinc'` | the `I` channel after loading |
| `TEMP_COL` | `'temp'` | station's own `Tair` |
| `COMP_COEFF` | `0.005` | mdeg · °C⁻¹ · 10⁻³ |
| `SPIKE_THRESHOLD` | `500.0` | mdeg deviation from local median |
| `SPIKE_WINDOW` | `7` | valid neighbours each side |
| `SPIKE_MIN_VALID` | `3` | minimum neighbours to evaluate a sample |

Outputs per station, written to `data/interim/sensor/`: `{st}_preprocessed.csv` with the
compensated signal under the original column name and the uncorrected values preserved as
`{signal_col}_raw`, and `{st}_dropped.csv` carrying every removed row with its `drop_reason`.

### 7.3 Both anchors are arbitrary, and both are parameter-dependent

Stages 3 and 4 each fix a constant from **the first retained record**: `T_ref` in the thermal
correction, and the zero level in the normalisation. Neither is a property of the structure.

**One anchor per station, never one per instrument.** This is the rule that matters most in
practice, and getting it wrong manufactures a defect that looks exactly like a finding. Where a
station's record spans two instruments — which at Gubbio means station 02 across the 2025-02-21
changeover — the compensation and the normalisation must be applied **once, to the joined record**,
with a single `T_ref` and a single anchor for the whole series. The slope term of Stage 3 still
takes the recording unit's own `Tair`, since the bias being cancelled is each instrument's response
to the temperature it actually sat at; only the reference is shared.

The natural alternative — compensate and normalise each instrument on its own first record, then
concatenate — is wrong, and it was done in this project until 2026-08-21. Normalisation subtracts a
constant, so two series normalised on two different records carry two unrelated constants, and
their difference appears at the join as a step. Measured at station 02 that step was **−126.09
mdeg**, and it decomposed as:

| Term | Contribution |
|---|---|
| The recorded channel | **+1.58 mdeg** |
| Difference between the two anchor readings | −27.00 mdeg |
| Difference between the two `T_ref` values, at 5 mdeg/°C | −100.39 mdeg |
| Sum | −125.81 mdeg, against −126.09 observed |

The dominant term is the reference temperature: the legacy anchor fell on a July day at 21.55 °C
and the current-era anchor on a February evening at 3.58 °C, and Stage 3 carries that gap of nearly
eighteen degrees straight into the level. **None of the step is instrumental** — see Section 3.3.
Anchoring once removes it at the source, leaving a residual of −8.99 mdeg across the changeover,
which is what the compensation removes for the 2.11 °C by which the two comparison windows differ
in season.

Do not estimate and subtract an "era offset" instead. Doing so treats an arithmetic artefact as a
measurement, and it will report a physical instrument step where none exists.

The three legacy units are the opposite case and keep their own anchors: they stand at three
different places on three different pieces of wall, and they have nothing to anchor in common.
Only where two instruments in succession measure the same wall does a shared anchor mean anything,
and there it is mandatory.

This has a practical consequence that is easy to miss. The first retained record is itself an
output of Stages 1 and 2 — change `SPIKE_THRESHOLD`, `SPIKE_WINDOW`, or the date range loaded, and
a different record may become the first survivor. Both anchors then move, and the entire series
shifts by a constant. Two runs of the pipeline with different spike settings therefore produce
series whose levels are not comparable, even though their increments are identical.

`T_ref` additionally carries the full measurement noise of one single temperature sample into a
constant offset applied to every subsequent value. A mean over an initial window would be the
more defensible choice.

None of this affects `ΔI`. It is a further reason to treat the differenced series as the primary
quantity and the levelled series as a presentation layer.

### 7.4 Known limits of the current implementation

Recorded here so they are not rediscovered. None is a defect in the code as written for the legacy
era; each is a boundary that the 2025 instrument change or the archive diagnosis has moved.

1. **The `STATIONS` dictionary in Notebook 00 describes three blocks of four channels and
   therefore assumes the 14-column era.** Applied to a current-era file it will not find the new
   package's six columns, and the three legacy blocks it does find are entirely zero after
   2025-02-21. Notebook 00 needs an era branch before it can read current data.
2. **`DECIMAL_COMMA` is a single flag applied to the whole load.** Section 4.4 shows the separator
   varies within individual files, so no single value of this flag is correct for the legacy era.
3. **`COMP_COEFF = 0.005` is the manufacturer's specified coefficient**, 5 mdeg/°C, prescribed to
   cancel the temperature-induced measurement bias of the inclinometer. Two consequences survive
   that provenance and are worth keeping in view: the value has not been fitted to these data, so
   it is not a calibration of these particular instruments; and there is no guarantee that the
   package installed in 2025 shares the coefficient of the legacy units it replaced.
4. **Compensation uses air temperature, not wall temperature.** The current-era package reports
   `Twall` directly, which is the physically closer driver of the instrument's thermal response.
   Whether it is the better compensation variable is an empirical question, and it interacts with
   the availability problem documented in the quality report — `Twall` is missing on 219 of the 416
   present days.
5. **The linear instantaneous model has no thermal lag term**, while the framework's own
   feature-engineering stage exists precisely because the structure's thermal response is lagged.
   Stage 3 and the lagged-feature stage are making inconsistent assumptions about the same physics.
6. **The pipeline produces a levelled series, not a differenced one.** Anything downstream that
   consumes the level inherits the arbitrary anchor of Section 7.3.

---

### 7.5 Sign convention and the expected direction of the thermal response

Section 7.1 establishes that the absolute level of `I` is meaningless. Its *changes* are not, and
they carry a sign whose physical meaning is fixed by the site geometry. This subsection records
that meaning, because it is not recoverable from the files and is the single piece of context most
likely to be guessed wrongly by anyone reading a correlation table.

**The site geometry.** The wall stands in an embankment situation with two distinct faces:

| Face | Exposure |
|---|---|
| **Valley side** | The more exposed to the sun. |
| **Mountain side** | The less exposed. |

**The expected mechanism.** Under daytime heating the valley side, receiving more direct
insolation, expands more than the mountain side. The differential thermal expansion tips the wall
**towards the mountain**.

**The convention.** A movement towards the mountain is a **negative** change in inclination.

**The prediction that follows.** During the day, when solar radiation and temperature are high, the
inclination should move negatively. Any heating driver — air temperature, wall temperature, solar
radiation — should therefore show a **negative** association with the inclination. A negative
correlation is the expected result at this site and must not be reported as an anomaly, a sign
error, or a reason to flip a coefficient.

**Plausible lag range — for external forcings.** The mechanism is direct insolation heating one
face of a masonry embankment. A thermal front reaching the depth that governs the inclination is
not expected to take longer than **12 hours**, and no process at the site would impose a slower
transport. Lag or cross-correlation scans against `Tair` and `SR` should be bounded accordingly.
Beyond that bound a second problem appears: on a near-periodic diurnal signal a delay of `d` hours
is indistinguishable from a *lead* of `24 − d`, so an unbounded scan can return an optimum that is
really an aliased lead. A lead cannot be represented by a causal filter and cannot be used by a
forecasting system, so an optimum found beyond 12 hours against an external forcing should be
treated as an artefact of the scan range rather than as a measurement.

**`Twall` is not an external forcing, and the same rule does not apply to it.** Air temperature and
solar radiation are inputs to the system: they must precede the response they cause, so
constraining their delay to be non-negative is physically correct. Wall temperature is an
**internal state variable** — a temperature measured at some depth inside the masonry — and nothing
requires it to precede the deformation. Which way the phase runs is set by where the probe sits
relative to the layers that actually drive the movement, which are those nearest the sun-exposed
valley face:

| Probe position | What it reads | Expected phase |
|---|---|---|
| **Shallow** | a fast, weakly damped signal close to the surface forcing | the inclination **lags** the probe |
| **Deep** | a strongly damped and delayed signal | the inclination **leads** the probe |

A scan against `Twall` should therefore admit **signed** delays. A negative optimum is a
measurement, not an error. It remains unusable by a forecasting system — applying it would require
the probe's future values at the moment the forecast is issued — so the correct treatment is to
measure it as a diagnostic and to clamp it to zero before building any predictive feature.

**The probe's depth is unknown**, and it is not recoverable from the archive. It is listed as an
open question in `data-quality-report-2026-08-10.md` Section 7. Until it is established, the phase
measured in `studies/obsolete/inclination_prediction/` Step 5b is the quantity of record; once the depth is
known, that phase is the consistency check to run against it.

**An unresolved contradiction, recorded rather than resolved.** Measured on the current-era record
at station 02 (Tier 1 window, 2025-02-21 to 2025-07-13, hourly grid), the **raw** inclination moves
*positively* with temperature, against the expectation above:

| Series | `r` with `Tair`, levels | `r` with `Tair`, diurnal band | OLS slope on `Tair` |
|---|---|---|---|
| Raw `I` | **+0.845** | **+0.966** | **+1.63 mdeg/°C** |
| Compensated | −0.956 | −0.975 | — |

The compensated series shows the expected negative sign, but it does so by arithmetic rather than
by physics: the compensation of Section 7.2 subtracts 5 mdeg/°C from a channel whose measured
thermal slope is +1.63, leaving a residual of −3.37 mdeg/°C. The sign flip is entirely a product of
the over-correction.

**The same slope appears on the legacy instrument.** Measured independently on the 14-column era
at the same station (2018-07-26 to 2025-02-20, hourly, `studies/obsolete/inclination_prediction_legacy/`),
the raw legacy inclinometer gives **+2.379 mdeg/°C** on the diurnal band against **+2.291** for the
current-era package. The two are separate units of hardware, replaced one for the other in
February 2025, and they agree to within 4 %.

Two independent instruments reproducing a slope that runs opposite to the structural expectation is
much harder to attribute to a per-instrument fault than to the instrument type or to the sign
convention itself. Note that this argument rests on the two being different units of hardware, not
on their levels being unrelated: Section 3.3 records that their recorded levels in fact agree to
within 1.58 mdeg across the changeover. The slope is a property of the response, and it is
unaffected by whether the two zeros coincide.

Three readings are consistent with this and the data cannot separate them:

1. The instrument's sign convention is inverted relative to the one assumed above, so a positive
   raw change is in fact movement towards the mountain. **The two-instrument agreement favours
   this reading.**
2. The instrument's own thermal drift is positive and larger than the structural response, masking
   it. This now requires two different instruments to drift by nearly the same amount, which is
   possible if it is a property of the sensor model but not if it is a per-unit defect.
3. The structural response at this station does not follow the differential-expansion expectation.

Until this is settled — by the calibration sheet, the installation record, or an independent
measurement of the sensor's mounting orientation — **the expected sign stated above is the site's
structural convention and the measured raw slope is a separate, contradicting fact.** Both belong
in any analysis that reads signs, and neither should be quietly dropped in favour of the other.
Recorded from the study `studies/obsolete/inclination_prediction/`, Step 4b.
