# Raw Data Quality Report — Gubbio Archive, 2026-08-10

Diagnosis of the `.adc` archive at `_UNIPG/__Mura-realtime/` as it stood on 2026-08-10. The
format conclusions drawn from this analysis are consolidated in
[`raw-data-format.md`](raw-data-format.md); this document records the measurements themselves and
their consequences for the modelling design.

## Scope of the analysis

| Era | Files in archive | Files analysed | Coverage of the analysis |
|---|---|---|---|
| Current, 20 columns, 2025-02-21 → 2026-08-10 | 416 | 416 | Complete |
| Legacy, 14 columns, 2018-07-26 → 2025-02-20 | 1926 | 504 (2018-07-26 → 2019-12-11) | Partial |

The current era is characterised exhaustively. Legacy-era figures are drawn from an eighteen-month
sample and should be treated as indicative of the whole legacy period, not as a census of it.
Extending the legacy analysis requires copying the remaining files off Google Drive streaming
first, which at the observed throughput takes on the order of two hours.

---

## 1. Archive extent and whole-day gaps

The archive spans 2018-07-26 to 2026-08-10, that is 2938 calendar days, of which 2342 are present
as files. The 596 missing days amount to 20.3 % of the record.

The missingness is **not** scattered. It resolves into exactly seven contiguous outages and zero
isolated missing days:

| Outage start | Duration [days] |
|---|---|
| 2022-09-23 | 271 |
| 2022-06-02 | 112 |
| 2026-03-07 | 103 |
| 2024-08-28 | 42 |
| 2024-05-05 | 40 |
| 2025-09-26 | 17 |
| 2020-06-06 | 11 |

Per calendar year, the days present and missing are: 2018, 159/0 (partial year, acquisition began
in July); 2019, 365/0; 2020, 355/11; 2021, 365/0; 2022, 153/212; 2023, 194/171; 2024, 284/82;
2025, 348/17; 2026, 119/103 (partial year).

This structure matters for the framework's gap-reconstruction stage. Filling a scattered hour from
its neighbours is a different statistical proposition from reconstructing a 271-day block in which
no information about the structure's behaviour was recorded at all. Any claim of conformal
coverage over such a block rests entirely on the environmental regressors and on the assumption
that the structural response mapping did not change while the system was down — an assumption
that a 271-day outage is precisely the situation least able to support.

---

## 2. The February 2025 instrument changeover

The transition from 14 to 20 columns occurs between `GUBBIO_20250220.adc` (14 columns) and
`GUBBIO_20250221.adc` (20 columns). The changeover date of 21 February 2025 is confirmed exactly;
there is no transitional period and no file with an intermediate column count.

The consequence is more severe than a schema change. Across all 416 files of the current era, all
twelve legacy columns are zero on **100 %** of records. The new package did not augment the
legacy network — it replaced it. From 2025-02-21 onwards the archive contains one instrumented
location, not three.

---

## 3. Channel-level profile of the current era

Statistics over all 416 current-era files, with sentinel values excluded from the ranges.

| Column | Channel | Zero rate | Records at −55 | Min | Max |
|---|---|---|---|---|---|
| 3–14 | legacy b1, b2, b3 | 100 % | — | — | — |
| 15 | `n_Batt` [V] | 0 % | 0 | 3.251 | 5.157 |
| 16 | `n_Tair` [°C] | 0 % | 0 | −5.30 | 38.95 |
| 17 | `n_RH` [%] | 0 % | 0 | 22.77 | 100.30 |
| 18 | `n_I` [mdeg] | 0 % | 0 | 527.00 | 2359.13 |
| 19 | `n_SR` [W/m²] | 9.6 % | 0 | 0.125 | 8191.875 |
| 20 | `n_Twall` [°C] | 0 % | 15572 | −55.00 | 57.06 |

Three of these entries are defects rather than measurements.

**`n_Twall` = −55.0 on 15572 records.** This is the probe's open-circuit indication. It affects 219
distinct days in two long runs, described in Section 4.

**`n_SR` reaching 8191.875 W/m² on 3103 records.** The value is 2¹³ − 0.125 and appears at times
such as 03:20 and 04:20, when incident solar radiation is necessarily zero. It is a wrap-around of
a near-zero or negative reading into an unsigned full-scale field, not a saturation caused by
sunlight. The physical ceiling at this latitude is on the order of 1000–1100 W/m².

**`n_I` reaching 527 mdeg.** Eighty-six records fall below 2000 mdeg against a working band of
roughly 2100–2250. The extreme values cluster on 2026-06-18, the first day of acquisition after
the 103-day outage, and are consistent with instrument settling rather than with structural
movement. One isolated excursion on 2025-05-10 at 14:40 is not explained by a restart.

The `n_SR` zero rate of 9.6 % is not a defect. Night-time zeros are the expected behaviour of a
pyranometer and must be preserved rather than treated as missing.

---

## 4. Usable windows in the current era

Classifying each of the 536 days from 2025-02-21 to 2026-08-10 by whether the day is present and
whether `n_Twall` is valid on it:

| State | Days |
|---|---|
| Present, `Twall` valid, all 72 records | 155 |
| Present, `Twall` valid, incomplete day | 42 |
| Present, `Twall` = −55 on at least one record | 219 |
| Absent from the archive | 120 |

The chronology is:

| From | To | State | Days |
|---|---|---|---|
| 2025-02-21 | 2025-07-13 | usable | **143** |
| 2025-07-14 | 2025-09-25 | `Twall` failed | 74 |
| 2025-09-26 | 2025-10-12 | missing | 17 |
| 2025-10-13 | 2026-03-06 | `Twall` failed | 145 |
| 2026-03-07 | 2026-06-17 | missing | 103 |
| 2026-06-18 | 2026-08-10 | usable | **54** |

The wall-temperature failure and the acquisition outages interleave rather than overlap, which is
the worst arrangement available. If `n_Twall` is treated as a required regressor, the current era
yields **197 usable days in exactly two blocks — 143 days and 54 days — separated by 393 days
during which either the probe or the whole system was down.**

The 143-day block runs from late February to mid-July, and the 54-day block covers high summer.
Together they sample roughly one half of an annual cycle, and they sample it discontinuously.

---

## 5. Legacy-era defects

From the 504-file sample covering 2018-07-26 to 2019-12-11:

- **165 files (33 %) contain duplicate `(date, time)` records.** Some duplicates are
  byte-identical; others are not. Where they differ, the pattern is systematic rather than random:
  one copy carries real values in a block where the other copy carries zeros. On 2019-04-15 at
  10:00:00, one of the four copies of the record has block 1 zeroed while another reports
  `3,330 / 8,830 / 91,335 / 1843,750`. Both records are present in the same file. Deduplicating by
  first or last occurrence would therefore discard a third of a day's block-1 measurements at
  random.
- **68 files (13 %) mix comma and dot decimal separators internally.** In the 2019-04-15 example
  the conflicting duplicates are distinguished precisely by their separator, which suggests that
  the archive was assembled by concatenating exports from two different tools without
  reconciliation.
- **32 files (6 %) contain at least one record dated to a day other than the filename date**,
  typically a single `23:40:00` record belonging to the previous day.
- Per-unit availability is low and highly variable. Monthly liveness for block 1 ranges from 28 %
  to 100 %; block 2 was entirely absent through October 2018. The three units fail independently,
  so a requirement that all three report simultaneously would reduce the usable legacy record far
  below what any single block offers.

---

## 6. Consequences for the study design

Four points follow directly from the measurements above and should be settled before further
modelling.

**The stated plan to work from 2025-02-21 onwards buys a clean schema at a high price.** It gives
one station instead of three, and 197 `Twall`-valid days instead of 536. A NeuralProphet
configuration with a yearly seasonality term cannot identify that term from 143 contiguous days
plus a detached 54-day block; the seasonal component would be fitted to less than one cycle and
would not be separable from trend. Either the yearly term is dropped and the framework's claims
are restated in terms of daily and annual-subset behaviour, or the record has to be extended
backwards into the legacy era.

**Wall temperature is the binding constraint, and it may not need to be.** `Twall` is unavailable
on 219 of 416 present days, while `Tair`, `RH`, `SR` and `I` are available on all of them. If
`Twall` enters the model as one regressor among several rather than as a required input, the
usable record grows from 197 days to 416. Whether that is acceptable depends on how much of the
inclinometer response `Twall` explains that `Tair` and `SR` jointly do not — which is exactly the
question the regressor-diagnosis stage exists to answer, and which should therefore be run before
the `Twall` requirement is fixed. It also connects to the availability-during-gaps filter already
in the methodology: a regressor that is missing on 53 % of days is precisely what that filter is
designed to demote.

**The legacy era is recoverable but not cheap.** Its defects — duplicate records with conflicting
payloads, mixed separators, day-boundary bleed — are all mechanical and can be handled by a
correct loader. The station identity of the three blocks, which cannot be recovered by parsing,
has since been supplied from the installation record: the blocks appear in station order, b1 =
st01, b2 = st02, b3 = st03.

**The two eras cannot be joined into one inclinometer series without an explicit treatment.** The
current-era instrument is a physical reinstallation with its own baseline. Concatenating across
2025-02-21 produces a step that no environmental regressor explains, and any anomaly detector run
across that boundary will fire on it.

---

## 7. Open questions

Resolved since this report was first written, both from the installation record:

- The legacy blocks appear in station order — b1 = st01, b2 = st02, b3 = st03. Recorded in
  `raw-data-format.md` Section 3.3.
- The legacy column order is not different from the current package's first four channels. Every
  legacy block carries `Batt, Tair, RH, I` in that order, consistent across the 504-file sample.

Still requiring the user's input:

1. Was the 2025-07-14 wall-temperature failure ever serviced, and is the probe expected to return?
   The record resumes with valid `Twall` on 2026-06-18, which suggests either a repair during the
   2026 outage or an intermittent fault.
2. Is there an installation or maintenance log that would explain the seven outages and the
   per-unit dropouts? Attributing gaps to known causes materially strengthens the missingness
   argument in the paper.
3. **How far inside the masonry is the `n_Twall` probe?** The depth is not recorded anywhere in
   the archive, and it decides how the wall-temperature channel should be interpreted. The
   deformation is governed by the temperature field near the sun-exposed valley face, so a shallow
   probe should be followed by the inclination while a deep one should be preceded by it — the sign
   of the phase between them is diagnostic of the installation. That phase has been measured, on
   the diurnal band, in `studies/obsolete/inclination_prediction/` Step 5b, and is recorded in
   `raw-data-format.md` Section 7.5. An answer to this question does not require re-running
   anything: it is a consistency check against a number already on record.
