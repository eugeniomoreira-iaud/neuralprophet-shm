# Study · Do three thermometers tell the same story?

Three independent sources measure the environment at Gubbio. This study puts them
on one grid under one naming scheme and measures how far apart they are — in
absolute terms, in their daily dynamics, and in their relationship with the
calibrated inclination.

**The complete evaluation is in
[`report/proxy_comparison_report.pdf`](report/proxy_comparison_report.pdf).**
This file only says how to run and rebuild things.

## The three sources

| Suffix | Source | What it measures | Coverage of archive span |
|---|---|---|---|
| `_str` | On-structure logger | Air inside the instrument housing on the wall | 70.1 % |
| `_gs` | Ground station, Gubbio town | Standard exposure | 94.8 % |
| `_era5` | ERA5 reanalysis (Oikolab) | ~9 km grid cell average | 99.4 % |

**They are not replicates.** A housing on a sun-exposed wall, a screen in town and
a nine-kilometre grid average are three different quantities that share a name.
Disagreement is expected; the study measures it.

## Why it had to come first

`studies/unified_dataset/` established that every on-structure channel fails at
once: on **100 %** of the hours the inclinometer is missing, so are its air
temperature, humidity and battery. No on-structure covariate can ever fill a real
gap. An independent source is the only escape — and whether that escape is real
depends on availability *and* on carrying the same relationship with the wall.

## What it found

- **The deadlock is broken.** During the hours the target is missing, the ground
  station is present on **94.3 %** and ERA5 on **99.6 %**. The on-structure
  sensor is present on 0.0 %, by construction.
- **Temperature substitutes well.** Ground station agrees with the on-structure
  sensor to 1.12 °C mean absolute (r = 0.988); ERA5 to 1.53 °C (r = 0.978).
- **Humidity does not.** Limits of agreement span fifty percentage points. Do not
  substitute it between sources.
- **An expectation was overturned.** ERA5's diurnal amplitude is **19 % larger**
  than the on-structure sensor's, not smaller. The sensor closest to the wall is
  the most *damped* — a housing with thermal mass on massive masonry buffers the
  daily swing.
- **Agreement collapses under differencing.** Temperature correlation falls from
  0.99 in levels to **0.70** in first differences. Most of the level agreement is
  a shared annual cycle; the hour-to-hour movement, which is what the wall
  responds to, agrees far less well. This bounds proxy-driven reconstruction.
- **A defect is confirmed by triangulation.** The on-structure radiation channel
  agrees with neither proxy (r ≈ 0.74) while the two proxies agree with each
  other at r = 0.930, slope 0.998. It also reads ~60 W/m² at midnight and places
  solar noon *after* its own temperature maximum. Do not use it as a physical
  measurement without recalibration.
- **The relationship with the wall must be measured inside contiguous blocks,
  within one instrument era.** Computed on the whole 2018–2026 record at once
  the differenced relationship looks negligible for every source. Cut into
  blocks it is not: air temperature reaches r = −0.95 in levels and −0.84 in
  first differences, which is what the two prediction studies report on their
  own blocks.
- **A defect had to be registered first.** The record carries 175 single-hour
  acquisition faults — 0.248 % of the grid — in which the inclination, the air
  temperature and the battery all fail together and recover on the next sample.
  They live almost entirely in the first difference and they reverse
  conclusions, so they are marked in `P_23` with the evidence for each and
  masked on the calibrated level. They are a property of
  `data/interim/unified/`, which every study reads.
- **The apparent sign reversal of the on-structure radiation channel was one of
  those artefacts**, not a fourth symptom of its known defect: r goes from
  +0.036 to −0.533 once the faults are registered. Nothing here reopens the
  sign question of `docs/raw-data-format.md` §7.5.
- **All three agree on sign, no channel leads the deformation, and every driver
  optimises at zero transport delay.** Only solar radiation needs an operator —
  a one- to two-hour thermal inertia lifts explained variance from 0.570 to
  0.708 for the ground station.
- **Solar radiation is a real driver in the differenced domain and a small
  one.** It reaches roughly half the differenced correlation of air temperature
  and adds one to three points of explained variance beyond it. The
  on-structure pyranometer covers 12.7 % of the span; both proxies supply
  radiation on more than 94 % of the hours the target is missing.
- **Night zeros flatter every radiation statistic.** Restricted to daylight, the
  two proxies' agreement falls from r = 0.930 to 0.886 and their mean absolute
  difference nearly doubles, 47 to 90 W/m².

## Two rules the study obeys

**The target is fixed.** `inc_comp` is the calibrated reading — the
manufacturer's formula applied with the temperature measured at the transducer.
That is an instrument calibration, not a regression, so the coefficient is never
re-estimated and no alternative target is built. The only permitted adjustment is
the levelling correction across the 2025-02-21 regime change.

**The clock is measured, not assumed.** Two independent tests: an absolute one
against computed solar noon, and a relative one by cross-correlation. They
disagree for the on-structure radiation channel, which is how that channel's
defect was found.

## Contents

| File | Role |
|---|---|
| `proxy_comparison_study.py` | The study, in `py:percent` format. **Edit this one.** |
| `proxy_comparison_study.ipynb` | Paired notebook. Generated by `jupytext`. |
| `pc_lib.py` | Proxy loading and unit harmonisation, canonical naming, the two clock tests, pairwise agreement and Bland–Altman, diurnal comparison, relationship with the target, and the data-dictionary generator. |
| `tests/` | Assert scripts for the library additions. Run each with `python tests/<name>.py`; no pytest is required or installed. |
| `report/` | Report source and PDF. |
| `outputs/` | Tables and figures. Gitignored. |

`pc_lib.py` imports from four sibling studies. Moving or renaming any of them
breaks it.

## The data dictionary

`docs/proxy-data-dictionary.md` is the canonical channel reference for every
study that follows. It is **generated** from the mapping tables in `pc_lib` by
`pc_lib.write_data_dictionary`, so code and documentation cannot drift apart —
do not edit it by hand.

## Reproducing

```bash
conda activate neuralprophet_env
cd studies/proxy_comparison
jupytext --to ipynb proxy_comparison_study.py
jupyter nbconvert --to notebook --execute --inplace proxy_comparison_study.ipynb
```

Runtime is a few minutes; there is no model fitting beyond OLS and the operator
scans.

**Inputs.** `studies/unified_dataset/build_unified_dataset.py` must have run
first. The study reads its output plus `data/raw/proxies/meteosystem_gubbio.csv`
and `data/raw/proxies/oikolab_weather.csv`, and nothing else — it never touches
the raw archive.

> **Note on `jupytext`.** `jupytext --to ipynb` performs a *sync*, and takes
> whichever of the pair is newer as the source. After running the notebook the
> `.ipynb` is newer, so a later `--to ipynb` will overwrite your `.py` edits.
> Delete the `.ipynb` first if you have just edited the `.py`.

## Rebuilding the report

Run the notebook first — the report reads figures from `outputs/`, which is not
committed.

```bash
cd report
pdflatex proxy_comparison_report.tex   # twice, for the ToC
```

The shared preamble is `../../_shared/reportstyle.tex` and stays inside the
package set shipped with a basic TeX Live installation.
