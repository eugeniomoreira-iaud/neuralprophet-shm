# Study · The monitoring system of the Mura Urbiche di Gubbio, and its record

What the instrumentation is, how the acquisition writes its files, how the
inclinometer is compensated, how much data each of the three stations has
contributed since **26 July 2018**, and what station 02's four core channels
look like across the whole archive.

**The complete document is
[`report/data_exploration_report.pdf`](report/data_exploration_report.pdf).**
This file only says how to run and rebuild things.

This is the first study to read in this project. Every other one assumes the
instrumentation, the file format, the compensation and the shape of the
archive.

## What it reads

**The raw `.adc` archive, directly.** One parameter, `RAW_ARCHIVE_DIR` at the top
of the notebook, points at the read-only archive in `_UNIPG/__Mura-realtime/`;
nothing else is an input, and `studies/unified_dataset/build_unified_dataset.py`
is not a prerequisite. Files are copied once into `.cache/` before parsing,
because Google Drive serves the archive at roughly ten files per minute — the
first run is slow and shows a progress bar, later runs take seconds.

This study is the one that decides what in the archive is a measurement and what
is not. It cannot be handed a cleaned dataset: the wall-temperature probe spent
eight months reporting the `-55.0` failure sentinel, and an ingest that turned
those readings into gaps before this study opened them would hide exactly what
this study exists to report.

## What it writes

`../../data/interim/archive/gubbio_archive_20min.csv` — **the study's product**,
the union of every archive file on the native **20-minute** grid, about 212 000
rows and 69 columns, roughly 44 MB. Nothing here aggregates: the hourly grid the
rest of the pipeline uses exists so that the record can be aligned against hourly
external proxies, and building it belongs to the study that prepares that
comparison.

Every channel appears three times:

| Column | Content |
|---|---|
| `{block}_{channel}` | The field **exactly as recorded**, sentinels included |
| `{block}_{channel}_flag` | What was decided about the value; empty when it stands |
| `{block}_{channel}_ok` | The value the study uses |

A flag is of one of two kinds, and they differ in what `_ok` gets:

- a **rejection** — `sentinel`, `wrap`, `out_of_range` — says the value is not a
  measurement and that nothing is known in its place, so `_ok` is **missing**;
- a **correction** — `night` — says the value is wrong but its true value is
  known, so `_ok` carries that value.

There is one correction. Radiation recorded while the sun is more than six
degrees below the horizon cannot be radiation, and its true value is zero, so
`n_sr_ok` is zero there and `n_sr_flag` says `night`. **That zero is the only
number in any `_ok` column that was not measured**; the recorded value sits
beside it untouched, and a consumer who would rather have a gap can make one from
the flag. Evaluating where the sun was requires knowing the logger's clock, which
the study measures rather than assumes: the archive is in Italian civil time and
the logger does observe daylight saving.

Blocks are `st01`, `st02`, `st03` and `n`. The current-era package is `n` rather
than `st02` even though it stands at station 02: its inclinometer shares no
baseline with legacy block b2, and keeping them in separate columns makes that
structural rather than a convention someone has to remember.

Derived columns follow: `{block}_inc` and `{block}_inc_comp` per block, then the
target station's chain `inc` → `inc_comp` → `inc_comp_joined` →
`inc_comp_cleaned`. **`inc_comp_cleaned` is the analysis column.**

`inc_spike` and `sr_suspect` are verdicts, not measurements, and carry their own
columns because they condemn a reading without either rejecting or correcting it.
The first marks interpolated inclination samples. The second marks the 161 days
on which the radiation channel has no diurnal cycle — reporting as much in the
dark as at high sun, or nothing at all through a summer midday. Honour them.
Gaps are never filled; only spikes inside observed stretches are replaced.

`era` says which column layout produced a slot — `legacy` or `current`, decided
by the record's own field count rather than by its date — and is **empty where
nothing was recorded**. That is narrower than the column of the same name in the
unified hourly dataset, which labels every slot in an era's date range whether or
not anything was written there. Cross-checked against it, the two never disagree
on a slot that carries data.

`gubbio_archive_20min_manifest.json` beside it records the archive extent, the
record counts, every flag total, the offset applied and the filter settings, so
the file can be audited without re-running the notebook.

Tables and figures go to `outputs/`, which the report reads.

## Contents

| File | Role |
|---|---|
| `data_exploration_study.py` | The study, in `py:percent` format. **Edit this one.** |
| `data_exploration_study.ipynb` | Paired notebook. Generated by `jupytext`. |
| `de_lib.py` | The study's library: archive ingest, flagging and correction, the census, the views the notebook plots from. |
| `report/` | Report source and PDF. |
| `outputs/` | Tables, LaTeX table bodies, and figures. Gitignored. |
| `.cache/` | Local copy of the `.adc` files. Gitignored, and safe to delete at the cost of a slow next run. |

`de_lib` does not reimplement the file format. Parsing comes from
`shmlib.adc.parse_file` and `shmlib.adc.parse_legacy_file`, the documented
thresholds and the compensation from `shmlib.adc`, the archive extent and the era
boundaries from `shmlib.site`, and the solar geometry behind the night correction
from `shmlib.solar`, which carries the site coordinates and their provenance. The
study holds its own decisions — what to condemn, where to cut, which day to
believe — and takes nothing else from anywhere.

## Two figures are supplied by hand

`outputs/DE_F01_stations_map.png` (the map of the three station locations) and
`outputs/DE_F02_st02_sketch.png` (the plan and section of the embankment at
station 02) are **not** generated by the notebook. Both are already in place
and the report includes them directly. Replacing either one means dropping a
new file at the same path; no change to
`report/data_exploration_report.tex` is needed.

## Reproducing

```bash
conda activate neuralprophet_env
cd studies/01_data_exploration
jupytext --to ipynb data_exploration_study.py
jupyter nbconvert --to notebook --execute --inplace data_exploration_study.ipynb
```

**Figures.** Plotting goes through `apply_report_style()`, defined near the
top of `data_exploration_study.py`, which is already fixed at seaborn's
`paper` context. Figure width is set per call through `figsize=(6.38, ...)`,
the manuscript column width in inches; change it there to re-export at another
size.

## Rebuilding the report

Run the notebook first — the report reads figures and table bodies from
`outputs/`, which is not committed.

```bash
cd report
pdflatex data_exploration_report.tex   # twice, for the ToC
```

The shared preamble is `../../_shared/reportstyle.tex` and stays inside the
package set shipped with a basic TeX Live installation.
