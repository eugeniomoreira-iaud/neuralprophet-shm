# Study 3 · The thermomechanical response: what moves the wall, and how long it takes

Study 1 said what the record contains. Study 2 said what the external sources say about the
weather at this site. Neither measured how the wall responds. This study screens every candidate
driver the project has against one response — study 1's analysis column, `inc_comp_cleaned` — in
three rounds, one per source family: the sensors on the structure, the ERA5 reanalysis, and the
weather station in Gubbio town.

**The complete document is
[`report/thermomechanical_response_report.pdf`](report/thermomechanical_response_report.pdf).**
This file only says how to run and rebuild things.

**How the coupling is measured.** Each pair is scanned over two parameters, not one: a
transport delay, which shifts a driver without changing its shape, and a thermal time constant,
which low-passes it because masonry integrates its forcing rather than following it. They are not
interchangeable — a filter with the right time constant and no delay produces almost the same peak
correlation as a delay with no filter, while meaning something entirely different — so both are
scanned together and the whole grid is reported. Two bounds on that grid carry the study's
reasoning and are set out in the report: the diurnal band is scanned over delay alone, because on
a band of one period a time constant and a delay are the same phase shift and fitting both flips
the sign of the result; and on the levels the grid stops at three days, because beyond that the
optimum slides to the annual cycle and measures season rather than inertia.

**Scope.** The response is the compensated, cleaned inclination and nothing else. The
raw-versus-compensated contradiction of `docs/raw-data-format.md` § 7.5 is cited by this study and
deliberately not re-opened by it: the numbers here are measured on the compensated series and
describe that series. Screening the raw channel alongside it is the obvious next study.

## What it reads

| Input | Provenance |
|---|---|
| `../../data/interim/archive/gubbio_archive_20min.csv` | **Study 1's product.** The response, and every on-structure driver |
| `../../data/raw/proxies/oikolab_weather.csv` | ERA5 through Oikolab, read with study 2's channel map |
| `../../data/raw/proxies/meteosystem_gubbio.csv` | The Gubbio station, read with study 2's channel map |

Study 1 is a prerequisite: its export must exist before this notebook runs. Study 2 is **not** — no
output of study 2 is read. The two proxy exports are loaded through the same `shmlib.proxies`
functions and the same maps study 2 uses, so a channel means here exactly what it means there, but
this study never waits on study 2's choice of source. It screens all three and reports whether a
coupling survives the change.

## The three rounds

| Round | Source | Drivers |
|---|---|---|
| 1 | On-structure | `tair` (study 1's joined column, both eras), `sr` (`n_sr_ok`, suspect days honoured), `twall` (`n_twall_filtered`), `rh`, `batt` |
| 2 | ERA5 | `tair`, `rh`, `sr`, `tdew`, `rain`, `wspd`, and the two components of `wdir` |
| 3 | Gubbio station | the same, plus `pres` and `rain_rate` |

Three properties of that set are load-bearing:

- **Supply voltage is a negative control, not a candidate.** No mechanism moves a wall by battery
  voltage, so whatever it scores is meant to be the study's noise floor. It did not behave as one
  — a solar-charged supply in a sun-exposed housing carries the daily cycle in its voltage — and
  the report says so and reads the floor as conservative rather than null.
- **Wind direction is never scanned as a bearing.** A bearing wraps, so every linear statistic on
  degrees is dominated by where the scale was cut. It enters as its two Cartesian components.
- **Wall temperature is not an external forcing.** It is an internal state variable at unknown
  depth, so it is scanned over signed delays where the forcings are scanned forwards only. Its
  optimum is negative — the inclination leads the probe — which is a measurement, and which must
  be clamped to zero before any predictive feature is built from it.

## What it writes

Tables and figures into `outputs/`, which the report reads: `TR_01`–`TR_08` for tables (and
`TR_T01`, `TR_T03`, `TR_T04`, `TR_T06`, `TR_T08` as LaTeX table bodies) and `TR_F01`–`TR_F10` for
figures:

| Figure | Shows |
|---|---|
| `TR_F01` | The response's first analysis week, as its recorded level and as the diurnal band the rest of the study screens. |
| `TR_F02` | Round 1's lag curves — correlation against transport delay, diurnal band, on-structure drivers, current era. |
| `TR_F03` | Round 2's lag curves, ERA5, diurnal band, whole record. |
| `TR_F04` | Round 3's lag curves, the Gubbio station, diurnal band, whole record. |
| `TR_F05` | The response against the strongest surviving driver overall, as a hexbin cloud with the fitted gain and its confidence interval. |
| `TR_F06` | That driver's — and every shortlisted driver's — gain re-fitted month by month at the lag step 7 chose. |
| `TR_F07` | Round 1's delay-by-time-constant operator grid, on-structure, levels. |
| `TR_F08` | Round 2's operator grid, ERA5, levels. |
| `TR_F09` | Round 3's operator grid, the Gubbio station, levels. |
| `TR_F10` | Step 7.5's native-resolution delay scan, one row per instrument era, one panel per driver variable, one curve per source family. |

`TR_F07`–`TR_F09` are the delay-by-time-constant grid, one per source family; those three
are the study's only two-dimensional figures and they are not self-explanatory, so section 7.1 of
the report, *How to read one of these grids*, is the guide to them — what the axes are, what the
colour can and cannot say, and what each shape of bright region means.
No dataset is exported. This study's product is a measurement, not a file the pipeline consumes.

**`TR_07`/`TR_08`/`TR_F10` are step 7.5's native-resolution delay**, a re-measurement of the same
diurnal delay `TR_02`/`TR_T04` report from the hourly coupling scan, but on the archive's own
twenty-minute grid instead of the hourly one — a daily-demeaned Pearson correlation against a
displaced reference (`shmlib.temporal_alignment.reference_shift_scan`), scanned separately over
the legacy and current instrument eras since `sr_str` and every other current-only channel exist
from 2025-02-21 on. The wall follows radiation by twenty to forty minutes across the three sources
and both eras (forty minutes exactly against ERA5 in the current era) and leads air temperature by
forty to eighty minutes; the hourly grid's own "zero to one hour" for the same drivers
(`TR_T04_coupling_diurnal.tex`) is that same delay rounded to the hour it can resolve, not a
disagreement. Every recovered correlation is negative, as the site's geometry predicts, with no
exception. **Study 5's `RADIATION_DELAY_H` parameter should read its value from
`TR_08_native_delay_summary.csv`** — the `sr_era5`/current row's `delay_minutes` (currently 40),
converted to hours — rather than from the hourly coupling table, which cannot state the delay more
finely than the hour.

## Contents

| File | Role |
|---|---|
| `thermomechanical_response_study.py` | The study, in `py:percent` format. **Edit this one.** |
| `thermomechanical_response_study.ipynb` | Paired notebook, generated by `jupytext`. |
| `report/` | Report source and PDF. |
| `outputs/` | Tables, LaTeX table bodies and figures. Gitignored. |
| `tests/` | Unit tests for the `shmlib` decisions this study relies on. |

This study has no library of its own. The thermal operator, the delay-and-time-constant scan,
the band separation, the gains and their stability live in `../shmlib/coupling.py`; the response loader and the era join in
`shmlib.proxies`; the wind decomposition in `shmlib.meteo`; the figures in `shmlib.figures`; step
7.5's native-resolution delay scan in `shmlib.temporal_alignment`, written for study 5's
clock-alignment question and reused here for a physical delay instead.
`shmlib.coupling.lag_scan` is also what `shmlib.quality.clock_check` now uses for study 2's clock
test — one implementation of the search, two callers, differing only in the objective they
maximise and the range they scan.

## Reproducing

```bash
conda activate neuralprophet_env
cd studies/03_thermomechanical_response
jupytext --to ipynb thermomechanical_response_study.py
jupyter nbconvert --to notebook --execute --inplace thermomechanical_response_study.ipynb
```

Run `auto_watcher.py` from the repository root during a working session and the `.py` and
`.ipynb` stay in sync without the first command.

The tests run without a test runner:

```bash
python studies/03_thermomechanical_response/tests/test_shmlib_study03.py
```

## Rebuilding the report

Run the notebook first — the report reads figures and table bodies from `outputs/`, which is not
committed.

```bash
cd report
pdflatex thermomechanical_response_report.tex   # twice, for the ToC
```

The shared preamble is `../../_shared/reportstyle.tex`.

## The check that says the machinery is right

Section 7.5 of `docs/raw-data-format.md` records, from a now-retired study, the compensated
response against on-structure air temperature over 2025-02-21 to 2025-07-13: r = −0.956 on the
levels and −0.975 on the diurnal band. Re-measured here through `shmlib`, the same window gives
**−0.956 and −0.974**. The agreement is what licenses the rest of the study's numbers; a
disagreement would have been a finding about the retired study rather than something to tune away.

## What the retired study contributed, and what it did not

The two-parameter operator comes from `studies/obsolete/inclination_prediction/`, whose figures
scan delay against time constant on a grid this study now reproduces. The **method** is sound and
is adopted here. Its **numbers** are not: its own grid stopped at 168 h, its radiation optimum sat
exactly on that stop, and the gain it reported from that cell is the seasonal slide this study
documents rather than a thermal measurement. That is the distinction the `obsolete/` rule exists
to enforce — the code and the reasoning there can be read and learned from, the conclusions cannot
be cited.
