# Studies

Each numbered folder here is one self-contained investigation: a question, the code that answers
it, the tables and figures that answer produces, and a report that states the answer in prose. A
study is not a pipeline stage. It may be read on its own, it owns its own library, and it is
finished when its report is written.

```
studies/
├── README.md                          this file
├── shmlib/                            the shared library — facts, never methods
├── _shared/reportstyle.tex            the shared LaTeX preamble
├── 01_data_exploration/               study 1
├── 02_proxy_forcing_characterization/ study 2
├── 03_thermomechanical_response/      study 3
├── 04_neuralprophet_inclination_prediction/ study 4
└── obsolete/                          retired studies — conclusions not trusted
```

---

## The sequence

The number in the folder name **is** the reading order, and it is also the dependency order: a
study may build on any study before it and none after it.

| # | Study | Question | Status |
|---|---|---|---|
| 1 | [`01_data_exploration/`](01_data_exploration/) | What is the instrumentation, how does the acquisition write its files, which recorded values are measurements, and what does the archive contain? | Complete |
| 2 | [`02_proxy_forcing_characterization/`](02_proxy_forcing_characterization/) | What do the external proxy databases say about air temperature and solar radiation at this site, how far apart are they, and how do they compare with the on-structure radiation over the narrow window in which that channel is trustworthy? | In progress |
| 3 | [`03_thermomechanical_response/`](03_thermomechanical_response/) | Which candidate drivers does the wall actually respond to, after how long, with what gain and in which direction — screened separately against the on-structure sensors, ERA5 and the town station, so that a coupling can be seen to survive a change of source? | Complete |
| 4 | [`04_neuralprophet_inclination_prediction/`](04_neuralprophet_inclination_prediction/) | What is the compensated inclination record made of, is a newly arrived reading the one the measured environment predicts, and how far ahead is forecasting worth anything? | In progress |

A new study takes the next free number when it is started, not when it is finished. Numbers are
never reused and never renumbered: a report that cites study 2 must still find study 2 there a year
later.

### Study 1 is the foundation

`01_data_exploration/` is the only study whose method is trusted without qualification. It reads
the raw `.adc` archive directly, decides what in it is a measurement and what is not, and publishes
that decision as a table in which every channel appears three times: as recorded, as a flag naming
the verdict, and as the value the project uses. Its product,
`data/interim/archive/gubbio_archive_20min.csv`, is the sensor-side input to everything that
follows. Any later study that needs a sensor value takes it from there, or re-derives it and says
why.

### What a later study may assume

- **The record and its defects** — from study 1's report and its exported table. Do not re-litigate
  a rejection decision; if one looks wrong, fix it in study 1 and re-export.
- **The file format** — `docs/raw-data-format.md`. Binding on any code that parses an `.adc` file.
- **The site coordinates and the channel names** — `docs/proxy-data-dictionary.md`.

Everything else is the later study's own to establish.

---

## The library-first rule

**A study notebook is written out of `shmlib` functions, every important variable is visible in the
notebook, and no code lives outside `shmlib`.** Before writing new code: understand what already
exists — `/graphify` answers "what already does X" and "what calls Y" across the tree faster than
reading files one at a time; if a function already does it, call it; if one nearly does, adapt it
without changing what existing callers get; only then write a new one, in `shmlib`. Data pointers
and governing parameters live in the notebook's parameter cell and are passed in as arguments,
never buried in a function body — a study must be re-aimed at another station or another window by
editing that cell alone.

The binding statement of this rule, with the reasoning and the conditions on step 2, is
`instructions-pipeline.md` § Studies at the repository root. It is the only place it is written in
full; this paragraph is a signpost, not a second copy.

## The shared library

[`shmlib/`](shmlib/) is the studies' only library. Every function and every constant a study uses
lives there.

| Module | Holds |
|---|---|
| `shmlib.adc` | The `.adc` file format, both era parsers, the instrument constants, the documented compensation |
| `shmlib.site` | Eras, stations, channels, the logger's clock, the archive's extent |
| `shmlib.solar` | Site coordinates and NOAA solar geometry |
| `shmlib.meteo` | Circular statistics for directional data, and the decomposition of a bearing |
| `shmlib.coupling` | Thermal operators, delay-and-time-constant scans, band separation, gains and their stability |
| `shmlib.prediction` | Gap-safe changes and segments, chronological folds, NeuralProphet wrappers, baselines, scores, block-bootstrap skill and gap closure |
| `shmlib.tables` | LaTeX table bodies and cell formatting |
| `shmlib.viz` | Colour identity, report style, figure saving |

**What belongs in it.** Everything. A file format, a physical constant, a site fact, a plotting
convention — and equally a function that encodes a decision, such as which day to condemn or which
lag to accept. There is no second tier and no judgement to make about which library a function is
destined for.

**Where the decisions live, then.** In the notebook that makes them, as the arguments it passes. A
function that encodes a choice takes that choice as a documented argument; the study states its
value in its parameter cell. That is what keeps a decision arguable — it is visible on the page a
reader is already reading, rather than in a private module one directory away. It is also what
keeps the studies independent: before `shmlib` existed, study 1 imported its parser from a
superseded study, which meant a retired conclusion and a working parser could not be separated.

`shmlib` is not installed as a package. A study reaches it by putting `studies/` on the path, which
every study notebook does in its imports cell:

```python
import sys, os
sys.path.insert(0, os.path.abspath('..'))      # studies/, for shmlib
sys.path.insert(0, os.path.abspath('../..'))   # repo root, for heritageshm

from shmlib import adc, meteo, site, solar, tables, viz
```

It is a different library from `heritageshm/` at the repository root: that one is the production
pipeline, this one is the studies' common ground. Its tests run without a test runner:

```bash
python studies/shmlib/tests/test_shmlib.py
```

### Importing one study from another

A later study may import an earlier study's library directly, and study 2 imports study 1's
`de_lib` for the schema and the verdicts of the exported archive. That is allowed because the
dependency runs forwards, in the same direction as the numbering. Study 3 imports nothing from
either: it reads study 1's exported archive and, for the proxies, the same `shmlib` loaders and
channel maps study 2 uses, which is how it stays independent of a study that is still in progress
while still meaning the same thing by a channel name. The reverse never happens: an
earlier study that imported a later one would make its own conclusions depend on work done after it
was written.

---

## Retired studies

`obsolete/` holds six earlier studies whose **conclusions are not trusted** and must not be cited or
built upon. Since their load-bearing code moved into `shmlib`, nothing outside that folder imports
them. See [`obsolete/README.md`](obsolete/README.md).

---

## Layout of a study folder

```
NN_<study_name>/
├── README.md                    How to run and rebuild. Not where the findings live.
├── <study_name>_study.py        The study, in py:percent format. Edit this one.
├── <study_name>_study.ipynb     Paired notebook, generated by jupytext. Never edit directly.
├── outputs/                     Tables, LaTeX table bodies, figures. Gitignored.
├── report/                      Report source and PDF. The findings live here.
├── tests/                       Unit tests for the shmlib functions this study relies on.
└── .gitignore                   outputs/, .cache/, __pycache__/, LaTeX build artefacts.
```

**Conventions that hold across every study.**

- The `.py` file is the source of truth. The `.ipynb` is generated; editing it directly corrupts
  the pair. Run `auto_watcher.py` from the repository root during a session, or `jupytext --sync`
  afterwards.
- Each study has a two-letter prefix, and that prefix, upper-cased, names every artefact it writes:
  `DE_01_*.csv`, `DE_F01_*.png`, `PF_01_*.csv`, `PF_F01_*.png`. Tables are numbered in one sequence
  and figures in another. The prefix comes from the study's name, not from its number, so an
  artefact keeps its identity even if it is quoted far from the folder it came from.
- Notebooks orchestrate; `shmlib` implements. A cell longer than about ten lines, or one that
  defines a function, builds a table row by row, performs a statistical test or composes a figure
  axis by axis, belongs in `shmlib` as a documented function. See the library-first rule above.
- `shmlib` functions take every path and every governing parameter as an argument. Notebooks own all
  I/O paths and all choices.
- The report reads its figures and table bodies from `../outputs/`, which is not committed, so the
  notebook must be run before the report can be rebuilt. The shared preamble is
  `_shared/reportstyle.tex`, included as `\input{../../_shared/reportstyle.tex}` from a report
  directory one level below a study folder.
- Figure and colour rules are in `instructions-pipeline.md` at the repository root, implemented in
  `shmlib.viz`, and binding: Okabe–Ito for categories, Cividis for scalars, one fixed colour per
  measured channel across every figure in the project.
