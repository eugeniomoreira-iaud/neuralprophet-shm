# Obsolete studies

Six earlier investigations. **Their conclusions are not trusted and must not be cited, carried into
the manuscript, or built upon.** They were written before `../01_data_exploration/` established what
in the raw archive is a measurement and what is not, and several of them are built on cleaned or
compensated inputs whose preparation has since been superseded. Read them for context — what was
tried, what looked promising, which dead ends are already closed — never for a result.

Anything one of these studies concluded that still matters must be **re-established** by a current
study from study 1's exported archive, not quoted from here.

---

## Nothing outside this folder imports them any more

These studies were once load-bearing: study 1 took its `.adc` parsers, its documented thresholds,
its era constants and its solar geometry from four of them, which meant a retired conclusion and a
working parser could not be separated. All of that code now lives in `../shmlib/`, extracted
verbatim and checked against the originals over 45 real archive files, 15 constants and 6 derived
quantities — every hash identical.

| Was imported from | Now lives in |
|---|---|
| `thermal_compensation/tc_lib` — `parse_file`, thresholds, compensation, sampling | `shmlib.adc` |
| `thermal_compensation_legacy/lc_lib` — `parse_legacy_file`, legacy columns | `shmlib.adc` |
| `unified_dataset/ud_lib` — archive extent, era boundaries, target station | `shmlib.site` |
| `proxy_comparison/pc_lib` — site coordinates, solar elevation, solar noon, daylight mask | `shmlib.solar` |
| `tc_lib` — `set_context`, `figsize`, `_finish` | `shmlib.viz` (as `finish`) |

Each folder here still contains its own copy of those libraries, still importing its siblings by
relative path, so the studies remain runnable as they were written. They are self-contained: they
import each other and nothing else in the project imports them.

**Deleting this folder is now safe** as far as the current studies are concerned — study 1, study 2
and `shmlib` do not reference it. What would be lost is the record of what was already attempted,
which is the reason it is still here.

---

## What is here

| Folder | What it asked | Why it is retired |
|---|---|---|
| `unified_dataset/` | Build one hourly, cleaned, provenance-labelled table of the whole archive, then measure what imputation can do for it | Its ingest cleaned the record before anything had established what a defect was; study 1 replaced it as the authoritative read of the archive |
| `proxy_comparison/` | Do the on-structure logger, the town ground station and ERA5 tell the same story? | Superseded by study 2, `../02_proxy_forcing_characterization/`, which re-asks the temperature and radiation half of it from study 1's archive. Its clock tests are the part worth reading |
| `thermal_compensation/` | Is the documented fixed thermal correction good for a prediction system? Current instrument era | Verdict reached on the superseded ingest; the sign contradiction it left open is documented in `docs/raw-data-format.md` §7.5 |
| `thermal_compensation_legacy/` | The same question on the legacy era, full calendar year 2021 | As above |
| `inclination_prediction/` | Which channel of the extended package predicts the inclination, can a combination beat it, how far ahead can it forecast? | Accepts the compensation that its sibling study rejected, on inputs study 1 supersedes |
| `inclination_prediction_legacy/` | The same set on the six-and-a-half-year legacy record, plus periodicity and gap-filling | As above |

Each folder keeps its own `README.md` and its report PDF, both written when the study was current.
**Those documents state their findings as settled. They are not.** This file overrides them.

---

## Note on paths

These studies were moved down one directory level when this folder was created. Every relative path
inside them was re-depthed accordingly: `../../` became `../../../` for the repository root, for
`data/`, for `docs/` and for `_shared/reportstyle.tex`. The `.py` files carry the correction; their
paired `.ipynb` notebooks still hold the old paths until `jupytext --sync` is run, which is harmless
as long as nothing here is executed.
