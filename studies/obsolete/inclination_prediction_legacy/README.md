# Study · Predicting the inclination from the legacy sensor set

Station 02 of the 14-column network, **2018-07-26 to 2025-02-20** — six and a
half years, against the 143 usable days of the extended package. Channels:
`tair`, `rh`, `batt`, `inc`. No solar radiation, no wall temperature.

**The complete evaluation is in
[`report/inclination_prediction_legacy_report.pdf`](report/inclination_prediction_legacy_report.pdf).**
This file only says how to run and rebuild things.

## What this study is for

Two families of question. The first is what the extra length buys and is
specific to this era. The second is the set `../inclination_prediction/` asked,
repeated here so the two eras are **directly comparable** — every ported test
uses the sibling study's machinery unchanged, so any difference in result is a
difference in the data, not in the method.

**Specific to the long record**

1. **Which periodicities are present** — Lomb–Scargle on the gapped record, so
   nothing is imputed before the spectrum is estimated. Two controls: the
   **window function** (the periodogram of the observation mask alone, which
   catches outage artefacts) and a **permutation null** for false-alarm levels.
2. **Over which stretch to decompose them**, and whether they persist —
   sliding-window harmonic fits tracking amplitude and phase.
3. **How to fill missing values** — measured by injecting gaps of known length
   and scoring four fillers against them, not asserted.

**Ported from the extended-sensor study**

4. **Best single predictor?** — operator scan on both bands, a signed-delay
   control, ranking without autoregression on two windows, negative control.
5. **Can a combination beat it, and how is that diagnosed?** — collinearity,
   exhaustive search, leave-one-out gain against permutation importance, and an
   out-of-period test on a block held back entirely.
6. **How far ahead, with what uncertainty, and what is the error made of?** —
   the sibling study's horizon and calibration work merged with this study's
   nested error budget, scored at eleven horizons out to one year.

Question 6 merges the two deliberately: a horizon curve says how large the error
is, and the budget says which component produced it.

**Raised by the answers, added afterwards**

7. **How much of the record can NeuralProphet actually use?** — the gap
   tolerance sweep, windows gained against hours invented, whether the record is
   one series or two, and the decomposition and prediction regimes the
   longest-block restriction had been hiding.

Question 7 exists because Questions 2 and 6 were answered under a restriction
this study asserted rather than tested. Step 13b tests it, and Questions 1–6 are
deliberately left as they were so that every ported result stays comparable with
the sibling study.

## Two hard limits, stated up front

**180 d cannot be distinguished from 182.62 d.** Separating an independent
180-day process from the annual second harmonic needs a frequency resolution of
8.0×10⁻⁵ cycles/day, i.e. a record of about **34 years**. Six and a half will
never do it. The study settles the question by **phase** instead: a true second
harmonic stays phase-locked to the annual term, an independent process drifts.

**No contiguous block holds two annual cycles.** Even absorbing interruptions of
up to three days, the longest unbroken stretch is under 1.7 cycles. The annual
term therefore cannot be identified from any single window and must be fitted
across the gaps — which a least-squares harmonic fit does and a *single fitted
window* cannot. Hence the split used in Questions 1–6: **spectrum and harmonics
on the whole era, autoregressive work on the longest block.**

**That second half is weaker than it looks, and Step 13b takes it apart.** An
autoregressive model needs windows of consecutive observations, not one
continuous series, and those exist in every block. Fitting across every complete
stretch gives **6.57 annual cycles and 2.26× the training windows for 1.5%
fabricated data**, and cuts the deployable configuration's prediction intervals
from 164–185 mdeg to 12–23. Questions 1–6 keep the restriction for
comparability; Question 7 measures what it cost.

**The 438-day outage of 2022–2023 never closes.** No defensible tolerance bridges
it. Anything requiring a strictly continuous series is limited to the earlier
half, at 3.71 cycles and 14.3% fabricated data.

## Premises

Unchanged from `../inclination_prediction/` and not re-tested here: the logged
`I` channel is read as millidegrees (in this era it runs 2105–2300, i.e.
**outside** the ±2000 mdeg the datasheet certifies — accepted by instruction);
the documented compensation is taken as correct; the grid is one hour. The site's
sign convention and the 12-hour lag bound for external forcings are in
[`docs/raw-data-format.md`](../../docs/raw-data-format.md) §7.5.

## Contents

| File | Role |
|---|---|
| `inclination_prediction_legacy_study.py` | The study, in `py:percent` format. **Edit this one.** |
| `inclination_prediction_legacy_study.ipynb` | Paired notebook. Generated by `jupytext`. |
| `lp_lib.py` | Spectral estimation, harmonic decomposition, imputation benchmark, error budget, sensitivity. Loading comes from `../thermal_compensation_legacy/lc_lib.py`; ranking, subsets and horizon machinery from `../inclination_prediction/ip_lib.py`. |
| `report/inclination_prediction_legacy_report.tex` | Report source. Reads figures from `../outputs/`. |
| `report/inclination_prediction_legacy_report.pdf` | The report. Tracked. |
| `outputs/` | Tables and figures. Gitignored. |
| `.cache/` | Local copy of the 1926 legacy `.adc` files. Gitignored. |

`lp_lib.py` imports from three sibling studies. Moving or renaming any of them
breaks it.

## Reproducing the study

```bash
conda activate neuralprophet_env
cd studies/inclination_prediction_legacy
jupytext --to ipynb inclination_prediction_legacy_study.py
jupyter nbconvert --to notebook --execute --inplace inclination_prediction_legacy_study.ipynb
```

The cache holds 1926 files. Populating it from scratch takes about three hours
at the observed Drive streaming rate; copying
`../thermal_compensation_legacy/.cache/` across first covers 1244 of them
instantly. The permutation null (200 replicates) and the NeuralProphet stage
dominate the remaining runtime — set `N_PERMUTATIONS = 0` or
`RUN_NEURALPROPHET = False` to shorten a working pass.

**Figures.** All plotting goes through seaborn; set `CONTEXT = 'paper'` to
re-export at manuscript column size.

## Rebuilding the report

Run the notebook first — the report reads figures from `outputs/`, which is not
committed.

```bash
cd report
pdflatex inclination_prediction_legacy_report.tex   # twice, for the ToC
```

The shared preamble is `../../_shared/reportstyle.tex` and stays inside the
package set shipped with a basic TeX Live installation.
