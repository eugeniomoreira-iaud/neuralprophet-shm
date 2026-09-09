# Task 7.3 · Study 05 revision pass — unified plan

**Status:** Phase A closed; Phase B in progress. **Created:** 2026-09-07.

| Phase | State |
|---|---|
| A · inputs and rulings | **Closed.** Both numbers recovered, three of the check file's numbers refuted and corrected, rulings R1–R7 taken. |
| B.1 · `GROUND_STAMP_OFFSET` | **Done.** Parameter wired, guidance written; no library work was needed. |
| B.2 · per-source delay | **Done.** `build_regressor_sets` and `ladder_frame` both gained `radiation_delay_min`; notebook on minutes; D3's misquotation removed; 6 new tests. |
| B.3 · `GM_F15` coverage chart | **Done.** `figures.plot_regressor_coverage`, wired and smoke-run through the notebook's own call. |
| B.4 · `GM_F16` gap histogram | **Done.** `figures.plot_gap_size_histogram`, same. |
| B.5 · D15 | **Done.** Decision written into the design document; `coupling.reset_thermal_lag_filter`, `prediction.radiation_filter_sweep`, `figures.plot_radiation_filter_sweep`, the notebook cell, `GM_10b` and `GM_F17`. |
| **B · whole phase** | **Complete.** `shmlib` 165 passed; Study 05's own suite 85 passed, 1 xfailed. Every new figure smoke-run through the notebook's own call. |
| C · the run | Ready to start. Not blocked by the commit question below. |
| D · prose | Not started. |
| E · data-quality report | Not started; independent. |

Test count as each piece landed: 141 before the pass, 148 after the filter, 154 after the two
inventory charts, 160 after the sweep, 165 after `GM_F17`.

**Commits are held, deliberately (2026-09-07).** `stamp_offset` does not exist at `HEAD`: the whole
feature is another session's uncommitted work in `shmlib/proxies.py`, and Study 05's B.1 calls it.
Study 05's revised prose will also cite `PF_16_shift_summary` and `TR_08_native_delay_summary`,
neither of which is committed either — 1,569 uncommitted lines across Studies 02 and 03, plus the
untracked `shmlib/temporal_alignment.py` that the native scan imports. The user's ruling of
2026-09-07 is that the Study 02/03 session lands its own work first; Study 05 then commits on top.
The working tree is complete and correct meanwhile, so the run is not blocked — only the commit is.

**One defect found in the sweep before it ran.** `radiation_filter_sweep` took `freq` as its own
named parameter and never handed it on, so every fit — baseline and candidates alike — would have
run on `neuralprophet_backtest`'s default grid rather than the study's twenty-minute one, silently.
Fixed, and pinned by `test_the_study_grid_reaches_every_fit`, since the suite's monkeypatched fits
could not have caught it.
 **Parent plan:**
`docs/superpowers/plans/2026-09-05-study05-greybox-monitoring.md`, Task 7.3. **Ledger:**
`.superpowers/sdd/2026-09-05-study05-greybox-monitoring/progress.md`. **Source of the notes:**
`studies/05_greybox_monitoring/report/report05_check.md`, which stays the user's file and is not
edited by this plan; every note in it appears below under an item number, and the checkbox in the
source file is ticked as each item closes.

This plan merges the fifteen notes of the check file with the orchestrator's gate rulings and
resequences them so that the whole revision costs **one full notebook run** rather than three. It
supersedes the ordering sketched in chat on 2026-09-07; it does not supersede the check file's
content, which is reproduced here item by item.

Study 05 is closed through Task 7.2: report §1–§14 written, 45 pages, `STUDY_COMPLETE = True`,
all sixteen test suites green, three code reviews approved, and the serial control run reproducing
every `GM_` table bit for bit against the parallel run. Everything below is a revision of a
finished study, not a continuation of an unfinished one.

---

## 1 · Gate rulings, closed before any work begins

The check file leaves three decisions to the orchestrator and the previous session raised three
more. All six are ruled here so that Phase B can be executed without a further gate.

### R1 · `tair_str` stays in the on-structure set; the mixed response is declared, not engineered

Note N11 offers two ways to handle the on-structure air temperature: state in the text that its
gain is a mixed air-and-radiation response, or move the column to the station's and declare the
substitution in D2.

**Ruling: state it; do not move the column.** The `'str'` set is already
`{'tair': 'tair_str', 'rh': 'rh_str', 'sr': 'sr_gs'}` — its radiation is borrowed from the station
by D2. Moving the air temperature as well would leave `rh_str` as the only on-structure channel in
a set whose entire purpose is to represent what the wall's own instrument sees. The three-set
contrast that Sections 6 and 9 rest on would collapse, and a full rerun would be spent making the
report say less. The mixed response is a property of the probe and belongs in the prose, where a
reader can weigh it, together with the displacement evidence from Study 02's `PF_16_shift_summary`.

Consequence: N11 is prose only and costs no run.

### R2 · The daily-phase chart stays on the sweep floor

The phase chart's control limit sits at 1.00, the floor of the limit sweep, because the joint alarm
requires the EWMA and the CUSUM to agree and the shared CUSUM decision interval, not the EWMA
limit, is what gates that chart.

**Ruling: hold; sweep `CUSUM_H` per chart is refused, and no prose change is required.** Sweeping
the decision interval per chart is tuning a detector against the answer it is being asked to give,
and the report already carries the honest version of the situation twice. Section 10 states the
mechanism, gives the empirical evidence for it — the chart sat on the previous run's floor of 2.00
with the same run length on the same two episodes — and concludes that the chart is as sensitive as
D10's shared CUSUM allows. Section 13 repeats it as a limitation. Verified in the source on
2026-09-07; nothing to change.

### R3 · The bridge interval stays the pre-outage fit's raw band, and the calibrated reading stays
the one the verdict is written on

**Ruling: hold, with one addition (item N16).** D12 named the conformal interval, and the bridge
carries the raw quantile band of the single pre-outage fit instead, because no residuals exist
inside a gap to calibrate on. That deviation is already declared where it happens — Section 11
states it before any verdict is read, quotes the under-coverage, and writes its closing verdict
paragraph "at the interval this study trusts", the calibrated half-width; Section 13 repeats the
deviation and asks the reader to prefer the calibrated reading. The autumn 2025 outage is already
reported as inside the calibrated half-width. The doctrine is consistent and needs no change.

What is not consistent is Table 2's presentation: `GM_14`'s Verdict column is computed against the
raw band alone, so a reader who scans the table without reading the paragraph above it takes away
the reading the study has just asked them not to prefer. That gap is closed by N16.

### R4 · The two unbuilt design items stay declared, not built

The refit-slope companion beside the slow chart, and the expectation drawn through a gap, are
declared in Section 13 as designed-but-not-built. **Ruling: accept the declaration.** Section 13
exists for exactly this, the verdicts do not depend on either item, and both are better placed in
Study 06 than bolted onto a closed study.

### R5 · One run, not three

The check file's items fall into three classes: prose, which costs nothing; figures, which need the
notebook re-executed; and parameters, which need the notebook re-executed **from Movement 0** and
change every number downstream. Executing them in the check file's own order would run Movement 1
for the two new figures, then run everything again for the parameters, then re-check the prose
twice against two different sets of numbers.

**Ruling: every run-affecting change lands before the notebook is executed, and the notebook is
executed once.** This is what Phases B and C below encode, and it is why R1 had to be ruled first —
had `tair_str` moved to the station column, that too would have had to land before the run.

### R6 · D15 is developed against dumped state, never inside a full run

The radiation-filter sweep (N13) is the largest item in the pass: a new `shmlib` function, its
tests, a sweep cell, a new table and a new figure. **Ruling: build and test it against the
pre-Movement-5 state dump, exactly as the plan's binding rule of 2026-09-07 requires, and let the
single full run of Phase C be its first end-to-end execution.** Debugging a new sweep by
re-executing a 36-minute notebook is how a one-run revision becomes a five-run revision.

---

## 2 · Item index

Every note in `report05_check.md`, in the order it appears there, with the phase that executes it.
"Rerun" says what the item costs: *none* for prose, *in-run* for something the single Phase C run
produces, *from Movement 0* for a parameter that changes every number.

| # | Item | Report § | Type | Rerun | Phase |
|---|---|---|---|---|---|
| N1 | Acronyms GS, STR, ERA5 identified at first appearance | §2 | prose | none | D |
| N2 | Table 1 → grouped stacked bar chart, double Y axis, `inclination` first | §2 | figure | in-run | B |
| N3 | Rewrite the confusing coverage paragraph | §2 | prose | none | D |
| N4 | Text around the new chart made compliant with it no longer being a table | §2 | prose | none | D |
| N5 | Data-quality report to cover cleaning gaps, not only outages | — | external doc | none | E |
| N6 | Remove references to the internal data-quality report; make the text self-contained | §2 | prose | none | D |
| N7 | Table 2 → log-log gap histogram, exact 20-minute counts as bins, real-value tick labels | §2 | figure | in-run | B |
| N8 | Replace the DST doubt with a time-base statement citing Study 02's `PF_16_shift_summary` (note resolved 2026-09-06; the action item remains) | §2 | prose | none | D |
| N9 | `GROUND_STAMP_OFFSET` in the parameter cell, passed to `load_ground_station`, documented | §2 | parameter | from Movement 0 | B |
| N10 | Radiation delay in minutes, per source, replacing `RADIATION_DELAY_H = 1`; guidance text corrected to quote Study 03 faithfully | §3.3 / D3 | parameter | from Movement 0 | B |
| N11 | `tair_str` described as a mixed air-and-radiation response, the choice stated in D2 | §6 / D2 | prose | none | D |
| N12 | The expectation stated before Model B's figure, then compared to the learned kernels | §7 | prose | none | D |
| N13 | D15: `TAU_SWEEP_H` radiation-filter sweep, `GM_10b` and one figure, run as a diagnostic beside Model B | §7 / D15 | code + diagnostic | in-run | B |
| N14 | Undocumented wall azimuth and pyranometer mounting stated as a limitation | §13 | prose | none | D |
| N15 | Station stamping inferred, not declared; residual after correction as the bound | §13 | prose | none | D |
| N16 | `GM_14`'s verdict presented against the calibrated half-width as well as the raw band (added by ruling R3) | §11 | prose, optionally table | none, or in-run if a column is added | D |

Fifteen notes from the check file, one added by ruling. Nothing in the check file is dropped: the
resolved timezone note N8 keeps its action item, and the two limitations N14 and N15 keep their
separate identities rather than being folded together.

---

## 3 · Phase A · Inputs and decisions — **closed 2026-09-07**

Phase A recovered the two numbers that N9 and N10 depend on. Both are settled elsewhere in the
tree, so neither has to be chosen here. It also refuted three numbers the check file states, which
are corrected below rather than carried into the prose.

### A.1 · `GROUND_STAMP_OFFSET` — confirmed at 15 minutes

Study 02 sets `GROUND_STAMP_OFFSET = '15min'` in its parameter cell
(`studies/02_proxy_forcing_characterization/proxy_forcing_characterization_study.py:214`, "half the
station's documented 30-minute interval") and documents it as "the correction an interval mean
stamped at its end requires to move it to the centre it actually describes" (same file, line 120).
The library agrees: `shmlib/proxies.py:429-431` tells a study that reads the stamp as the
interval's end to pass `'15min'`.

Two qualifications go into the prose. First, the end-of-interval stamping is Study 02's
**empirically favoured hypothesis, not a documented vendor convention** — the vendor's own
`auxiliary/meteosystem_italy.py` documents a thirty-minute reporting interval and says nothing about
which edge a mean is stamped at; what supports the reading is that the station's current-era
air-temperature gain from displacement collapses from 0.007 to 0.0002 once the correction is
applied, which is the signature of a correction that explains the discrepancy. This is exactly what
item N15 asks Section 13 to say, and A.1 supplies the wording. Second, the value is settled in
Study 02's **notebook, README and library docstring, not in its compiled report**, whose comparison
and conclusion sections are unwritten; Study 05 therefore cites `PF_16_shift_summary` and the
parameter cell, never a Study 02 report section.

**Consequence for B.1: no library work.** `load_ground_station` already takes a `stamp_offset`
argument (`shmlib/proxies.py:394-396`, default `None`, which keeps the vendor's stamp), and Study 02
already passes it. N9 reduces to declaring the parameter in Study 05's cell, passing it at the call
site, and writing its Parameter Tuning Guidance.

### A.2 · The native-resolution radiation delay — present, in minutes, and per source

Study 03's native step exists in the working tree: `TR_07_native_delay_scan.csv` (20-minute shifts
over ±240 min), `TR_08_native_delay_summary.csv`, `TR_T08_native_delay_summary.tex` and figure
`TR_F10_native_delay_scan`, produced by "Step 7.5 · Diurnal delay at native resolution". The
summary's `delay_minutes` column, read directly, gives the delay of the **inclination** behind each
driver, with the station's stamp correction already applied at load time:

| Era | `sr_era5` | `sr_gs` | `sr_str` | `tair_era5` | `tair_gs` | `tair_str` |
|---|---|---|---|---|---|---|
| current | 40 | 20 | 20 | −60 | −60 | **0** |
| legacy | 40 | 40 | insufficient support | −80 | −40 | **0** |

Negative means the wall leads the driver. The hourly table the current notebook misquotes,
`TR_02_coupling_all.csv`, is confirmed to hold what the check file says it holds: on the diurnal
band, `sr_str` at 1 h and `sr_gs` and `sr_era5` at 0 h, with `lag` in **hours**. Study 03 itself
explains the discrepancy — "a delay read off an hourly scan can only ever land on the hour … that
range is the rounding an hourly grid imposes on the true delay, not a statement that the delay
itself is unresolved below the hour."

**Ruling R7 · one delay per radiation column, taken from the current era.** Study 05 consumes only
two radiation columns: `sr_gs`, which the on-structure and station sets share, and `sr_era5`. The
delays are therefore `sr_gs = 20 min` and `sr_era5 = 40 min`. The legacy era puts `sr_gs` at 40
rather than 20, a difference of exactly one slot, which is the scan's own resolution; an
era-dependent delay is not built for a one-slot difference, and the current-era value is preferred
because it is the era the monitor runs on. The prose states the legacy figure and this reasoning
rather than hiding it. Overruling this in favour of an era-switched delay is available and costs a
notebook parameter plus a branch in `build_regressor_sets`; it is not recommended.

### A.3 · Three numbers in the check file are wrong, and are corrected here

These were checked against the CSVs directly, not taken from a summary.

1. **N11's "leads free air by 40 to 80 min in both eras (Study 02, `PF_16_shift_summary`)".**
   `PF_16_shift_summary.csv` gives the `str`–`era5` air-temperature displacement as **−60 minutes in
   the legacy era and −40 in the current one**. The range is 40 to 60, it differs *by era* rather
   than spanning a range within each, and "in both eras" is wrong. The figure 40 to 80 does exist,
   but it belongs to Study 03 and to a different pair — the **inclination's lead over the station
   and ERA5 air temperatures** (40, 60, 60 and 80 minutes across the four era-source combinations of
   `TR_08`). N11 cites Study 02 for a Study 03 number about a different quantity.
2. **N11's appeal to humidity.** `PF_16_shift_summary.csv` scans `tair` and `sr` only; it contains no
   `rh` row. Nothing about relative humidity may be sourced to it.
3. **N12's expectation, "about 40 min, two slots, against radiation, and the wall leads by 1 to 1.5 h
   against air temperature".** Both numbers belong to the wrong pairs for the model they are meant to
   check. Model B is fitted on the **on-structure set**, whose radiation is `sr_gs` and whose air
   temperature is `tair_str`. `TR_08` gives `sr_gs` at **20 minutes, one slot**, and `tair_str` at
   **exactly zero** in both eras. The 40-minute figure is ERA5's radiation and the 1-to-1.5-hour lead
   is the station's and ERA5's air temperature — neither of which Model B reads.

The third correction is the most valuable thing Phase A produced, because it makes N11 and N12 the
same physical fact seen twice: the on-structure air probe sits in a sun-exposed housing, heats with
the sun as the wall does, and therefore shows **zero** displacement against the inclination while
free air shows an hour's lead. That is why its gain is a mixed air-and-radiation response, and it is
what Section 7's expectation must be written against.

### A.4 · Two defects found in other people's files, reported and not touched

- **Study 03's figure caption is stale.** `thermomechanical_response_study.py:796-799` says the
  sources agree "at about 40 minutes, once the station is placed at its interval centre by
  `GROUND_STAMP_OFFSET`", and the prose at 818-825 says "forty minutes exactly against ERA5 in the
  current era". `TR_08` shows `sr_gs` and `sr_str` agreeing at 20 minutes in the current era and
  ERA5 alone at 40. The caption asserts an agreement the table does not show. Study 03 has
  uncommitted work from another session; this is reported to the user and left alone.
- **Study 05's D3 guidance text misquotes Study 03**, which is item N10 and is fixed by B.2.

---

## 4 · Phase B · Every run-affecting change, before the notebook is executed

Nothing in Phase B is executed end to end. Each item is developed against the pre-Movement-5 state
dump described in the run-procedure memory, and unit-tested where it is library code.

**B.1 — `GROUND_STAMP_OFFSET` (N9).** No library work, per A.1. Add
`GROUND_STAMP_OFFSET = '15min'` to the parameter cell, pass it as `stamp_offset=GROUND_STAMP_OFFSET`
at the `load_ground_station` call site, and document it in that cell's Parameter Tuning Guidance:
name, purpose, accepted values, default, and effect downstream. The effect statement must say that
the `'str'` set borrows `sr_gs` and therefore inherits the correction, and that the stamping
convention is inferred rather than documented by the vendor. Leave `accumulations=()` for the
station as it is — the station reports interval means, not accumulations; it is the *stamp*, not the
accumulation handling, that was missing.
*Commit:* `fix(study05): stamp the ground station's interval means at their interval midpoint`.

**B.2 — Radiation delay in minutes, per column (N10).** This one does need a library change, which
A.2 established:
`proxies.build_regressor_sets(..., radiation_delay_h=1.0, freq='20min', delayed_role='sr')`
converts with `slots = int(round(radiation_delay_h / step_hours))` — a single delay in hours applied
to every set. Under R7 the two radiation columns need different delays, 20 minutes for `sr_gs` and
40 for `sr_era5`, so the parameter must become per-set.

- *Library:* extend `build_regressor_sets` to accept the delay in minutes, either per set or as a
  scalar that applies to all — a mapping keyed by set name is the smaller change, since the
  notebook's `REGRESSOR_SETS` is already keyed that way. **Every existing caller must keep its
  present behaviour**: the call sites are Study 05's notebook, Study 05's sidequest and two
  `shmlib` tests, so the old keyword stays accepted and the tests are re-run unchanged. Adding a
  new keyword beside the old one is preferred to changing the old one's meaning, and the two must
  not both be honoured silently — passing both is an error.
- *Notebook:* replace `RADIATION_DELAY_H = 1` with delays declared in **minutes** per set, and let
  the library convert to slots on the study's `NATIVE_FREQ`. Document in the Parameter Tuning
  Guidance that `sr_gs` takes 20 and `sr_era5` 40, that these are `TR_08_native_delay_summary.csv`'s
  current-era figures, that the legacy era puts `sr_gs` one slot later, and why the current era wins
  (R7).
- *Guidance text, D3:* correct the misquotation. Study 03's `TR_02_coupling_all.csv` gives one hour
  for `sr_str` alone, zero for `sr_gs` and `sr_era5`; the study's own explanation is that an hourly
  scan can only land on the hour, and the native step resolves what the hourly grid rounded. The new
  text quotes the native table, not the hourly one, and says which is which.

*Commits:* one for the library extension with its tests, one for the notebook parameters and D3's
guidance text.

B.1 and B.2 must both land before the run; they must not be split across it.

**B.3 — Coverage chart, `GM_F15` (N2).** *Input verified 2026-09-07:* the chart consumes the frame
`proxies.regressor_set_coverage` already returns — columns `set`, `role`, `accepted`, `filled`,
`coverage` — which is exactly what Table 1 prints as `GM_01`. No change is needed upstream of the
figure; `shmlib/figures.py` holds no bar-chart or histogram function today, so this is a new
function rather than an adaptation.

A new `shmlib.figures` function drawing what Table 1 now
holds: vertical bars grouped by source, stacked to separate accepted from filled slots, a left axis
in raw slot counts and a right axis in per cent — the two are proportional because the window's
total slot count is shared by every source — and `inclination` as the first group. The function
takes the coverage frame and returns the figure; the notebook calls it and exports under the `GM_`
convention. Identity colours are fixed by the repository's graphical guidelines, the legend goes
below the axes, and no data line carries an outline.
*Commit:* `feat(study05): draw the record's coverage by source as a grouped bar chart`.

**B.4 — Gap histogram, `GM_F16` (N7).** *Input verified 2026-09-07:* `prediction.gap_inventory`
already returns one row per maximal gap with an `n_slots` column, the exact count of missing
20-minute slots the note asks the bins to be defined on, alongside `duration_h` and `gap_class`.
The figure therefore consumes the per-gap frame `gaps`, not the `gap_classes` aggregate the
notebook currently builds for Table 2, and `gap_inventory` itself needs no change — its
`DEFAULT_GAP_CLASSES` binning stays for the table's own use and is bypassed by the histogram.

A new `shmlib.figures` function drawing the gap inventory as
a histogram on logarithmic axes: bins defined on the exact count of missing 20-minute slots rather
than on the broad duration categories the table used, a logarithmic x-axis for gap duration and a
logarithmic y-axis for gap count, and — the note's own emphasis — tick labels showing real
durations and real counts, never scientific notation or log values. Check whether
`shmlib.diagnostics` already owns a gap-histogram function before writing a second one; adapting an
existing function with a new optional parameter is preferred to a new one beside it, and any
adaptation must leave every existing caller's behaviour unchanged.
*Commit:* `feat(study05): draw the gap inventory as a log-log histogram of missing slots`.

**B.5 — D15, the radiation-filter sweep (N13).** The largest item.

- *Design document:* add decision D15, stating the lumped heat balance `tau·dx/dt = a·R(t) − x(t)`,
  its solution as a causal exponential filter with one time constant, the relationship to Study 03's
  delay-and-filter operator of which D3 imposes only the delay half, and the rule stated in advance:
  **the filter counts as an improvement only if the bootstrap bounds on its skill exclude zero.**
- *Library:* a `shmlib` function applying the exponential filter to a radiation column, with the
  delay set to zero once the filter is on, the filter state reset after every target gap longer than
  D13's fill limit, and a warm-up of about `3·tau` flagged rather than silently included. It takes
  `tau` as an argument; the sweep values live in the notebook.
- *Tests:* a skill test with a synthetic bootstrap — a known filter recovered, the reset honoured at
  a gap, the warm-up flagged, and the block-bootstrap increment reproducible under the seed.
- *Notebook:* `TAU_SWEEP_H` in the parameter cell over approximately 1, 2, 4, 8, 12, 24 and 48
  hours, applied to the station radiation, with the Parameter Tuning Guidance stating what each
  value means and that this is a diagnostic.
- *Outputs:* table `GM_10b` reporting, per `tau`, the held-out MAE with the paired block-bootstrap
  increment over the delay-only fit, the learned radiation gain beside Study 03's −0.035 mdeg per
  W m⁻², and the residual's daily-sideband power; one figure beside Model B.
- *Scope discipline:* this runs as a diagnostic under D9's own rule. Model A's main line stays on
  the delay operator, and **nothing in Sections 4 to 8 is re-derived from the filter.** If the sweep
  says yes with margin, promotion to the main line is Study 06's business or the paper's model
  choice, because it changes every attribution number.

*Commits:* one for the design document and library function with its tests, one for the notebook
cell and the exports.

*Delegation:* B.3 and B.4 are figure functions with a precise specification and go to
`cavecrew-builder` or a Sonnet subagent, spawned in parallel. B.5's library function and tests are
delegated with the specification above; its design-document text is written by the orchestrator, as
is every word of D15 that reaches the report.

**Gate before Phase C:** all sixteen suites green from `studies/`, and each new cell exercised
against the state dump. A full run is not started to find a typo.

---

## 5 · Phase C · One full run

Regenerate the notebook from the `.py` with `jupytext --to ipynb`, then execute it with
`nbconvert`/`nbclient` into the session scratchpad with the study folder as the working directory —
never into a copy inside the repository, which the folder-honesty test flags. Expect about 36
minutes at `N_JOBS = 32`.

Verification, in order:

1. **Error cells counted explicitly.** `nbconvert` exits 0 on a failing cell; the count of error
   cells is read from the executed notebook, not inferred from the exit status.
2. **Every `GM_` artefact present**, including the three new ones, `GM_F15`, `GM_F16` and `GM_10b`.
3. **`GM_15` run metadata** carries the new parameters — `GROUND_STAMP_OFFSET`, the per-source
   delays in minutes, `TAU_SWEEP_H` — since Section 14 prints that table verbatim.
4. **The executed notebook copied back** into the study folder only after the checks above pass.
5. **The serial control run is not repeated.** Task 7.2 established that the parallel path
   reproduces the serial path bit for bit on this machine; this pass changes parameters and adds a
   diagnostic, not the parallelism. If a `GM_` table looks wrong rather than merely different, that
   is the moment to reconsider.

Expect **every number in Sections 4 to 12 to move.** That is what N9 and N10 buy, and Phase D is
sized for it.

---

## 6 · Phase D · Prose, against the new numbers

No prose is written before Phase C, because most of it quotes numbers the run has just changed.
Every word here is the orchestrator's; none of it is delegated. One commit per note, or per
coherent group where two notes touch one paragraph.

**D.1 — §2, the record (N1, N3, N4, N6, N8).** Group these: they are the same few paragraphs.
Identify GS, STR and ERA5 at first appearance. Rewrite the coverage paragraph the note quotes, now
describing a chart rather than a table, and rewrite it as a statement about what each set holds
rather than as a recital of percentages. Sweep the surrounding text for every sentence that still
calls Table 1 a table or Table 2 a table. Remove every reference to the internal data-quality
report and replace it with a self-contained description of the gaps and the outages, since the
reader will never see that document. Replace the paragraph raising the DST doubt with a short
time-base statement: all three time bases are UTC before any comparison, the one-hour figure was
ERA5's end-of-hour accumulation stamp, the same logger gives two different answers so the
displacement is the sensor and not the clock, citing Study 02's `PF_16_shift_summary`, and no shift
is applied to any structure channel.
*Commits:* two — one for the coverage rewrite and the chart compliance, one for the time-base
statement and the self-contained gap description.

**D.2 — §2 and D3, the parameter changes (N9, N10 prose half).** State the station's stamping
correction and the per-source radiation delays where the sets are introduced and in D3, and update
every number the run moved.
*Commit:* folded into the run's own numbers commit.

**D.3 — §6 and D2, `tair_str` (N11, under ruling R1 and with A.3's correction).** State in D2 that
the `'str'` set keeps the wall's own air temperature deliberately, and say at the gains comparison
that this gain is a mixed air-and-radiation response. Use the numbers Study 02 actually reports: the
on-structure probe leads ERA5's air temperature by **60 minutes in the legacy era and 40 in the
current one** (`PF_16_shift_summary.csv`), because it sits in a sun-exposed housing and heats with
the sun, while the station's and ERA5's air temperatures agree with each other to within one slot.
Do not write "40 to 80 minutes in both eras", and do not source any statement about relative
humidity to `PF_16`, which scans only air temperature and radiation. The text must not call this
gain the wall's response to air temperature. Cross-reference Section 7, where the same fact appears
as a zero-lag kernel.
*Commit:* `docs(study05): say what the on-structure air temperature measures`.

**D.4 — §7, the expectation before the figure (N12, with A.3's correction).** Write as the
expectation the numbers that belong to the columns Model B actually reads. Model B is fitted on the
on-structure set, whose radiation is `sr_gs` and whose air temperature is `tair_str`, and
`TR_08_native_delay_summary.csv` gives, on the 20-minute grid and before Model B was fitted, a delay
of **20 minutes, one slot, against radiation** and **exactly zero against `tair_str`**. The check
file's 40 minutes and 1-to-1.5-hour lead describe ERA5's radiation and the station's and ERA5's air
temperature, which this model does not read; they belong in the comparison as context, not as the
expectation.

Then compare. A radiation kernel peaking near one slot, and an air-temperature kernel whose weight
sits at lag zero rather than before it, confirm that the on-structure probe and the wall are heated
by the same sun on the same schedule — which is the mixed response D.3 states, seen from the other
side. A radiation kernel peaking at zero means the one-slot delay was absorbed elsewhere, most
plausibly into that zero-lag air-temperature term, and the text must say so. State the
free-air contrast explicitly: against the station's and ERA5's air temperature the wall leads by 40
to 80 minutes, and it is the absence of that lead against `tair_str` that identifies the housing.
*Commit:* `docs(study05): state the delay Model B is checked against before its figure`.

**D.5 — §7, D15's result (N13 prose half).** Report what the sweep found against the rule stated in
advance, and say plainly if the bootstrap bounds include zero. Keep the framing the check file set:
a modest MAE gain at best is expected, since the wall probe measures this state directly and the
ladder found it buys nothing; the likelier wins are the sign of the on-structure radiation gain,
currently reported as wrong, and the annual sidebands of the daily cycle left in the residual.
State the competition between filtered radiation and air temperature — the two are close in shape,
so held-out skill and not the gain is the judge. Name what is out of scope and left to Study 06:
night-time longwave cooling to a clear sky, which the pyranometer cannot see and ERA5's surface net
thermal radiation could supply, and asymmetric heating and cooling constants.
*Commit:* `docs(study05): report the radiation-filter sweep against its stated rule`.

**D.6 — §11, the calibrated verdict in the table (N16, from ruling R3).** Minimum: extend Table 2's
caption so that the verdict column is explicitly labelled as the raw-band verdict and the reader is
pointed at the calibrated reading in the text. Preferred if `GM_14` is regenerated anyway in Phase
C: add a second verdict column computed against the calibrated half-width, so that the table alone
says what the study's own uncertainty doctrine says. Choose the second only if it lands in Phase B;
after Phase C, the caption is the whole item.
*Commit:* `docs(study05): label the bridge verdict as the raw-band reading`.

**D.7 — §13, the two limitations (N14, N15).** State that neither the azimuth of the monitored wall
face nor the mounting orientation of the on-structure pyranometer is recorded anywhere in the
project, that the 20-minute phase between the wall pyranometer and horizontal radiation and the
choice to treat the wall's own radiation as the physical forcing both rest on that geometry, and
that one field measurement would close it. State separately that the ground-station offset applied
at input is inferred from the displacement scan and the 30-minute export cadence, that the vendor
documents no stamping convention, and give the residual after correction as the bound on the
remaining time-base uncertainty.
*Commit:* `docs(study05): state the geometry and stamping limitations`.

**D.8 — Sweep every number the run moved.** Sections 4 to 12 quote gains, intervals, thresholds,
limits, run lengths and shifts that Phase C has changed. Read the report against the new `GM_`
tables end to end; this is the single largest piece of work in Phase D and it is not delegated.
*Commit:* `docs(study05): bring every quoted number onto the corrected run`.

---

## 7 · Phase E · The data-quality report (N5)

Independent of every other phase and of the run; may be executed at any point, and has its own
commit by the parent plan's rule that a note asking for a document outside the study is planned as
its own change.

Update `docs/data-quality-report-2026-08-10.md` so that it covers the gaps produced by the cleaning
procedure and not only the outages: how many slots the cleaning removes, their distribution against
the outage gaps, and how the two combine into the gap inventory the study reads. Note that N6
removes the report from the reader's path in the manuscript — the report remains the internal record
and the study text becomes self-contained; the two items do not conflict.
*Commit:* `docs: record the cleaning procedure's gaps in the data-quality report`.

---

## 8 · Closing verification

After Phase D, in this order:

1. All sixteen suites green from `studies/` — 350 tests plus whatever B.5's skill test adds.
2. The report rebuilt with `pdflatex`, clean, no unresolved references, no pending marker.
3. The honesty test re-run with the closure guards on.
4. Every checkbox in `report05_check.md` ticked, with the commit that closed it noted beside it.
5. The parent plan's Progress block and the ledger updated: Task 7.3 complete, 42 of 42.
6. `studies/README.md` and the study README checked for any statement the new numbers falsify.

---

## 9 · Repository hygiene, binding on every commit of this pass

**Foreign hunks must not be staged.** Another session's uncommitted work sits in
`studies/shmlib/figures.py` (`plot_reference_shift_scan`), `studies/shmlib/proxies.py`, and Studies
02 and 03; `studies/shmlib/temporal_alignment.py` and its test are untracked from that same work.
The user's `studies/05_greybox_monitoring/sidequest/`, the Study 06 plan and spec under
`docs/superpowers/`, `docs/references/`, `graphify-out/`, `environment.yml`, `verify_env.py`,
`auto_watcher instructions.md` and `.claude/settings.local.json` are all outside this pass.

Commit by pathspec, and remember that `git commit -- <path>` commits the **working-tree** version of
that path: `studies/shmlib/figures.py` carries foreign hunks and B.3's new function will land in the
same file, so that file needs a filtered patch and `git apply --cached`, never a plain pathspec
commit. The same applies to any other shared module this pass touches.

**The branch is `study05-greybox-monitoring`, fourteen commits ahead of `origin` and unpushed.**
Nothing in this plan pushes. Google Drive sync should be paused before any heavy git operation.

**The auto-watcher is usually running.** Edit the `.py`, then confirm the `.ipynb` picked the change
up before executing anything.

---

## 10 · Risks

- ~~**A.1 or A.2 comes back empty.**~~ Retired 2026-09-07: both numbers were settled elsewhere and
  recovered. The risk that replaced it is that A.3 refuted three numbers the check file states, so
  the notes and the plan now disagree with each other where A.3 says they do; the plan's readings
  are the ones sourced to a CSV, and the check file's boxes are ticked against the corrected text.
- **R7 is a judgement, not a measurement.** One delay per radiation column, taken from the current
  era, papers over a one-slot era difference in `sr_gs`. It is stated in the prose and can be
  overruled into an era-switched delay at the cost of a notebook parameter and a branch in
  `build_regressor_sets`.
- **The run moves a verdict, not just a number.** The station stamping correction and the per-source
  delays feed the `'str'` set's radiation, hence the gains, the intervals, the control limits and
  the bridges. A changed outage verdict or a changed alarm episode is a legitimate outcome and is
  reported as such; it is not a reason to revert a correction that the evidence supports.
- **D15's sweep says yes with margin.** Then Section 7 reports it, Section 13 records that the main
  line was not moved, and promotion becomes Study 06's or the paper's decision. The sweep does not
  get promoted inside this pass, however good it looks.
- **B.5 overruns.** It is the one item that could double the pass. If its library function and tests
  are not ready when B.1 to B.4 are, run Phase C without it and add a second, shorter run for the
  diagnostic alone — the sweep is declared a diagnostic beside Model B precisely so that it can be
  detached from the main line without invalidating anything.
