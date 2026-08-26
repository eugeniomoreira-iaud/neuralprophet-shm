# SDD ledger — plan: docs/superpowers/plans/2026-08-25-study04-decomposition-and-anomaly.md

Spec: docs/superpowers/specs/2026-08-25-study04-decomposition-and-anomaly-design.md (read 2026-08-25).
Branch: `study04-rebuild`, cut from `main` at `9ebcc5c` (working tree was clean; plan and spec
already committed in `9ebcc5c`, so the plan's "commit them before Task 1" step was already done).

## Preflight

### Verified against disk before execution

| Claim in plan | Checked | Result |
|---|---|---|
| `clean_up.py`, `replace_report.py` in study 04 root | `ls` | present |
| `replace_panels2.py`, `replace_panels3.py` at repo root | `ls` | present at repo root, not under `studies/` |
| Report line 86 holds the fabricated paragraph | `grep` | present; all three forbidden phrases on that one line |
| Report's only `\includegraphics` is `NP_F01` | `grep` | true; `outputs/NP_F01_on_structure_record.png` exists |
| `studies/README.md:32` is the study 04 row | `sed` | true |
| Study 04 root holds only the notebook `.py` after deletions | `ls` | true (`__pycache__` is a directory, not matched by `glob('*.py')`) |
| `prediction._as_series`, `hourly_change`, `contiguous_segments` exist | `grep` | L16, L25, L76 |
| `viz.INC_COLOUR`, `MARK_COLOUR`, `FIGURE_WIDTH`, `SPAN_STYLE`, `CHANNEL_COLOUR`, `figsize`, `format_spines`, `finish` | `grep` | all present |
| `score_predictions` current signature | `grep` | `score_predictions(frame, group_cols)` — Task 9's `naive_scale`/`alpha` are additive |
| Test files the plan re-runs exist | `ls` | `shmlib/tests/test_shmlib.py`, `04.../tests/test_prediction.py`, `03.../tests/test_shmlib_study03.py`, `02.../tests/test_shmlib_study02.py` |
| `shmlib/__init__.py` exports a module list | `cat` | yes — but it does **not** list `prediction`; submodule import still works |

### Cross-task interface table

| Tasks | Shared file / interface | Finding |
|---|---|---|
| 2, 3, 4 | append to `shmlib/prediction.py`; all test in `test_gaps.py` | consistent; cumulative test counts 4 → 8 → 12 → 14 match the classes each task adds |
| 3 → 4, 5 | `contiguous_segments`, `hourly_change` | both exist with the signatures the plan calls |
| 4 | calls `_as_series` in its body, not declared in Consumes | exists; prose omission only, no defect |
| 5 → 6 | three `figures.plot_*` functions | consistent |
| 5 | `figures.py` importing `prediction` | plan uses a function-local import, so no import cycle with `__init__` |
| 7, 8, 9 | `prediction.py` line ranges `:213-285`, `:454-551`, `:554-603` | Tasks 2–4 append at end of file, so these ranges do not shift |
| 10 | `shmlib/__init__.py` "add monitoring if a list exists" | list exists |
| 12 | `test_decomposition.py` created by Task 7, appended by 12 | consistent |
| 3 → 17/18 | "Task 18 reads `n_windows` to choose `n_lags`" | **conflict:** Task 18 is gap closure; Model B and its `n_lags` are Task 17 |
| 13 → 14 | Task 14 consumes `predictions_a` | **conflict:** Task 13's Produces lists only `model_a`, `components_a`, `residual_a` |
| 13, 14 | both list `NP_F06_daily_cycle_and_response.{png,svg}` as Produces | **conflict:** one artefact, two owners |
| 17 → 18 | Task 18 consumes `segmented_b`, `hourly` | **conflict:** Task 17's Produces lists only `predictions_b` |
| 1 | `test_folder_honesty.py` README regex | vacuous once the README is rewritten (no `NP_` tokens remain) — a guard, not a live assertion |
| 5 | legend-position test on `plot_gap_anatomy` | **plan-mandated near-vacuous test:** that figure draws no legend, so the loop body never runs |
| 5 | `python 02_proxy_forcing_characterization/tests/*.py` | **defect:** a shell glob passed to `python` runs only the first file and treats the rest as `argv`. Works by luck today (one file), breaks silently when a second is added |
| 1 | "replace lines 44–45" of the notebook | the comment block actually spans lines 43–45 |

### Rulings

- **Ruling P1 (subagent roles):** `cavecrew-builder` carries only Read/Edit/Write/Grep/Glob — no
  Bash — so it cannot run the tests or make the commit that every implementation task in this plan
  requires, and it hard-refuses 3+ file scope. Implementers are therefore `general-purpose` agents
  pinned to `sonnet`, with "respond in caveman ultra mode" in every spawn prompt, per
  `instructions-core.md`. `cavecrew-investigator` is used for read-only location work and
  `cavecrew-reviewer` for task reviews, both on `sonnet`. *Cost if wrong:* subagent output is
  slightly longer than cavecrew's contract would give; no effect on the code produced.
- **Ruling P2 (Task 1 notebook edit):** replace the whole three-line comment block at lines 43–45,
  not the literal lines 44–45, since the plan's replacement text restates the sentence that begins
  on line 43. *Cost if wrong:* a stray half-sentence in a comment, visible in the diff.
- **Ruling P3 (Task 3 prose):** the task that reads `n_windows` to choose `n_lags` is Task 17
  (Model B), not Task 18 (gap closure). Carried into Task 17's dispatch. *Cost if wrong:* none to
  code; a pointer in a docstring aims at the wrong task number.
- **Ruling P4 (Task 13 → 14):** Task 13 must also expose `predictions_a` — the backtest output it
  already produces on the way to `components_a`. Carried into Task 13's dispatch as a required
  notebook variable. *Cost if wrong:* Task 14 stalls on a missing variable and the fix is one line.
- **Ruling P5 (`NP_F06` ownership):** the figure is written once, by whichever of Tasks 13 and 14
  actually emits it in its steps; if both do, Task 14 owns it and Task 13 drops it. Resolved
  against the task text when Task 13 is dispatched. *Cost if wrong:* a figure written twice, the
  second overwriting the first with identical content.
- **Ruling P6 (Task 17 → 18):** Task 17 must expose `segmented_b` and `hourly` alongside
  `predictions_b`. Carried into Task 17's dispatch. *Cost if wrong:* Task 18 stalls on a missing
  variable.
- **Ruling P7 (Task 5 glob):** the verification command becomes an explicit per-file loop
  (`for t in 02_proxy_forcing_characterization/tests/test_*.py; do python "$t"; done`) rather than
  `python .../tests/*.py`. *Cost if wrong:* none — strictly more tests run than the plan asked for.
- **Ruling P8 (Task 5 legend test):** kept as the plan writes it. It is a guard that becomes live
  the moment a legend is added to that figure, and rewriting a plan-mandated test to assert
  something else is a larger change than the finding warrants. Recorded so the final review sees
  it. *Cost if wrong:* one test in the suite proves less than its name suggests.

Graph note: `studies/graphify-out/graph.json` was built 2026-08-22 and its line numbers run about
two lines behind the working tree. Used for orientation only; every edit target is re-confirmed by
grep before it is touched.

- **Ruling P9 (interpreter):** `python` on PATH is `/Users/eugenio/anaconda3/bin/python`, which has
  no `neuralprophet` — running the suite with it produces two spurious `ModuleNotFoundError`
  errors in `test_prediction.py`. Every test run in this plan uses
  `/Users/eugenio/anaconda3/envs/neuralprophet_env/bin/python` (neuralprophet 0.8.0, Python
  3.10.20) by absolute path, since `conda activate` does not survive a non-interactive shell. This
  path goes into every implementer dispatch. *Cost if wrong:* a subagent reports failures that are
  environment artefacts rather than defects.
- **Ruling P10 (report PDF):** Task 1's file list does not mention
  `report/neuralprophet_inclination_prediction_report.pdf`, but the committed PDF was built from
  the `.tex` that carried the fabricated paragraph, so leaving it would keep the claim alive in the
  folder's actual deliverable — exactly what the task exists to prevent. Rebuilt twice with
  `pdflatex -interaction=nonstopmode` (exit 0, no errors) from the corrected source and committed
  with the task. *Cost if wrong:* a binary file in the diff that the plan did not ask for; revert
  it and the `.tex` still stands corrected.
- **Ruling P11 (`.claude/settings.local.json`):** left uncommitted. Its diff is two permission
  entries this session's tool approvals added — harness configuration, not study work, and not
  Task 1's business. *Cost if wrong:* the user re-approves two commands in a later session.

## Progress

Task 1: complete (commit 5ac1df9, orchestrator-run per plan, review clean — see below)
- Deleted `clean_up.py`, `replace_report.py` (study 04 root) and `replace_panels2.py`,
  `replace_panels3.py` (repo root); all four were tracked, so `git rm` applied to each.
- Removed the fabricated paragraph from `report/…_report.tex` (was line 86) and rebuilt the PDF.
- Replaced the stale three-line comment in the notebook source; `jupytext` synced the `.ipynb`
  (diff confirmed to be that comment and nothing else).
- Rewrote the study `README.md` and corrected `studies/README.md:32`.
- `test_folder_honesty.py`: 3 failures before the changes, `OK` (4 tests) after.
- Regression, all in `neuralprophet_env`: `test_prediction.py`, `test_shmlib.py`,
  `test_shmlib_study03.py`, `test_shmlib_study02.py` — all `OK`.
- No separate task review dispatched: the plan assigns Task 1 to the orchestrator, and its own
  hygiene tests are the acceptance gate. The final whole-branch review covers this diff.

Checkpoint 0: shown to the user, approved 2026-08-25 ("go on"). Phase 1 started.

Task 1 ledger tick committed separately as `7413880`.

Task 2: dispatched (general-purpose, sonnet, caveman ultra). BASE `7413880`.
Brief `task-2-brief.md`, report `task-2-report.md`. Briefs for Tasks 3–5 pre-generated.

### Rulings P4, P5, P6 resolved against the plan's step text (no code change needed)

Read while Task 2 was running. All three "conflicts" are errors in the tasks' **Files/Interfaces
headers**, not in the code the steps specify:

- **P4 — `predictions_a`:** Task 13's step code already binds it (`model_a, predictions_a =
  fits[MODEL_A_YEARLY]`, plan line 3588). Nothing to add; Task 13's dispatch only needs to say the
  variable must stay in scope for Task 14.
- **P5 — `NP_F06`:** Task 13's only figure call writes `NP_F05_decomposition_stack` (plan line
  3636); Task 14's writes `NP_F06_daily_cycle_and_response` (line 3856). **Task 14 owns `NP_F06`**;
  Task 13's header lists it wrongly. No duplicate write exists.
- **P6 — `segmented_b`, `hourly`:** Task 17's step code creates both (`hourly` at plan line 4376,
  `segmented_b` at 4388). Header omission only.

Consequence: the preflight found no real cross-task defect. P3 (Task 3 docstring pointing at Task
18 instead of 17) and P7 (the `python .../tests/*.py` glob) remain the only substantive plan
corrections, and both are cosmetic in effect.

Task 2: implementer returned DONE, commit `80b03eb`. Reported `test_gaps.py` 4/4, `test_shmlib.py`
60/60, `test_prediction.py` 20/20, `test_shmlib_study03.py` 44/44, all OK; no concerns; only the
two named files staged. Review package `review-7413880..80b03eb.diff` (1 commit, 6,334 bytes).
Task review dispatched to `cavecrew-reviewer` on sonnet.

Task 2: review returned spec ✅, quality NOT approved — 1 finding (0🔴 1🟡 0❓). Reviewer confirmed
the diff matches the brief's mandated code byte-for-byte, confirmed the insertion point between
`contiguous_segments` and `expanding_segment_folds`, and hand-traced the run-boundary arithmetic
against all four test cases plus the first-gap, last-gap and single-slot boundaries.

- **Ruling P12 (Task 2 finding, plan-mandated code):** `gap_inventory` assigns
  `label = classes[-1][2]` before scanning the bins, so a duration matching no bin is silently
  labelled with the last class instead of flagged. Unreachable under `DEFAULT_GAP_CLASSES`, whose
  five bins tile `(0, inf]`, and the study never passes custom bins — but `classes` is public and
  the docstring documents neither the fallback nor a coverage requirement. **Fixed in the docstring
  only; the executable code is left byte-identical.** Changing the fallback to a sentinel would
  move the returned `gap_class` values, which the plan's tests and the study's `NP_02` table are
  written against, and the plan mandates this implementation verbatim. *Cost if wrong:* a caller
  who passes non-tiling custom bins gets a mislabelled row instead of an error — documented rather
  than prevented.

Task 2: fix round 1/5 (1 addressed, 0 open — gap_class fallback documented; commits 80b03eb..cdecba2).
Fix verified docstring-only by direct diff inspection before re-review; scoped re-review (haiku)
returned ADDRESSED at `prediction.py:149-151`, no executable code changed, no new issues.

Task 2: complete (commits 80b03eb..cdecba2, review clean)

Task 3: dispatched (general-purpose, sonnet, caveman ultra). BASE `cdecba2`.
Brief `task-3-brief.md`, report `task-3-report.md`. Dispatch carries Ruling P3 (the brief's
"Task 18 reads `n_windows`" is a numbering error for Task 17) and the state of `prediction.py` and
`test_gaps.py` after Task 2.

Task 3: implementer returned DONE, commit `56d1655`. Reported `test_gaps.py` 8/8, `test_shmlib.py`
60/60, `test_prediction.py` 20/20, `test_shmlib_study03.py` 44/44, all OK; no concerns.
Review: spec ✅, quality **approved** (0🔴 1🟡 0❓). Reviewer independently re-derived the brief's
worked example (two 12-slot segments, `need=6` → 14 windows) and traced the empty-input path
(`np.median`/`.max()` never reached, `windows=0` via the `usable.size` guard).

Task 3: minor (deferred): `prediction.py:229` — `int(lag)`/`int(horizon)` truncate the values
stored in the `lag_hours`/`forecast_hours` columns, while `need` is computed from `float(lag)`/
`float(horizon)`. A caller passing a fractional hour would get window counts computed from the true
value but labels reporting the truncated one. Latent: every call site and test passes plain ints,
and the documented contract is int-only. Not entered into the fix loop — the reviewer approved
quality, the code is the plan's verbatim mandate, and the same docstring-versus-behaviour trade was
already ruled on in P12. Flagged here for the final whole-branch review to triage.

Task 3: complete (commits cdecba2..56d1655, review clean)

Task 4: dispatched (general-purpose, sonnet, caveman ultra). BASE `56d1655`.
Brief `task-4-brief.md`, report `task-4-report.md`. Dispatch tells the implementer to report rather
than adjust if any of the brief's assertions fail on the transcribed code — this is the task whose
table decides the study's cadence and target, so a silently tuned test would corrupt a design
decision rather than a convenience.

Task 4: implementer returned **BLOCKED** on the first attempt, correctly. Transcribed both blocks
verbatim; `test_gaps.py` ran 12 tests with 1 failure —
`TestCadenceEvidence.test_recovers_the_sign_of_a_known_coupling`, `corr_level` at `1h` = −0.753
against an asserted `< -0.9`. All three regression suites passed. It committed nothing and reported,
exactly as instructed.

- **Ruling P13 (Task 4, defective test fixture in the plan):** verified the arithmetic
  independently before ruling. In `TestCadenceEvidence._pair` the drift term reaches
  `0.01 * 4319 = 43.19` across the record while the coupled sinusoid it is added to has a
  peak-to-peak of `4 * 10.0 = 40`, so drift carries more variance than the coupling and caps
  `corr_level` at **−0.754** at both cadences. The assertion `< -0.9` could never pass;
  `corr_change` is −1.0 because differencing cancels a linear drift, which is why only the level
  half failed. The implementation is correct and stays byte-identical.
  **Fixed the fixture, not the threshold:** drift constant `0.01 → 0.002`, giving
  `corr_level = -0.985`. Loosening the threshold instead would have left the study's only sign-and-
  magnitude check guarded by an assertion weakened to fit a bad fixture — and this is the test
  behind design decisions D1–D3. Checked the class's other three tests against the change first:
  `drift_per_year` is asserted nowhere, the white-noise test is driven by its injected noise, and
  the era-boundary test compares `n_change` counts; none depends on the drift constant.
  *Cost if wrong:* the synthetic coupling in one unit test is cleaner than the real record's
  −0.865, so the test proves the function recovers a strong coupling rather than a marginal one.

Task 4: fix round 1/5 (1 addressed, 0 open — fixture drift `0.01 → 0.002`; commit `0434706`).
Verified on disk that the comment landed and only that constant moved.

Task 4: review returned spec ✅, quality NOT approved — 1 finding (1🟡). Reviewer confirmed the
authorised deviation was carried out exactly (four assertions byte-identical, `cadence_evidence`
byte-identical to the brief, correctly placed) and cleared the era guard, the shared
driver/response grid, the `drift_per_year` units and the `dropna()` pairing.

- **Ruling P14 (Task 4 finding, hardening a brand-new function):** `level.autocorr(1)` correlates
  positionally, so a series whose index OMITS timestamps — rather than carrying them as NaN —
  makes `level_autocorr1` pair samples that are not adjacent in time. That number is what decides
  D1, D2 and D3. Checked the exposure first: `proxies.harmonise` reindexes every column onto
  `pd.date_range(start, end, freq=freq)`, so this study's notebook always passes a dense frame and
  the defect is latent here. **Took the code fix anyway, not the docstring-only route used in
  P12.** The difference from P12 is that `cadence_evidence` is new and has no existing callers, and
  reindexing an already-dense input is provably a no-op — so the change cannot move any current
  result, while leaving it would plant a silent wrong number for anyone reusing this library on
  another record, which is the whole point of a transferable methodology. The fix is required to
  prove the no-op property: all four existing assertions must pass untouched, and a new test
  asserts that a row-sparse index and its NaN-filled equivalent now agree on `level_autocorr1`.
  *Cost if wrong:* one extra reindex per call, and a 13th test to maintain.

Task 4: fix round 2/5 (1 addressed, 0 open — inputs reindexed onto a complete grid; commit
`dc32058`). Before dispatching the re-review I loaded the pre-fix `prediction.py` from `0434706`
alongside the post-fix module and ran both on the same data: pre-fix the sparse and NaN-filled
inputs give `level_autocorr1` = 0.996114150 vs 0.996105268, post-fix both give 0.996105268. The new
test is therefore a genuine guard rather than a vacuous one, and the fix does what it claims. The
divergence is small on a 200-row fixture with three rows dropped; on the real record — 1,647 gaps,
24.4 % of slots missing — it would be far larger.
Scoped re-review: ADDRESSED at `prediction.py:302-306`, 0🔴 0🟡 0❓. Era alignment specifically
cleared: `_as_series` reindexes era by timestamp label rather than position, so gap rows become NaN
labels and `hourly_change`'s `notna()` guard excludes them — no difference can cross an instrument
era boundary. Empty-index call verified to return the documented ten-column shape without raising;
resampled branch unchanged; the four original assertions byte-identical.

Task 4: complete (commits 56d1655..dc32058, review clean)

Task 5: dispatched (general-purpose, sonnet, caveman ultra). BASE `dc32058`.
Brief `task-5-brief.md`, report `task-5-report.md`. Dispatch carries three corrections the brief
cannot know: the expected test count is now 15 rather than 14 (Task 4's guard test), Ruling P7
replaces the `python .../tests/*.py` glob with a loop, and the function-local
`from shmlib import prediction` in `plot_gap_anatomy` must stay inside the function body because
`shmlib/__init__.py` imports `figures` at package-import time.

Task 5: implementer returned DONE, commit `e4dd4c7`. I ran all six suites myself as checkpoint
evidence — `test_gaps.py` (15), `test_prediction.py`, `test_folder_honesty.py`, `test_shmlib.py`,
`test_shmlib_study03.py`, `test_shmlib_study02.py` — all `OK`. Traced the `RuntimeWarning` the
implementer mentioned: `np.corrcoef` dividing by a zero standard deviation, because the brief's own
figure-test fixture passes a pure ramp as the driver, whose hourly change is constant. It arises in
mandated test code, not in the new functions, and is benign.

Review: spec ✅ (both blocks byte-identical to the brief, diffed programmatically), quality NOT
approved — 1🟡 1🔵.

- **Ruling P15 (Task 5 finding, line-style exhaustion):** `plot_segment_survival` selects its line
  style with `position == 0`, so every group after the first draws the same dashed stroke while
  colour stays fixed at `viz.INC_COLOUR`. Exposure checked first: the notebook passes
  `SURVIVAL_FORECAST_HOURS = (8, 24)`, two horizons, so nothing in the study is wrong today.
  **Took the fix** — cycling `('-', '--', ':', '-.')` by `position % len(styles)` returns exactly
  `'-'` and `'--'` for two groups, so the study's figure is unchanged pixel for pixel, while a third
  horizon becomes legible instead of hidden under the second. Same no-op-on-intended-input
  reasoning as P14, and the repository's figure convention explicitly assigns grouping to line
  style because colour is already spent on channel identity. A guard test asserting three groups
  draw three distinct styles comes with it. *Cost if wrong:* one more test to maintain; the study's
  own figure cannot change.
- Task 5: minor (deferred): the 🔵 was report prose only — `task-5-report.md` named `plot_drift_scan`
  as the preceding function where the file actually ends at `plot_operator_grid`. Placement was
  correct; the sentence was not. Folded into the fix round since the report is being appended to
  anyway.

Task 5: fix round 1/5 (2 addressed, 0 open — style cycling and the report sentence; commit
`f9d7ffc`). Verified the no-op myself by rendering: two groups give `['-', '--']`, identical to
pre-fix, three give `['-', '--', ':']`. Scoped re-review (haiku) confirmed both ADDRESSED, the new
test genuinely discriminating, and colour/marker/linewidth/legend/labels all unchanged.

Task 5: complete (commits dc32058..f9d7ffc, review clean)

Task 6: complete (commit `e4f35e0`, orchestrator-run per plan). Ledger tick `95edd35`.

- **Ruling P16 (Task 6 import ordering):** the plan's Step 1 adds `monitoring` to the notebook's
  imports cell, but the plan's own File Structure table lists `studies/shmlib/monitoring.py` under
  *Created* — by Task 10, four tasks later. Importing it at Task 6 would make the notebook
  unrunnable. **Omitted `monitoring` from the imports cell**; `adc` and `tables` were added as the
  plan asks, since both already exist. `monitoring` joins when Task 10 creates it. *Cost if wrong:*
  one import line to add later, which Task 10 or Task 15 must not forget.
- **Ruling P17 (Task 6 vs the standing `NP_F01`):** the plan's replacement parameter cell drops
  `sr` and `twall` from `STR_MAP_CURRENT` and cuts `ON_STRUCTURE_COLUMNS` to three panels. The
  notebook's existing `NP_F01` cell would then have overwritten the standing five-panel record
  figure with a three-panel one — and `NP_F01` is not in Task 6's Produces list, while the spec's
  Decisions Taken (section 10.3) states that "the standing introduction and `NP_F01` are sound and
  reused". The report's surviving caption also describes five channels. **Kept both channel maps
  and the five-panel `ON_STRUCTURE_COLUMNS`;** `PREDICTOR_COLUMNS = ('tair', 'rh')` governs every
  model and the segment-survival requirement, so D11 is untouched — it excludes those channels from
  *models*, not from a figure describing what the package records. Verified after execution that
  `NP_F01` still renders five panels. *Cost if wrong:* the record figure and `NP_01` carry two
  channels the study does not model, at no analytical cost.
- **Ruling P18 (reference-table reproduction):** the notebook reproduces section 2.4 on every
  quantity a decision rests on — 1,647 gaps exactly, level autocorrelation 0.99771 exactly, change
  autocorrelation −0.0581 and +0.3494 exactly, `corr(inc, tair)` −0.8647 exactly, 72.80 % of missing
  time in the four outages over seven days, median segment 1.67 h, and 101 segments giving 15,000
  windows at a 24 h lag with a 24 h horizon, all exact. Two entries differ: accepted inclination
  slots are 62,931 against 62,937, and missing hours 6,791 against 6,789. The window is exactly the
  reference's 83,305 slots, and the two differences are one fact — six slots this pipeline masks
  that the design-time script accepted, 0.0095 % of accepted values. Because the gap *count* is
  identical, those six extended existing gaps rather than creating new ones. **Accepted as a
  reproduction rather than chased.** No decision moves: coverage reads 75.5 % rather than 75.6 %
  in the first decimal and nothing else shifts. The notebook's outputs are produced by the study's
  own versioned library and are now the authoritative measurement; where the report quotes a
  number it quotes `NP_01`–`NP_04`, not section 2.4. *Cost if wrong:* the report's coverage figure
  differs in the first decimal from the design document, which the user may prefer to reconcile.

Checkpoint 1 and 2 evidence, verified by direct inspection of the rendered figures: `NP_F02` shows
the count dominated by sub-hour dust and the missing *time* dominated by the four long outages at
73 %; `NP_F03` draws its legend below the axes with the two horizons separated by line style, as
the binding convention requires; `NP_F04` confirms D1, D2 and D3 rather than contradicting them —
level autocorrelation at ~1.0 on both grids, change autocorrelation negative at 20 min and +0.35
hourly, coupling strengthening from −0.78 to −0.89.

