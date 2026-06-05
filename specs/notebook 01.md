# Spec — Notebook 01 · Gap Classification, Visualization, and Usable-Stretch Selection (single station)

> **How to use this document.** Spec-driven development guide. It maps one-to-one onto the notebook's cells (Title → Imports → Step 1…N → Save) per the repo's Notebook Structure Standard. Build by satisfying each step's *Contract* and *Validation*; the *Acceptance Criteria* (§8) are the definition of done.

> **Scope (read first).** One station, **no proxy**, **no imputation of any kind**. NB01 only: validates the hourly series, classifies gaps, visualizes the variables and gaps, and selects the largest usable stretch *by classification*. Resampling to hourly is done in NB00; proxy join and timezone reconciliation happen in NB02; all gap filling happens in NB03.



---

## 1. Purpose & Pipeline Position

**Goal.** From one station's hourly series, produce (a) a fully gap-classified series with provenance flags, (b) diagnostic figures showing the variables and the gaps colored by class, and (c) the largest **usable stretch** — the longest window that contains only observed, micro, and meso gaps (i.e. bounded by macro/severe gaps).

**Upstream.** `00_Sensor_Preprocessing` → `data/interim/sensor/{station}_preprocessed.csv`, already **hourly**, NaN where missing, including on-site temperature and RH.
**Downstream.** `02` aligns the proxy onto the usable stretch and computes lags/screening; `03` imputes the micro/meso gaps inside the stretch (making it continuous) and the operational gaps of the full series; `04` trains the frozen baseline on the imputed stretch.

**Out of scope.** No proxy handling, no timezone reconciliation against a proxy, no imputation or bridging of *any* gap (micro included), no modelling. NB01 never alters a target value.

---

## 2. Inputs

| Input | Path | Contract |
|---|---|---|
| Station series | `data/interim/sensor/{TARGET_STATION}_preprocessed.csv` | `datetime` → `DatetimeIndex`, **hourly**; inclination (`TARGET_COL`) + on-site `AUX_COLS` (temperature, RH); NaN = missing; consistent units; documented `WORK_TZ`. |



---

## 3. User Parameters

| Parameter | Purpose | Accepted values | Default | Downstream effect |
|---|---|---|---|---|
| `TARGET_STATION` | Section selector / output namer (transferability switch) | e.g. `'st02'` | — | Names every artifact; re-run per station |
| `TARGET_COL` | Inclination column (gap classification target) | str | `'absinc'` | The series whose gaps are classified |
| `AUX_COLS` | On-site environmental columns for the viz | dict | `{'temp':'temp_c','rh':'rh_pct'}` | Plotted; not classified |
| `WORK_TZ` | Timezone of the series | IANA tz | `'UTC'` | Asserted; carried to NB02 |
| `EXPECTED_FREQ` | Expected step | offset | `'1h'` | Index-integrity assertion |
| `STUCK_TOL`, `STUCK_LEN` | Flatline detector | float, int | `STUCK_LEN`=6 | Dead-sensor runs counted as missing |
| `PHYS_BOUNDS` | Optional plausibility bounds for `TARGET_COL` | (lo,hi) or None | None | Out-of-bounds → NaN + flag |
| `GAP_THRESHOLDS` | Duration cut-offs (hours) | dict | `{micro:6, meso:48, macro:336}` | Defines micro/meso/macro/severe; couples to NB03 |
| `STRETCH_ALLOWED_CLASSES` | Gap classes permitted *inside* a usable stretch | set | `{'micro','meso'}` | macro/severe become stretch boundaries |
| `MIN_OBSERVED_FRAC` | Optional floor on observed fraction of a stretch | float or None | None | Rejects long-but-holey windows (see §9) |
| `NEIGHBOR_WINDOW` | Window for neighbour-support stats per gap | hours | `168` | Imputability precondition for NB03 |
| `GAP_CLASS_COLORS` | Colors for the shaded gap windows | dict | micro/meso/macro/severe → 4 colors | Figure styling |
| `STRETCH_SELECT` | Which usable stretch to designate | `'longest'` or stretch_id | `'longest'` | Override if longest is seasonally poor |

---

## 4. Outputs

| Artifact | File | Content |
|---|---|---|
| Flagged full series | `data/interim/sensor/{station}_flagged.csv` | `TARGET_COL`, `AUX_COLS`, `is_observed`, `quality_flag`, `gap_id`, `gap_class` (per-timestamp; empty when observed) — for NB03 |
| Usable stretch | `data/interim/sensor/{station}_analysis_stretch.csv` | The selected window sub-series **with its micro/meso gaps left intact** + flags — NB03 fills it, NB04 uses it |
| Gap inventory | `outputs/tables/01_01_{station}_gap_inventory.csv` | One row per gap (schema below) |
| Stretch table | `outputs/tables/01_02_{station}_stretch_table.csv` | Ranked candidate stretches (schema below) |
| Series + gaps figure | `outputs/figures/01_01_{station}_series_and_gaps.png` | The requested inclination/temp/RH view with gap windows by class + selected stretch |
| Gap histogram | `outputs/figures/01_02_{station}_gap_histogram.png` | Gap-duration distribution by class |

**Gap-inventory row schema:** `gap_id, start, end, duration_h, duration_samples, support_before_h, support_after_h, gap_class, onset_after_extreme`.
**Stretch-table row schema:** `stretch_id, start, end, length_h, n_observed, n_micro, n_meso, observed_frac, months_covered, selected`.

---

## 5. Step Specification

### Step 1 · Load Station Series
**Contract.** `dataloader.load_preprocessed_sensor(path, target_col, aux_cols) -> pd.DataFrame` (inclination + aux, `DatetimeIndex`).
**Validation.** `TARGET_COL` and `AUX_COLS` present; report span, total samples, observed fraction per column.

### Step 2 · Index Integrity
**Contract.** `preprocessing.assert_regular_grid(df, freq=EXPECTED_FREQ, work_tz=WORK_TZ) -> pd.DataFrame` (NaN inserted at any absent hour so gaps are explicit; no resampling).
**Validation.** Monotonic, unique, regular at `EXPECTED_FREQ`; tz == `WORK_TZ`; else hard fail (NB00's contract is checked, not patched).

### Step 3 · Quality Flags
**Contract.** `diagnostics.flag_quality(series, stuck_tol, stuck_len, bounds=PHYS_BOUNDS) -> (series, quality_flag, qc_report)` — flags stuck/flatlined runs and optional out-of-bounds as effectively missing, without overwriting raw values.
**Validation.** `quality_flag` ∈ {`ok`,`stuck`,`out_of_bounds`}; flagged values set missing for gap accounting; QC report counts each flag.

### Step 4 · Gap Inventory & Taxonomy (classification core)
**Taxonomy (defaults, configurable):**
- **micro** ≤ 6 h — sub-diurnal; high autocorrelation.
- **meso** > 6 h and ≤ 48 h — spans 1–2 diurnal cycles.
- **macro** > 48 h and ≤ 336 h (2 weeks).
- **severe** > 336 h — seasonal-drift risk.
**Contract.** `diagnostics.build_gap_inventory(series, thresholds=GAP_THRESHOLDS, neighbor_window=NEIGHBOR_WINDOW) -> gap_df`; also stamp a per-timestamp `gap_class` onto the flagged series.
**Validation.** Class counts reconcile with total missing samples.

### Step 5 · Per-Gap Onset Flag
**Goal.** The one testable cause signal from a single series.
**Contract.** `diagnostics.flag_onset_after_extreme(series, gap_df, criterion) -> gap_df` — marks gaps that begin immediately after a boundary/extreme reading (possible informative missingness — the safety-relevant case).
**Validation.** `onset_after_extreme` merged into the inventory. *(An optional missingness-vs-time/season association test may be added but is not required — keep NB01 to classification.)*

### Step 6 · Usable-Stretch Selection (by classification)
**Goal.** Find the largest window that contains only `STRETCH_ALLOWED_CLASSES` gaps; macro/severe gaps are the boundaries.
**Contract.** `diagnostics.find_usable_stretches(series, gap_df, allowed=STRETCH_ALLOWED_CLASSES, min_observed_frac=MIN_OBSERVED_FRAC) -> stretch_df` — segments the timeline at macro/severe gaps, keeps segments meeting the observed-fraction floor, ranks by length.
**Validation.** Stretch table ranked with `observed_frac` and `months_covered`; `STRETCH_SELECT` honoured; the designated stretch contains **no** macro/severe gap; the saved stretch retains its internal micro/meso gaps as NaN (no filling).

### Step 7 · Visualization
**Goal.** The requested view: inclination, temperature, RH, with gaps as colored windows by class, and the selected stretch marked.
**Contract.** `viz.plot_series_with_gap_windows(df, target_col, aux_cols, gap_df, class_colors=GAP_CLASS_COLORS, stretch=selected, save_plot_path=None) -> fig`.
**Design.** Three stacked panels sharing the time x-axis (inclination / temperature / RH) plotted as **scatter** points (so gaps read as absent points); vertical bands spanning all panels shade each inclination gap by class; the selected usable stretch is bracketed/highlighted across panels; legend maps colors to classes. Stacked panels (not one axis) because the three variables have incompatible scales. Also produce the gap histogram.
**Validation.** Figure renders with all three panels, class-colored bands aligned across panels, the stretch marked, and a legend; saved under the naming convention.

### Step 8 · Save (gated)
**Contract.** `if step_ok: save(...)` for all §4 artifacts following `{notebook_id}_{artifact_id}_{station}_{description}.{ext}`.
**Validation.** All artifacts exist; flagged series and stretch re-load with index/tz intact.

---

## 6. Methodological Rules Enforced Here
- **No target modification at all.** NB01 classifies and selects; it never interpolates, bridges, or fills any gap. The usable stretch is a *window selection*, saved with its internal gaps intact.
- **Provenance starts now.** `is_observed`, `quality_flag`, `gap_id`, `gap_class` originate here and must propagate through 03–04; the detection firewall depends on them, and NB04 can use them to exclude later-imputed points from false-alarm statistics.
- **NB00's contract is checked, not patched.** Non-hourly / wrong-tz input is a hard fail.
- **Single-station, transferable.** No site-specific constant outside the parameter block; no cross-station reference.

---

## 7. Edge Cases & Failure Modes
- Input not regular hourly / wrong tz → hard fail (Step 2).
- Temperature/RH absent from the station file → viz cannot be built here; move viz to NB02 (see §Assumption).
- Stuck/flatlined runs not stored as NaN → caught in Step 3.
- Longest candidate stretch is mostly meso gaps (holey) → rejected/penalised via `MIN_OBSERVED_FRAC`; surfaced by `observed_frac` (see §9).
- Aux channels (temp/RH) have different gaps than inclination → show their own missing points in their panels; classification windows are driven by inclination only (state this in the figure caption).
- No segment satisfies the allowed-class / observed-fraction rule → Step 6 reports "no usable stretch" rather than returning an inadequate one.

---

## 8. Acceptance Criteria (definition of done)
1. Series is regular, unique, monotonic hourly in `WORK_TZ` (asserted).
2. Gap inventory validates against the §4 schema; class counts reconcile with total missing samples; per-timestamp `gap_class` present.
3. `is_observed`, `quality_flag`, `gap_id` exist on the full series and are consistent with the inventory.
4. `onset_after_extreme` present per gap.
5. Stretch table ranked with `observed_frac` and `months_covered`; the designated stretch contains no macro/severe gap and retains its internal micro/meso gaps unfilled.
6. The series-and-gaps figure renders as specified (3 panels, class-colored windows, stretch marked, legend) and the gap histogram is produced.
7. All six artifacts saved with correct names; save gated on success.
8. No target value is altered anywhere in the notebook.

---

## 9. Design Decisions & Rationale
**Stretch defined by classification, not by gap-freeness.** Per your instruction, a usable stretch may contain micro and meso gaps and is bounded only by macro/severe gaps. This maximises usable span while keeping every internal gap in a class NB03 can fill reliably. The stretch is saved unfilled; NB03 makes it continuous, NB04 consumes it.

**Rank by length, but guard observed content.** "Largest stretch I can fit" is span-based, but a long window dominated by meso gaps is mostly imputed filler — a weak substrate for a frozen baseline whose job is to learn the *real* healthy response. `MIN_OBSERVED_FRAC` and the reported `observed_frac` let you reject or override such windows. Recommendation: set a floor (e.g. ≥ 0.7 observed) rather than blindly taking the maximum span.

**No bridging anywhere.** Earlier the spec allowed micro-bridging inside the stretch; you removed it. NB01 is now strictly non-mutating on the target, which keeps a clean separation: classification here, all filling in NB03. The only cost is that the saved stretch is not yet continuous — explicitly handed to NB03.

**Visualization design.** Three incompatible scales (inclination, °C, %) → stacked panels sharing the x-axis, not one twin-axis chart. Scatter markers make gaps self-evident (absent points), reinforced by class-colored bands aligned across panels so you can see whether temperature/RH drop out with inclination. The selected stretch is overlaid so the figure doubles as the stretch-selection record.

**Classification windows are inclination-driven.** The shaded bands represent gaps in `TARGET_COL`. Aux channels may have their own gaps, shown as their own missing points; this is stated in the caption to avoid implying the bands classify temp/RH.

---

## 10. Open Decisions for You
- **`MIN_OBSERVED_FRAC` value** (or leave off): the floor that separates "usable" from "too holey." Sets whether selection is pure span or span-with-content.
- **Baseline season coverage**: whether the designated stretch must span ≥ a full annual cycle for NB04, which may override `'longest'`.
- **`onset_after_extreme` criterion**: the extreme threshold (percentile or k·σ of the pre-gap window) used to flag potentially informative gaps.