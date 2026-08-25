# Data Exploration · The Monitoring System and Its Record — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `studies/data_exploration/` as the project's documentation of the Gubbio monitoring system — what the instrumentation is, how the record is written, how the inclinometer is compensated, how much data exists at each station, and what station 02's four core channels look like.

**Architecture:** One `py:percent` notebook that imports `ud_lib` from the sibling `unified_dataset` study and adds no library of its own. It reads only `data/interim/unified/*.csv`, writes four CSV tables, four LaTeX table bodies and five figures into `outputs/`, and a hand-written LaTeX report that `\input`s the generated table bodies and `\includegraphics` the figures.

**Tech Stack:** Python 3 in the `neuralprophet_env` Conda environment; pandas, numpy, matplotlib, seaborn (through `ud.set_context`); jupytext `py:percent` pairing; pdflatex with the shared preamble `studies/_shared/reportstyle.tex`.

## Global Constraints

- **Design document:** `docs/superpowers/specs/2026-08-15-data-exploration-diagnostics-design.md`. Read it before Task 1.
- **Never edit `.ipynb` files.** All work happens in `data_exploration_study.py`. The user runs the jupytext sync. (`neuralprophet-shm/CLAUDE.md`, Pairing Rule.)
- **Never read the raw `.adc` archive.** This study consumes `data/interim/unified/` only.
- **Never hardcode a station identifier** in a filename or a computation. Use `STATION = ud.TARGET_STATION`. (`CLAUDE.md`, Output Artifact Naming Convention.)
- **Every inclination series shown is compensated.** The raw `inc` channel is used for availability counting only, never plotted as a signal.
- **Commits:** the project's standing rule is that commits happen only when the user asks (`CLAUDE.md`, Operating Protocol). Each task below ends with a commit step. Ask the user once, before the first commit; if they decline, skip every commit step and report the working-tree state at the end instead.
- **Caveman ultra mode in chat; full prose in code, comments, and every word of the report.** (Parent `CLAUDE.md`.)
- **Environment:** every command runs with `conda activate neuralprophet_env` from `studies/data_exploration/` unless stated otherwise.
- **Compensation provenance:** the report states `k = 0.005` (5 mdeg per °C) as the **manufacturer's specified** correction for the temperature-induced measurement bias. This is a user decision recorded in the design document and it overrides the "placeholder" wording currently in `docs/raw-data-format.md` §7.4 item 3, which Task 7 corrects.

---

## File Structure

| Path | Responsibility |
|---|---|
| `studies/data_exploration/data_exploration_study.py` | The whole study: read, compute, write tables, table bodies and figures. Created new, replacing the existing file. |
| `studies/data_exploration/report/data_exploration_report.tex` | The report. §1–§4 are hand-written documentation; §5–§6 hold the figures and `\input` the generated table bodies. Replaces the existing file. |
| `studies/data_exploration/README.md` | How to run and rebuild. Replaces the existing file. |
| `studies/data_exploration/outputs/` | Four CSV tables, four `.tex` table bodies, five figure pairs (PNG + SVG). Gitignored. |
| `docs/raw-data-format.md` | One item corrected, §7.4 item 3. |

**On the generated table bodies.** The sibling reports transcribe their numbers into the `.tex` by hand. This study writes the four data-dependent table bodies to `outputs/DE_T0*.tex` and `\input`s them instead. The reason is that three of the four tables have a row count fixed by the data (nine years of coverage, two eras times four channels), so hand transcription would have to be redone every time the archive grows. The static documentation tables of §1–§4 stay hand-written, because they describe the file format and never change with the data.

---

### Task 1: Clear the ground and build the notebook skeleton

**Files:**
- Create: `studies/data_exploration/data_exploration_study.py` (overwrites the existing file)
- Delete: `studies/data_exploration/outputs/DE_01_missing_data.csv`, `DE_02_summary_statistics.csv`, `DE_F01_missingness.pdf`, `DE_F02_series.pdf`, `DE_F03_corr.pdf`, `DE_F04_dist.pdf`

**Interfaces:**
- Consumes: `ud_lib` from `../unified_dataset`; `data/interim/unified/unified_{station}_1h.csv`
- Produces: module-level names used by every later task — `frames` (dict station to DataFrame), `df` (the target station's frame), `STATION`, `STATIONS`, `CHANGEOVER`, `CHANNELS`, `CHANNEL_LABEL`, `OFFSET_WINDOW_DAYS`, `OUTPUT_DIR`

- [ ] **Step 1: Confirm the inputs exist**

Run:
```bash
ls -1 ../../data/interim/unified/
```
Expected: `unified_manifest.json`, `unified_st01_1h.csv`, `unified_st02_1h.csv`, `unified_st03_1h.csv`.

If they are absent, stop: `studies/unified_dataset/build_unified_dataset.py` has not been run, and nothing in this plan can proceed.

- [ ] **Step 2: Delete the stale outputs of the discarded study**

Run:
```bash
rm -f outputs/DE_01_missing_data.csv outputs/DE_02_summary_statistics.csv \
      outputs/DE_F01_missingness.pdf outputs/DE_F02_series.pdf \
      outputs/DE_F03_corr.pdf outputs/DE_F04_dist.pdf
ls -1 outputs/
```
Expected: the directory is empty, or holds only files the user added.

- [ ] **Step 3: Write the notebook header, title and configuration**

Write `data_exploration_study.py` with exactly this content:

```python
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Study · The monitoring system of the Mura Urbiche di Gubbio, and its record
#
# This study documents the static structural health monitoring installation on the
# medieval urban walls of Gubbio and the record it has produced since July 2018. It is
# the first document to read in this project: every other study assumes the
# instrumentation, the file format, the thermal compensation and the shape of the
# archive, and this one states them.
#
# It consumes the unified dataset built by `studies/unified_dataset/build_unified_dataset.py`
# and never opens a raw `.adc` file.
#
# 1. **Read the unified dataset** — all three stations, on the hourly grid.
# 2. **Coverage of the inclination record** — how much each station contributes, and
#    when the network narrowed from three instrumented locations to one.
# 3. **The instrument changeover** — the level step at the February 2025 replacement,
#    the seasonal part of it, and the gain diagnostic that qualifies the estimate.
# 4. **Station 02, variable by variable** — the compensated inclination, the air
#    temperature, the relative humidity and the supply voltage, one plot each.
#
# The narrative, the instrumentation description and the file-format specification are
# in `report/data_exploration_report.pdf`. This notebook produces the numbers and the
# figures that report reads.

# %% [markdown]
# ## Imports and configuration

# %%
# %load_ext autoreload
# %autoreload 2

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../unified_dataset'))

import ud_lib as ud

pd.set_option('display.width', 170)
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 80)

# %% [markdown]
# ### Parameters
#
# | Parameter | Purpose |
# |---|---|
# | `DATA_DIR` | Where `build_unified_dataset.py` wrote its output. |
# | `OUTPUT_DIR` | Where this study's tables, table bodies and figures are written. |
# | `STATION` | The station carried through steps 3 and 4. Station 02, at the Porta di Sant'Ubaldo, is the best covered and the only one spanning both instrument eras. |
# | `STATIONS` | Every station in the archive, for the coverage comparison of step 2. |
# | `CHANGEOVER` | The date the expanded instrument package replaced the legacy network. Marked on every figure. |
# | `CHANNELS` | The four channels plotted one by one in step 4. |
# | `OFFSET_WINDOW_DAYS` | Comparison window on each side of the changeover, used to estimate the instrument offset. Larger windows average more of the annual cycle into the estimate; smaller ones carry more measurement noise. |

# %%
DATA_DIR = '../../data/interim/unified'
OUTPUT_DIR = 'outputs'

STATION = ud.TARGET_STATION
STATIONS = ud.STATIONS
CHANGEOVER = pd.Timestamp(ud.CURRENT_START)

CHANNELS = ['inc_comp', 'tair', 'rh', 'batt']
CHANNEL_LABEL = {
    'inc_comp': 'Compensated inclination [mdeg]',
    'tair': 'Air temperature [\u00b0C]',
    'rh': 'Relative humidity [%]',
    'batt': 'Supply voltage [V]',
}
OFFSET_WINDOW_DAYS = 30

MARK_COLOUR = '#A5202B'      # the changeover line, matching the report palette
LEGACY_COLOUR = '#1F4E79'
CURRENT_COLOUR = '#1B6B3A'

os.makedirs(OUTPUT_DIR, exist_ok=True)
ud.set_context('notebook')

# %% [markdown]
# ## Step 1 · Read the unified dataset
#
# One frame per station on the hourly grid, spanning the whole archive. The ingest
# fills nothing: every value present here is a measurement.

# %%
frames = {s: ud.load_unified(DATA_DIR, s) for s in STATIONS}
df = frames[STATION]

print(f'{STATION}: {len(df)} hourly slots, '
      f'{df.index.min()} to {df.index.max()}')
print(f'columns: {list(df.columns)}')
for col in ('inc_comp', 'inc_comp_joined'):
    print(f'  {col:16s} observed {df[col].notna().sum():6d} '
          f'({100 * df[col].notna().mean():.1f} %)')
```

- [ ] **Step 4: Run the skeleton and confirm it reads the archive**

Run:
```bash
conda activate neuralprophet_env
jupytext --to ipynb data_exploration_study.py
jupyter nbconvert --to notebook --execute --inplace data_exploration_study.ipynb
```
Expected: completes without error. The printed extent runs from `2018-07-26 00:00:00` to a timestamp on `2026-08-13`, roughly 70,600 hourly slots, and `inc_comp` observed at close to 70 %.

If `ModuleNotFoundError: ud_lib`, the `sys.path` insert for `../unified_dataset` is wrong or the sibling study has been moved.

- [ ] **Step 5: Commit**

```bash
cd ../..
git add studies/data_exploration/data_exploration_study.py
git commit -m "feat(data_exploration): rebuild the study skeleton on the unified dataset"
```

---

### Task 2: Coverage of the inclination record

**Files:**
- Modify: `studies/data_exploration/data_exploration_study.py` (append)

**Interfaces:**
- Consumes: `frames`, `STATIONS`, `CHANGEOVER`, `OUTPUT_DIR`, `MARK_COLOUR` from Task 1
- Produces: `classification` (DataFrame indexed by station), `yearly` (DataFrame, years by station); files `DE_01_station_classification.csv`, `DE_02_yearly_coverage.csv`, `DE_T01_station_classification.tex`, `DE_T02_yearly_coverage.tex`, `DE_F01_inclination_coverage.{png,svg}`

- [ ] **Step 1: Append the per-station classification**

Append to `data_exploration_study.py`:

```python
# %% [markdown]
# ## Step 2 · Coverage of the inclination record
#
# ### What each station contributes
#
# The archive spans eight years, and no station covers all of it. Two of the three
# legacy stations stopped recording long before the February 2025 replacement, so the
# network narrowed in stages rather than at a single moment.

# %%
classification = ud.classify_stations(frames, freq=ud.ANALYSIS_FREQ)
display(classification[['eras', 'first_reading', 'last_reading', 'span_days',
                        'observed_days', 'coverage_of_span_%',
                        'coverage_of_archive_%']])
classification.to_csv(f'{OUTPUT_DIR}/DE_01_station_classification.csv')
```

- [ ] **Step 2: Append the yearly coverage**

`ud_lib` provides monthly completeness only; the yearly view is a three-line aggregation and stays here.

```python
# %% [markdown]
# ### Year by year
#
# The same fact as numbers rather than as a picture: the fraction of each calendar
# year in which the inclinometer reported at all.

# %%
yearly = pd.DataFrame({
    s: frames[s]['inc'].notna().resample('1YS').mean() for s in STATIONS
})
yearly.index = yearly.index.year
yearly.index.name = 'year'
display((100 * yearly).round(1))
yearly.round(4).to_csv(f'{OUTPUT_DIR}/DE_02_yearly_coverage.csv')
```

- [ ] **Step 3: Append the coverage figure with the changeover marked**

The shared plotter is called without a save path, the changeover line is added to every
panel, and the figure is saved here — so the mark is added without reimplementing the
plot.

```python
# %% [markdown]
# ### The record, drawn
#
# One strip per station, the fraction of each day observed. The dashed line is the
# 21 February 2025 instrument changeover.

# %%
fig = ud.plot_station_completeness(
    frames, column='inc',
    title='Daily availability of the inclinometer, by station')
for ax in fig.axes:
    ax.axvline(CHANGEOVER, color=MARK_COLOUR, lw=1.0, ls='--')
ud._finish(fig, OUTPUT_DIR, 'DE_F01_inclination_coverage')
plt.show()
```

- [ ] **Step 4: Append the two LaTeX table bodies**

```python
# %% [markdown]
# ### Table bodies for the report
#
# The report holds the table headers and `\input`s these bodies, so that a rebuilt
# archive never needs a number transcribed by hand.

# %%
with open(f'{OUTPUT_DIR}/DE_T01_station_classification.tex', 'w') as fh:
    for station, row in classification.iterrows():
        fh.write(f'\\texttt{{{station}}} & {row["eras"]} & '
                 f'{row["first_reading"][:10]} & {row["last_reading"][:10]} & '
                 f'{row["span_days"]:.1f} & {row["observed_days"]:.1f} & '
                 f'{row["coverage_of_span_%"]:.1f} \\\\\n')

with open(f'{OUTPUT_DIR}/DE_T02_yearly_coverage.tex', 'w') as fh:
    for year, row in yearly.iterrows():
        cells = ' & '.join(
            '---' if pd.isna(row[s]) else f'{100 * row[s]:.1f}'
            for s in STATIONS)
        fh.write(f'{year} & {cells} \\\\\n')
```

- [ ] **Step 5: Run and verify against the sibling study**

Run:
```bash
conda activate neuralprophet_env
jupytext --to ipynb data_exploration_study.py
jupyter nbconvert --to notebook --execute --inplace data_exploration_study.ipynb
python - <<'PY'
import pandas as pd
c = pd.read_csv('outputs/DE_01_station_classification.csv', index_col=0)
print(c[['observed_days', 'coverage_of_archive_%']])
assert abs(c.loc['st02', 'observed_days'] - 2061.0) < 0.6, c.loc['st02']
print('st02 observed days agree with the unified study')
PY
ls -1 outputs/
```
Expected: `st01` about 1118.7 observed days, `st02` 2061.0, `st03` 984.1; the assertion passes; `outputs/` holds `DE_01`, `DE_02`, `DE_T01`, `DE_T02` and the `DE_F01` PNG and SVG.

A disagreement on `st02` means this study is reading a different unified build from the one the sibling report was written against. Stop and report it rather than adjusting the assertion.

- [ ] **Step 6: Commit**

```bash
cd ../..
git add studies/data_exploration/data_exploration_study.py
git commit -m "feat(data_exploration): per-station and per-year inclinometer coverage"
```

---

### Task 3: The instrument changeover

**Files:**
- Modify: `studies/data_exploration/data_exploration_study.py` (append)

**Interfaces:**
- Consumes: `df`, `OFFSET_WINDOW_DAYS`, `OUTPUT_DIR` from Task 1
- Produces: `offset_info` (dict), used by Task 4's inclination figure. Its keys, from `ud.estimate_era_offset`: `left_end`, `right_start`, `interruption_days`, `n_before`, `n_after`, `mean_before`, `mean_after`, `raw_step`, `seasonal_step`, `deseasoned_step`, `annual_amplitude`, `diurnal_amp_before`, `diurnal_amp_after`, `gain_ratio`, `offset`. Files `DE_04_era_offset.csv`, `DE_T04_era_offset.tex`

- [ ] **Step 1: Append the offset estimate**

```python
# %% [markdown]
# ## Step 3 · The instrument changeover
#
# The package installed on 21 February 2025 is a physical reinstallation. Its
# inclinometer shares no baseline with the legacy unit it replaced, so the two eras
# sit at unrelated levels and a step appears at the boundary.
#
# The step is estimated by comparing the mean of the last `OFFSET_WINDOW_DAYS` of
# legacy readings with the mean of the first `OFFSET_WINDOW_DAYS` of current-era ones.
# Taken naively that difference confounds the instrument with the season, because the
# two windows sit at different points of the annual cycle; a harmonic model fitted to
# the legacy era is therefore subtracted first, and what survives is attributed to the
# instrument.
#
# The diagnostic reported beside the offset is the ratio of diurnal amplitudes either
# side of the boundary. The daily thermal cycle is a property of the wall, not of the
# logger, so a ratio far from one says the instruments differ in gain as well as in
# zero — and a single additive constant cannot absorb a multiplicative difference.

# %%
offset_info = ud.estimate_era_offset(
    df, window_days=OFFSET_WINDOW_DAYS, deseason=True, target='inc_comp')

offset_table = pd.Series(offset_info).to_frame('value')
display(offset_table)
offset_table.to_csv(f'{OUTPUT_DIR}/DE_04_era_offset.csv')

print(f'Interruption                   : '
      f'{offset_info["interruption_days"]:.2f} d')
print(f'Raw step across the changeover : '
      f'{offset_info["raw_step"]:+.2f} mdeg')
print(f'Attributable to the season     : '
      f'{offset_info["seasonal_step"]:+.2f} mdeg')
print(f'Attributed to the instrument   : '
      f'{offset_info["offset"]:+.2f} mdeg')
print(f'Diurnal amplitude before/after : '
      f'{offset_info["diurnal_amp_before"]:.2f} / '
      f'{offset_info["diurnal_amp_after"]:.2f} mdeg '
      f'(ratio {offset_info["gain_ratio"]:.2f})')
```

- [ ] **Step 2: Append the table body**

```python
# %%
_OFFSET_ROWS = [
    ('Interruption', f'{offset_info["interruption_days"]:.2f} d'),
    ('Legacy window mean', f'{offset_info["mean_before"]:+.2f} mdeg'),
    ('Current window mean', f'{offset_info["mean_after"]:+.2f} mdeg'),
    ('Raw step', f'{offset_info["raw_step"]:+.2f} mdeg'),
    ('Attributable to the season', f'{offset_info["seasonal_step"]:+.2f} mdeg'),
    ('\\textbf{Attributed to the instrument}',
     f'$\\mathbf{{{offset_info["offset"]:+.2f}}}$ \\textbf{{mdeg}}'),
    ('Diurnal amplitude, legacy side',
     f'{offset_info["diurnal_amp_before"]:.2f} mdeg'),
    ('Diurnal amplitude, current side',
     f'{offset_info["diurnal_amp_after"]:.2f} mdeg'),
    ('\\textbf{Gain ratio}',
     f'$\\mathbf{{{offset_info["gain_ratio"]:.2f}}}$'),
]

with open(f'{OUTPUT_DIR}/DE_T04_era_offset.tex', 'w') as fh:
    for label, value in _OFFSET_ROWS:
        fh.write(f'{label} & {value} \\\\\n')
```

- [ ] **Step 3: Run and verify the offset against the sibling study**

Run:
```bash
conda activate neuralprophet_env
jupytext --to ipynb data_exploration_study.py
jupyter nbconvert --to notebook --execute --inplace data_exploration_study.ipynb
python - <<'PY'
import pandas as pd
mine = pd.read_csv('outputs/DE_04_era_offset.csv', index_col=0)['value']
theirs = pd.read_csv('../unified_dataset/outputs/U_03_era_offset.csv',
                     index_col=0)['value']
for key in ('raw_step', 'offset', 'gain_ratio'):
    a, b = float(mine[key]), float(theirs[key])
    print(f'{key:12s} this study {a:>10.3f}   ingest {b:>10.3f}')
    assert abs(a - b) < 1e-6, key
print('the recomputed offset matches the ingest exactly')
PY
```
Expected: `raw_step` −127.667, `offset` −132.476, `gain_ratio` 1.338, all three assertions passing.

If `U_03_era_offset.csv` is absent, the sibling study's outputs have not been produced; note it and check the three numbers against the values quoted here instead.

- [ ] **Step 4: Commit**

```bash
cd ../..
git add studies/data_exploration/data_exploration_study.py
git commit -m "feat(data_exploration): estimate the instrument offset at the changeover"
```

---

### Task 4: Station 02, variable by variable

**Files:**
- Modify: `studies/data_exploration/data_exploration_study.py` (append)

**Interfaces:**
- Consumes: `df`, `CHANNELS`, `CHANNEL_LABEL`, `CHANGEOVER`, `STATION`, `OFFSET_WINDOW_DAYS`, `offset_info`, the three colour constants
- Produces: `summary` (DataFrame indexed by era and channel); files `DE_03_summary_by_era_{station}.csv`, `DE_T03_summary_by_era.tex`, `DE_F02_{station}_inclination.{png,svg}`, `DE_F03_{station}_tair.{png,svg}`, `DE_F04_{station}_rh.{png,svg}`, `DE_F05_{station}_batt.{png,svg}`

- [ ] **Step 1: Append the per-era summary statistics**

The statistics are split by era deliberately. Pooling them would average two unrelated
inclinometer baselines into a single meaningless mean, which is the defect of the study
this one replaces.

```python
# %% [markdown]
# ## Step 4 · Station 02, variable by variable
#
# Station 02 stands at the Porta di Sant'Ubaldo. It is the best-covered record in the
# archive and the only one spanning both instrument eras, and it is the station every
# later study in this project works on.
#
# ### What the four channels look like, era by era
#
# The two eras are summarised separately throughout. Pooling them would average two
# unrelated inclinometer baselines into one number, and the standard deviation of that
# pooled series would be dominated by the step between the instruments rather than by
# anything the wall did.

# %%
rows = []
for era, sub in df.groupby('era', sort=False):
    for col in CHANNELS:
        s = sub[col].dropna()
        rows.append({
            'era': era,
            'channel': col,
            'count': int(s.size),
            'mean': round(float(s.mean()), 3) if s.size else np.nan,
            'sd': round(float(s.std()), 3) if s.size > 1 else np.nan,
            'min': round(float(s.min()), 3) if s.size else np.nan,
            'median': round(float(s.median()), 3) if s.size else np.nan,
            'max': round(float(s.max()), 3) if s.size else np.nan,
        })
summary = pd.DataFrame(rows).set_index(['era', 'channel'])
display(summary)
summary.to_csv(f'{OUTPUT_DIR}/DE_03_summary_by_era_{STATION}.csv')

with open(f'{OUTPUT_DIR}/DE_T03_summary_by_era.tex', 'w') as fh:
    for (era, channel), row in summary.iterrows():
        # An escaped underscore cannot be built inside an f-string expression
        # on Python below 3.12, so the LaTeX-safe name is made first.
        safe = channel.replace('_', r'\_')
        fh.write(f'{era} & \\texttt{{{safe}}} & '
                 f'{row["count"]:,.0f} & {row["mean"]:.2f} & {row["sd"]:.2f} & '
                 f'{row["min"]:.2f} & {row["median"]:.2f} & '
                 f'{row["max"]:.2f} \\\\\n')
```

- [ ] **Step 2: Append the inclination figure, three panels**

Panel (a) is the whole archive, panel (b) is the changeover at close range with the two
comparison-window means drawn as the levels the offset differences, panel (c) is the
joined series.

```python
# %% [markdown]
# ### Inclination
#
# The compensated series, which is the measurement this project works on. Its absolute
# level carries no structural information — it is set by how the instrument sat in its
# mount at installation — so the level is referred to an arbitrary anchor within each
# era and only its changes are interpreted.
#
# The middle panel is the offset procedure of step 3, drawn: the two horizontal levels
# are the window means either side of the changeover, and their difference is the raw
# step the estimate starts from. The lower panel is the joined series, with the
# estimated offset removed.

# %%
left_end = pd.Timestamp(offset_info['left_end'])
right_start = pd.Timestamp(offset_info['right_start'])
window = pd.Timedelta(days=OFFSET_WINDOW_DAYS)

fig, axes = plt.subplots(3, 1, figsize=ud.figsize(11, 8.4))

ax = axes[0]
ax.plot(df.index, df['inc_comp'], lw=0.4, color='0.35')
ax.axvline(CHANGEOVER, color=MARK_COLOUR, lw=1.0, ls='--')
ax.set_ylabel('inc_comp [mdeg]')
ax.set_title('(a) the whole archive, anchored within each era')

ax = axes[1]
zoom = df.loc[left_end - 3 * window:right_start + 3 * window, 'inc_comp']
ax.plot(zoom.index, zoom.values, lw=0.7, color='0.35')
ax.axvline(CHANGEOVER, color=MARK_COLOUR, lw=1.0, ls='--')
ax.hlines(offset_info['mean_before'], left_end - window, left_end,
          color=LEGACY_COLOUR, lw=2.2,
          label=f'legacy window mean {offset_info["mean_before"]:+.1f} mdeg')
ax.hlines(offset_info['mean_after'], right_start, right_start + window,
          color=CURRENT_COLOUR, lw=2.2,
          label=f'current window mean {offset_info["mean_after"]:+.1f} mdeg')
ax.legend(fontsize='small', loc='best')
ax.set_ylabel('inc_comp [mdeg]')
ax.set_title(f'(b) the changeover at close range: a raw step of '
             f'{offset_info["raw_step"]:+.1f} mdeg, of which '
             f'{offset_info["seasonal_step"]:+.1f} is the season')

ax = axes[2]
ax.plot(df.index, df['inc_comp_joined'], lw=0.4, color=CURRENT_COLOUR)
ax.axvline(CHANGEOVER, color=MARK_COLOUR, lw=1.0, ls='--')
ax.set_ylabel('inc_comp_joined [mdeg]')
ax.set_title(f'(c) the same record with the estimated offset of '
             f'{offset_info["offset"]:+.1f} mdeg removed')

fig.suptitle(f'{STATION}: compensated inclination')
ud._finish(fig, OUTPUT_DIR, f'DE_F02_{STATION}_inclination')
plt.show()
```

- [ ] **Step 3: Append the three remaining channel plots**

```python
# %% [markdown]
# ### Air temperature, relative humidity and supply voltage
#
# One plot each, on the same axis span as the inclination above. The supply voltage is
# not a structural quantity: it is the acquisition system's own health, and it is shown
# because it is the channel that explains the outages.

# %%
for fid, col in (('DE_F03', 'tair'), ('DE_F04', 'rh'), ('DE_F05', 'batt')):
    fig, ax = plt.subplots(figsize=ud.figsize(11, 3.0))
    ax.plot(df.index, df[col], lw=0.4, color='0.35')
    ax.axvline(CHANGEOVER, color=MARK_COLOUR, lw=1.0, ls='--')
    ax.set_ylabel(CHANNEL_LABEL[col])
    ax.set_title(f'{STATION}: {CHANNEL_LABEL[col].split(" [")[0].lower()}')
    ud._finish(fig, OUTPUT_DIR, f'{fid}_{STATION}_{col}')
    plt.show()
```

- [ ] **Step 4: Run the whole notebook and check every artefact**

Run:
```bash
conda activate neuralprophet_env
jupytext --to ipynb data_exploration_study.py
jupyter nbconvert --to notebook --execute --inplace data_exploration_study.ipynb
python - <<'PY'
import os
expected = [
    'DE_01_station_classification.csv', 'DE_02_yearly_coverage.csv',
    'DE_03_summary_by_era_st02.csv', 'DE_04_era_offset.csv',
    'DE_T01_station_classification.tex', 'DE_T02_yearly_coverage.tex',
    'DE_T03_summary_by_era.tex', 'DE_T04_era_offset.tex',
]
for stem in ('DE_F01_inclination_coverage', 'DE_F02_st02_inclination',
             'DE_F03_st02_tair', 'DE_F04_st02_rh', 'DE_F05_st02_batt'):
    expected += [stem + '.png', stem + '.svg']
missing = [f for f in expected if not os.path.exists(os.path.join('outputs', f))]
print('missing:', missing or 'none')
assert not missing
PY
```
Expected: `missing: none`.

- [ ] **Step 5: Look at the five figures**

Open `outputs/DE_F02_st02_inclination.png` and confirm the middle panel shows two
horizontal levels either side of the dashed line, visibly offset from one another, and
that the lower panel has no step at the dashed line. Open the other four and confirm
each has data across the full width and a visible changeover line.

If panel (b) shows the levels but no visible series, the zoom window is empty — check
that `left_end` and `right_start` came from `offset_info` and not from `CHANGEOVER`.

- [ ] **Step 6: Commit**

```bash
cd ../..
git add studies/data_exploration/data_exploration_study.py
git commit -m "feat(data_exploration): per-era statistics and the four station 02 channels"
```

---

### Task 5: The report, sections 1 to 4 — the documentation

**Files:**
- Create: `studies/data_exploration/report/data_exploration_report.tex` (overwrites the existing file)

**Interfaces:**
- Consumes: `docs/raw-data-format.md` §§1–4 and §7 for every fact; `studies/_shared/reportstyle.tex` for `\panel`, `\degC`, the `L`/`R`/`Y` column types and the accent colours
- Produces: a `.tex` that compiles on its own, with sections 5 and 6 added by Task 6

Everything in this task is prose and static tables. No number in it comes from the data;
all of it comes from `docs/raw-data-format.md`. Check each table against that document
field by field as you write it.

- [ ] **Step 1: Write the preamble, title and abstract**

Write `report/data_exploration_report.tex`:

```latex
% ---------------------------------------------------------------------------
% Study report · The monitoring system of the Mura Urbiche di Gubbio
%
% Build:  pdflatex data_exploration_report.tex  (twice, for the ToC)
% Figures and table bodies are read from ../outputs/ and are produced by the
% paired notebook. The site map DE_F00_site_map is supplied by hand.
% ---------------------------------------------------------------------------
\documentclass[11pt,a4paper]{article}

\newcommand{\runninghead}{The Gubbio monitoring system and its record}

\input{../../_shared/reportstyle.tex}
\graphicspath{{../outputs/}}

\title{\vspace{-1.2cm}\textbf{The monitoring system of the Mura Urbiche di
Gubbio, and the record it has produced}\\[2mm]
\large Instrumentation, acquisition format, thermal compensation, and eight
years of measurement}
\author{Study report · \texttt{studies/data\_exploration/}}
\date{15 August 2026}

\begin{document}
\maketitle
\thispagestyle{fancy}

\begin{abstract}
\noindent
This report documents the static structural health monitoring installation on
the medieval urban walls of Gubbio --- the \emph{Mura Urbiche} --- and the
record it has produced since July 2018. It states what the instrumentation
measures, how the acquisition system writes its files, how the inclinometer is
compensated for temperature, how much data each station has actually
contributed, and what the four core channels of the principal station look
like across the whole archive.

It is the first document to read in this project. Every other study assumes the
instrumentation, the file format, the compensation and the shape of the
archive; this one states them, and every number in it is recomputed from the
unified dataset rather than quoted from elsewhere.
\end{abstract}

\tableofcontents
```

- [ ] **Step 2: Write section 1, the monitoring system**

Append:

```latex
% ===========================================================================
\section{The monitoring system}
\label{sec:system}

The \emph{Mura Urbiche} of Gubbio are the medieval urban walls of the town, in
Umbria. The monitored stretch stands in an embankment situation, with a
sun-exposed valley face and a shaded mountain face, and the installation exists
to measure how the wall responds --- slowly, and mostly to temperature --- over
years rather than over events.

The system is a static one. It does not measure vibration or dynamic response;
it measures the inclination of the wall, sampled every twenty minutes, together
with the environmental variables that drive it and the supply voltage of the
unit that records it.

Three measurement stations were installed. \textbf{Station 02 stands at the
Porta di Sant'Ubaldo}, a principal gate of the medieval circuit, and is the
station carried through Sections~\ref{sec:coverage} and~\ref{sec:station}: it
is the best-covered record in the archive and the only one spanning both
instrument generations.

\begin{center}
\small
\begin{tabularx}{\textwidth}{L{1.6cm}L{1.6cm}L{2.8cm}Y}
\toprule
\textbf{Station} & \textbf{Block} & \textbf{Raw columns} & \textbf{Note} \\
\midrule
\texttt{st01} & b1 & 3--6 (legacy) & Legacy unit only \\
\texttt{st02} & b2 & 7--10 (legacy), 15--20 (current) &
\textbf{Porta di Sant'Ubaldo.} The only station instrumented in both eras; the
2025 package is a physical reinstallation, not an addition \\
\texttt{st03} & b3 & 11--14 (legacy) & Legacy unit only \\
\bottomrule
\end{tabularx}
\end{center}

\noindent{\footnotesize The block-to-station mapping comes from the
installation record and is not inferable from the files themselves
(\texttt{docs/raw-data-format.md} Section 3.3).}

% --- Site map ---------------------------------------------------------------
% The map of the three station locations is supplied by hand, not generated by
% the notebook. Drop the file into studies/data_exploration/outputs/ as
% DE_F00_site_map.png (or .pdf) and uncomment the block below.
%
% \begin{figure}[H]
% \includegraphics{DE_F00_site_map}
% \caption{The three measurement stations on the Mura Urbiche. Station 02 sits
% at the Porta di Sant'Ubaldo.}
% \end{figure}
```

- [ ] **Step 3: Write section 2, the instrumentation**

Append:

```latex
% ===========================================================================
\section{The instrumentation}
\label{sec:instruments}

Each legacy acquisition unit reports four channels: its own supply voltage, air
temperature and relative humidity at the unit, and the inclinometer reading.
The three units are independent and report into a single file.

On \textbf{21 February 2025} an expanded package replaced the legacy network at
station 02. It reports the same four channels and adds two: incident solar
radiation and wall surface temperature. Its inclinometer is a physically
reinstalled instrument and \textbf{shares no baseline with the unit it
replaced} --- a point Section~\ref{sec:station} returns to, because it governs
how the two eras may be joined.

\begin{center}
\small
\begin{tabularx}{\textwidth}{L{1.9cm}L{3.3cm}L{1.5cm}L{2.1cm}Y}
\toprule
\textbf{Symbol} & \textbf{Quantity} & \textbf{Unit} & \textbf{Era} &
\textbf{Sentinel} \\
\midrule
\texttt{Batt}   & Supply voltage of the unit & V &
both & \texttt{0.000} \\
\texttt{Tair}   & Air temperature at the unit & \degC &
both & \texttt{0.000} \\
\texttt{RH}     & Relative humidity & \% &
both & \texttt{0.000} \\
\texttt{I}      & Inclinometer reading & mdeg &
both & \texttt{0.000} \\
\texttt{n\_SR}  & Incident solar radiation & W/m$^2$ &
current only & \texttt{8191.875} \\
\texttt{n\_Twall} & Wall surface temperature & \degC &
current only & \texttt{-55.0} \\
\bottomrule
\end{tabularx}
\end{center}

\noindent{\footnotesize Observed ranges are deliberately absent here: they
differ between the two eras and are reported per era in
Section~\ref{sec:station}. Sentinel handling is specified in
Section~\ref{sec:format}.}

\panel{What the inclinometer reading is, and what it is not}{%
\texttt{I} is reported in millidegrees, but it is \textbf{not} an inclination
referred to any physical datum. It is a raw transducer reading whose absolute
level is fixed by how the instrument happened to sit in its mount at
installation, plus whatever offset the electronics carry.

\medskip
\textbf{The absolute level of the inclinometer series carries no structural
information. Only its changes do.} A statement of the form ``station 02 is at
2160\,mdeg'' is meaningless; the first difference, which is invariant to every
arbitrary constant in the chain, is the quantity with a physical meaning
independent of how the record was assembled.

\medskip
Every level in this report is therefore referred to an arbitrary anchor, and is
shown to make the record's \emph{shape} visible rather than to be read off the
axis.}
```

- [ ] **Step 4: Write section 3, the acquisition format**

Append:

```latex
% ===========================================================================
\section{How the record is written}
\label{sec:format}

The acquisition system writes one plain-text file per calendar day into a flat
directory, named \texttt{GUBBIO\_YYYYMMDD.adc}. The archive is the acquisition
system's output and is read-only to this project; it is never committed to the
repository.

\begin{center}
\small
\begin{tabularx}{\textwidth}{L{5.0cm}Y}
\toprule
\textbf{Property} & \textbf{Value} \\
\midrule
Encoding & Plain text, one record per line \\
Field separator & Tab \\
Header row & None \\
Field 1 & Date, \texttt{DD/MM/YY} \\
Field 2 & Time, \texttt{HH:MM:SS} \\
Fields 3 onwards & Numeric measurements \\
Nominal sampling interval & 20 minutes, 72 records on a complete day \\
Decimal separator & \textbf{Both \texttt{.} and \texttt{,} occur} \\
\bottomrule
\end{tabularx}
\end{center}

\subsection{Two column layouts, one changeover}

The measurement columns are fixed-width blocks, one block per acquisition unit,
each legacy unit contributing four channels in the order \texttt{Batt},
\texttt{Tair}, \texttt{RH}, \texttt{I}. The number of columns changed exactly
once, on 21 February 2025.

\begin{center}
\small
\begin{tabular}{llll}
\toprule
\multicolumn{4}{l}{\textbf{Legacy era, 2018-07-26 to 2025-02-20 --- 14
columns}} \\
\midrule
1 & \texttt{date} & 8  & \texttt{b2\_Tair} [\degC] \\
2 & \texttt{time} & 9  & \texttt{b2\_RH} [\%] \\
3 & \texttt{b1\_Batt} [V] & 10 & \texttt{b2\_I} [mdeg] \\
4 & \texttt{b1\_Tair} [\degC] & 11 & \texttt{b3\_Batt} [V] \\
5 & \texttt{b1\_RH} [\%] & 12 & \texttt{b3\_Tair} [\degC] \\
6 & \texttt{b1\_I} [mdeg] & 13 & \texttt{b3\_RH} [\%] \\
7 & \texttt{b2\_Batt} [V] & 14 & \texttt{b3\_I} [mdeg] \\
\bottomrule
\end{tabular}
\end{center}

\begin{center}
\small
\begin{tabular}{ll}
\toprule
\multicolumn{2}{l}{\textbf{Current era, 2025-02-21 onwards --- 20 columns}} \\
\midrule
1--2   & \texttt{date}, \texttt{time} \\
3--14  & legacy blocks b1, b2, b3 --- \textbf{identically zero in this era} \\
15     & \texttt{n\_Batt} [V] \\
16     & \texttt{n\_Tair} [\degC] \\
17     & \texttt{n\_RH} [\%] \\
18     & \texttt{n\_I} [mdeg] \\
19     & \texttt{n\_SR} [W/m$^2$] \\
20     & \texttt{n\_Twall} [\degC] \\
\bottomrule
\end{tabular}
\end{center}

The legacy blocks are retained in their original positions and written as
constant zero on every record: no legacy station reports data after the
changeover. A loader must therefore branch on the record's field count rather
than on its date, so that a file whose content disagrees with the expected era
is handled rather than mangled.

\subsection{Sentinels and conventions}

\begin{center}
\small
\begin{tabularx}{\textwidth}{L{2.4cm}L{3.4cm}Y}
\toprule
\textbf{Value} & \textbf{Channels} & \textbf{Meaning and required action} \\
\midrule
\texttt{0.000} & \texttt{Batt}, \texttt{Tair}, \texttt{RH}, \texttt{I} &
Channel unavailable. A battery at 0\,V and a humidity of 0\,\% are physically
implausible here, and the zeros appear in whole-block runs coinciding with
known outages. \textbf{Must become \texttt{NaN} at read time}; treated as
measurements they inject flatlines, and the return to normal values then reads
as a step change \\
\texttt{0.000} & \texttt{n\_SR} &
\textbf{A real reading.} Zero is legitimate at night and must be preserved \\
\texttt{-55.0} & \texttt{n\_Twall} &
Probe-failure sentinel, the bottom of the sensor range emitted as an
open-circuit indicator. Must become \texttt{NaN} \\
\texttt{8191.875} & \texttt{n\_SR} &
Unsigned wrap-around of a near-zero reading, $2^{13}-0.125$. Any value above
roughly 1400\,W/m$^2$ is unphysical at this latitude and must be rejected \\
\midrule
\multicolumn{3}{l}{\textbf{Decimal separator}} \\
\multicolumn{3}{Y}{Both \texttt{3.500} and \texttt{3,500} occur, and the two
coexist \emph{inside a single file} in 2019. The separator must be normalised
per field --- detecting it once from the first lines and applying that decision
to the whole file silently corrupts values} \\
\bottomrule
\end{tabularx}
\end{center}

\subsection{From the raw grid to the analysis grid}

Every study in this project works on an hourly grid rather than on the raw
20-minute records. The aggregation is performed once, by
\texttt{studies/unified\_dataset/build\_unified\_dataset.py}, which is the only
code in the project that reads the archive. Slots with no data are carried as
\texttt{NaN} rather than dropped, so that completeness is always measured
against the intended sampling grid rather than against the records that
happen to exist. The dataset that results carries provenance on every row, and
the ingest fills nothing: every value in it is a measurement.
```

- [ ] **Step 5: Write section 4, the thermal compensation**

Append:

```latex
% ===========================================================================
\section{Thermal compensation}
\label{sec:compensation}

The inclinometer carries a measurement bias that varies with temperature: part
of what the transducer reports as a change in inclination is the instrument's
own thermal response rather than the wall's movement. The manufacturer
specifies a linear correction to cancel it, and that correction is applied to
every inclination series in this project:

\begin{equation*}
I_{\mathrm{comp}}(t) \;=\; I(t) \;-\; \bigl(T(t) - T_{\mathrm{ref}}\bigr)
\cdot k \cdot 1000
\end{equation*}

\noindent with $T$ the station's own on-board air temperature and
$T_{\mathrm{ref}}$ a fixed reference temperature.

\begin{center}
\small
\begin{tabularx}{\textwidth}{L{3.0cm}L{2.6cm}Y}
\toprule
\textbf{Parameter} & \textbf{Value} & \textbf{Meaning} \\
\midrule
$k$ & $0.005$ &
The manufacturer's coefficient, 5\,\mdegC \\
$T$ & \texttt{tair} &
The station's own air temperature, not the wall temperature \\
$T_{\mathrm{ref}}$ & first retained record &
Reference temperature of the correction \\
\bottomrule
\end{tabularx}
\end{center}

Two properties of this correction matter for reading anything below.

\textbf{It is instantaneous and linear.} There is no lag term: the model
assumes the instrument's thermal response is immediate and proportional. The
structure's own thermal response is not, which is why the framework's
feature-engineering stage exists at all.

\textbf{It fixes an arbitrary constant.} $T_{\mathrm{ref}}$ is taken at the
first retained record, so the compensated series carries a constant that is a
property of where the record happens to start rather than of the wall. In the
unified dataset the anchor is applied \emph{within each era separately}, which
is why the two eras of Section~\ref{sec:station} sit at unrelated levels by
construction. None of this affects the first difference, which is invariant to
every such constant.

\textbf{Every inclination series in this report is compensated.} The raw
channel is used in Section~\ref{sec:coverage} to count what was recorded, and
is never plotted as a signal.
```

- [ ] **Step 6: Compile what exists so far**

Run:
```bash
cd report
printf '\\end{document}\n' >> data_exploration_report.tex
pdflatex -interaction=nonstopmode data_exploration_report.tex >/dev/null
pdflatex -interaction=nonstopmode data_exploration_report.tex | tail -20
```
Expected: `Output written on data_exploration_report.pdf`. No `Undefined control sequence`, no `File ... not found`.

`LaTeX Warning: Reference 'sec:coverage' ... undefined` and the same for `sec:station` **are expected here** and are not errors: Section~1 refers forward to two sections that Task 6 has not written yet. They must be gone after Task 6.

Then remove the temporary `\end{document}` again — Task 6 adds sections 5 and 6 before it:
```bash
python - <<'PY'
p = 'data_exploration_report.tex'
s = open(p).read()
assert s.rstrip().endswith('\\end{document}')
open(p, 'w').write(s.rstrip()[:-len('\\end{document}')].rstrip() + '\n')
PY
cd ..
```

If `\mdegC` is reported undefined, the shared preamble was not found: check that the
`\input` path is `../../_shared/reportstyle.tex` and that the build is run from
`report/`.

- [ ] **Step 7: Check every static table against the specification**

Open `docs/raw-data-format.md` and verify, line by line:
Section 3.3 against the station table of §1; Sections 3.1 and 3.2 against the two column
tables of §3.1; Section 4 against the sentinel table of §3.2; Section 7.2 against the
formula and parameters of §4. Correct the report where they disagree — the specification
wins, except on the coefficient's provenance, which Task 7 corrects in the other
direction.

- [ ] **Step 8: Commit**

```bash
cd ../..
git add studies/data_exploration/report/data_exploration_report.tex
git commit -m "docs(data_exploration): report sections 1-4, the monitoring system"
```

---

### Task 6: The report, sections 5 to 8 — the record

**Files:**
- Modify: `studies/data_exploration/report/data_exploration_report.tex` (append)

**Interfaces:**
- Consumes: `outputs/DE_T01` to `DE_T04` (table bodies), `outputs/DE_F01` to `DE_F05` (figures), and the printed values from Tasks 2 to 4 for the narrative sentences
- Produces: the finished report PDF

Task 4 must have run, so that `outputs/` holds the table bodies and figures.

- [ ] **Step 1: Print the numbers the prose needs**

Run from `studies/data_exploration/`:
```bash
conda activate neuralprophet_env
python - <<'PY'
import pandas as pd
c = pd.read_csv('outputs/DE_01_station_classification.csv', index_col=0)
print(c[['eras', 'first_reading', 'last_reading', 'observed_days',
         'coverage_of_archive_%']].to_string())
o = pd.read_csv('outputs/DE_04_era_offset.csv', index_col=0)['value']
print('\nraw_step', o['raw_step'], ' seasonal', o['seasonal_step'],
      ' offset', o['offset'], ' gain_ratio', o['gain_ratio'])
s = pd.read_csv('outputs/DE_03_summary_by_era_st02.csv')
print('\n', s.to_string())
PY
```
Keep the output. Every value quoted in the prose below must match it; where this plan
quotes a number, verify it and correct the `.tex` if the run disagrees.

- [ ] **Step 2: Append section 5, the coverage**

The three observed-day figures below come from `DE_01` and are expected to read 1118.7,
2061.0 and 984.1. Replace them if Step 1 printed otherwise.

```latex
% ===========================================================================
\section{Coverage of the inclination record}
\label{sec:coverage}

The archive spans eight years of calendar time. It does not contain eight years
of measurement, and it never contained three stations' worth for more than a
fraction of that span.

\begin{center}
\small
\begin{tabular}{llllR{1.5cm}R{1.9cm}R{1.9cm}}
\toprule
\textbf{Station} & \textbf{Eras} & \textbf{First} & \textbf{Last} &
\textbf{Span [d]} & \textbf{Observed [d]} & \textbf{of span [\%]} \\
\midrule
\input{../outputs/DE_T01_station_classification.tex}
\bottomrule
\end{tabular}
\end{center}

\panel{The network failed in stages, not at the changeover}{%
It is tempting to read the February 2025 replacement as the moment the legacy
network ended. The record says otherwise: \textbf{station 01 stopped recording
in June 2022 and station 03 in June 2023} --- two years and eight months before
the replacement respectively.

\medskip
For most of the archive there was never a three-station network to compare
across. Any analysis requiring more than one location is confined to the period
before June 2022, and every study in this project that works on a single
station does so because the archive leaves no alternative.}

\begin{figure}[H]
\includegraphics{DE_F01_inclination_coverage}
\caption{Daily availability of the inclinometer at each station across the
whole archive. The dashed line is the 21 February 2025 instrument changeover.
The staggered right-hand edges are the point: the three records end three years
apart, and two of them end before the changeover.}
\end{figure}

\subsection{Year by year}

\begin{center}
\small
\begin{tabular}{lR{2.2cm}R{2.2cm}R{2.2cm}}
\toprule
\textbf{Year} & \textbf{\texttt{st01} [\%]} & \textbf{\texttt{st02} [\%]} &
\textbf{\texttt{st03} [\%]} \\
\midrule
\input{../outputs/DE_T02_yearly_coverage.tex}
\bottomrule
\end{tabular}
\end{center}

\noindent{\footnotesize Fraction of each calendar year in which the
inclinometer reported. 2018 and 2026 are partial years by definition: the
archive begins on 26 July 2018 and ends on 13 August 2026.}
```

- [ ] **Step 3: Append section 6, station 02**

The offset numbers below are expected to read −127.67, +4.81, −132.48 and 1.34. Replace
them if Step 1 printed otherwise.

```latex
% ===========================================================================
\section{Station 02, variable by variable}
\label{sec:station}

Station 02 stands at the Porta di Sant'Ubaldo. It is the best-covered record in
the archive and the only one instrumented in both eras, and it is the station
every later study in this project works on.

\begin{center}
\small
\begin{tabular}{llR{1.5cm}R{1.5cm}R{1.4cm}R{1.5cm}R{1.5cm}R{1.5cm}}
\toprule
\textbf{Era} & \textbf{Channel} & \textbf{Count} & \textbf{Mean} &
\textbf{SD} & \textbf{Min} & \textbf{Median} & \textbf{Max} \\
\midrule
\input{../outputs/DE_T03_summary_by_era.tex}
\bottomrule
\end{tabular}
\end{center}

\noindent{\footnotesize Units: \texttt{inc\_comp} in mdeg, \texttt{tair} in
\degC, \texttt{rh} in \%, \texttt{batt} in V. \textbf{The two eras are
summarised separately throughout.} A pooled mean of \texttt{inc\_comp} would
average two unrelated instrument baselines, and its standard deviation would be
dominated by the step between them rather than by anything the wall did.}

\subsection{Inclination}

\begin{figure}[H]
\includegraphics{DE_F02_st02_inclination}
\caption{The compensated inclination at station 02. (a) the whole archive, with
each era anchored separately. (b) the changeover at close range: the two
horizontal levels are the 30-day window means either side of it, and their
difference is the raw step. (c) the same record with the estimated instrument
offset removed.}
\end{figure}

The two eras do not agree on a level, and they were never expected to. The
package installed in February 2025 is a physical reinstallation: its
inclinometer sat differently in its mount, so its arbitrary zero is a different
arbitrary zero. Panel (a) shows the consequence directly.

\subsubsection*{How the offset is estimated}

The legacy record ends on 2025-02-19 and the current one begins on 2025-02-21,
an interruption of about a day, so a level comparison across the boundary is
possible. It is also misleading if taken naively, because the two comparison
windows sit at different points of the annual cycle and part of the step
between them is simply the season.

The procedure is therefore three steps. The mean of the last 30 days of legacy
readings is compared with the mean of the first 30 days of current-era ones,
giving the \emph{raw step}. A harmonic model fitted to the legacy era is
evaluated on both windows and subtracted, removing the part of the step the
season accounts for. What survives is attributed to the instrument, and is the
constant subtracted to produce \texttt{inc\_comp\_joined}.

\begin{center}
\small
\begin{tabular}{lR{2.8cm}}
\toprule
\textbf{Quantity} & \textbf{Value} \\
\midrule
\input{../outputs/DE_T04_era_offset.tex}
\bottomrule
\end{tabular}
\end{center}

\panel{A constant offset does not reconcile the two instruments}{%
The offset is estimable and the estimate is reported above. The diagnostic
beside it is the one that matters: the \textbf{diurnal amplitude} --- the
signal's response to the daily thermal cycle, which is a property of the wall
and not of the logger --- rises by roughly a third across the changeover.

\medskip
The wall did not become a third more responsive to sunlight overnight. Either
the new instrument has a different gain, or it is mounted where the thermal
response is larger, or both. \textbf{A single additive constant cannot absorb a
multiplicative difference}, so the joined series reconciles the \emph{levels}
of the two eras and does not reconcile their \emph{scales}.

\medskip
This is why \texttt{inc\_comp\_joined} exists as a clearly named derived column
and never as the default, and why any analysis spanning the changeover should
either work in the difference domain or report its results per era.}

\subsection{Air temperature}

\begin{figure}[H]
\includegraphics{DE_F03_st02_tair}
\caption{Air temperature at station 02, measured by the unit itself rather than
by a weather service. The annual cycle is the dominant feature and the outages
are visible as flat interruptions.}
\end{figure}

This is the temperature that enters the compensation of
Section~\ref{sec:compensation}. It is measured at the acquisition unit, not
inside the masonry, and it is not the wall temperature: the current-era package
reports that separately.

\subsection{Relative humidity}

\begin{figure}[H]
\includegraphics{DE_F04_st02_rh}
\caption{Relative humidity at station 02. The channel saturates at 100\,\% for
extended periods in winter.}
\end{figure}

\subsection{Supply voltage}

\begin{figure}[H]
\includegraphics{DE_F05_st02_batt}
\caption{Supply voltage of the station 02 acquisition unit. This is not a
structural quantity; it is the health of the system that produces every other
channel in this report.}
\end{figure}

The battery channel is included because it is the channel that explains the
record. The interruptions in Section~\ref{sec:coverage} are outages of the
acquisition unit, and when the unit is down every channel it carries is down
with it --- the inclinometer, the temperature and the humidity together.
```

- [ ] **Step 4: Append sections 7 and 8, and close the document**

```latex
% ===========================================================================
\section{Caveats}

\textbf{The compensation is instantaneous and linear.} It has no lag term,
while the structure's thermal response is lagged. The correction and the
project's own feature-engineering stage are making different assumptions about
the same physics.

\textbf{The joined series reconciles levels, not scales.} The difference in
diurnal amplitude across the changeover is not removed by an additive offset
and is not explained here.

\textbf{The raw channel contradicts the site's sign convention.} The wall
stands in an embankment, so daytime heating of the exposed valley face is
expected to tip it towards the mountain, which is a negative change by the
instrument's convention. Measured on the current era the \emph{raw} channel
instead moves at $+1.63$\,\mdegC\ against air temperature, and the legacy
instrument gives a comparable positive slope. The compensated series shows the
expected negative sign, but it does so by subtracting 5\,\mdegC\ from a channel
whose measured slope is positive. This is an unresolved contradiction, not a
result; it is documented in full in \texttt{docs/raw-data-format.md}
Section 7.5 and must be read before any sign in this project is interpreted.

\textbf{This report covers the four core channels only.} Solar radiation and
wall temperature exist in the current era alone and are not treated here. Nor
is the anatomy of the gaps: how long they are, when they start, and what
mechanism produced them. Both are natural extensions of this report.

% ===========================================================================
\section{Artefact inventory}

\begin{center}
\small
\begin{tabularx}{\textwidth}{L{4.6cm}Y}
\toprule
\textbf{Artefact} & \textbf{Content} \\
\midrule
\texttt{DE\_01} & Station classification: eras, extent, observed days,
coverage \\
\texttt{DE\_02} & Yearly coverage of the inclinometer, per station \\
\texttt{DE\_03} & Per-era summary statistics for the four core channels \\
\texttt{DE\_04} & The instrument offset and its diagnostics \\
\texttt{DE\_T01} -- \texttt{DE\_T04} & The same four tables as LaTeX bodies,
read directly by this report \\
\addlinespace
\texttt{DE\_F00} & Site map. \textbf{Supplied by hand, not generated} \\
\texttt{DE\_F01} & Inclinometer coverage, three stations \\
\texttt{DE\_F02} & Compensated inclination, the changeover, and the join \\
\texttt{DE\_F03} -- \texttt{DE\_F05} & Air temperature, relative humidity,
supply voltage \\
\bottomrule
\end{tabularx}
\end{center}

Reproduction, from the study directory with \texttt{neuralprophet\_env} active:

\begin{quote}
\ttfamily\small
jupytext -{}-to ipynb data\_exploration\_study.py \\
jupyter nbconvert -{}-to notebook -{}-execute -{}-inplace \\
\phantom{jupyter nbconvert }data\_exploration\_study.ipynb
\end{quote}

\texttt{studies/unified\_dataset/build\_unified\_dataset.py} must have run
first: this study reads its output and nothing else.

\end{document}
```

- [ ] **Step 5: Build the report**

Run:
```bash
cd report
pdflatex -interaction=nonstopmode data_exploration_report.tex >/dev/null
pdflatex -interaction=nonstopmode data_exploration_report.tex | tail -30
grep -c "LaTeX Warning: Reference" data_exploration_report.log || true
cd ..
```
Expected: `Output written on data_exploration_report.pdf` with a page count. No
`File ... not found` for any `DE_F` or `DE_T` name, and no undefined references.

If a `DE_T0*.tex` is reported missing, Task 4 has not been run since the table-body cells
were added — rerun the notebook.

- [ ] **Step 6: Read the PDF**

Open `report/data_exploration_report.pdf` and confirm: five figures appear, all four
`\input` tables have rows, the abstract and both `\panel` boxes render, and no table
overflows the text width.

- [ ] **Step 7: Commit**

```bash
cd ../..
git add studies/data_exploration/report/data_exploration_report.tex
git commit -m "docs(data_exploration): report sections 5-8, the record"
```

---

### Task 7: README and the specification correction

**Files:**
- Create: `studies/data_exploration/README.md` (overwrites the existing file)
- Modify: `docs/raw-data-format.md` (§7.4 item 3 only)

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: nothing consumed by later tasks

- [ ] **Step 1: Write the README**

Write `studies/data_exploration/README.md`:

```markdown
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

`data/interim/unified/unified_{station}_1h.csv`, produced by
`studies/unified_dataset/build_unified_dataset.py`. **That ingest must have run
first.** This study never opens a raw `.adc` file.

## Contents

| File | Role |
|---|---|
| `data_exploration_study.py` | The study, in `py:percent` format. **Edit this one.** |
| `data_exploration_study.ipynb` | Paired notebook. Generated by `jupytext`. |
| `report/` | Report source and PDF. |
| `outputs/` | Tables, LaTeX table bodies, and figures. Gitignored. |

There is no library file. The study imports `ud_lib` from
`../unified_dataset`; moving or renaming that study breaks it.

## The site map is supplied by hand

`outputs/DE_F00_site_map.png` is **not** generated by the notebook. Drop the
map of the three station locations there and uncomment the figure block near
the top of `report/data_exploration_report.tex`. The report compiles without
it.

## Reproducing

```bash
conda activate neuralprophet_env
cd studies/data_exploration
jupytext --to ipynb data_exploration_study.py
jupyter nbconvert --to notebook --execute --inplace data_exploration_study.ipynb
```

**Figures.** All plotting goes through the shared seaborn theme; set
`ud.set_context('paper')` in the configuration cell to re-export at manuscript
column size.

## Rebuilding the report

Run the notebook first — the report reads figures and table bodies from
`outputs/`, which is not committed.

```bash
cd report
pdflatex data_exploration_report.tex   # twice, for the ToC
```

The shared preamble is `../../_shared/reportstyle.tex` and stays inside the
package set shipped with a basic TeX Live installation.
```

- [ ] **Step 2: Correct the coefficient's provenance in the specification**

In `docs/raw-data-format.md`, Section 7.4, replace item 3 exactly:

Find:
```markdown
3. **`COMP_COEFF = 0.005` is a placeholder** carried from the module default and marked in the
   notebook as pending the calibration sheet. It has not been fitted to these data, and there is
   no reason to expect the instrument installed in 2025 to share a coefficient with the legacy
   units.
```

Replace with:
```markdown
3. **`COMP_COEFF = 0.005` is the manufacturer's specified coefficient**, 5 mdeg/°C, prescribed to
   cancel the temperature-induced measurement bias of the inclinometer. Two consequences survive
   that provenance and are worth keeping in view: the value has not been fitted to these data, so
   it is not a calibration of these particular instruments; and there is no guarantee that the
   package installed in 2025 shares the coefficient of the legacy units it replaced.
```

- [ ] **Step 3: Verify nothing else in the specification contradicts it**

Run:
```bash
cd ../..
grep -n "placeholder\|calibration sheet\|never been calibrated" docs/raw-data-format.md
```
Expected: no hit in Section 7.4. Section 7.5 must be untouched — its argument is that a
5 mdeg/°C correction applied to a channel whose measured slope is +1.63 mdeg/°C flips the
sign arithmetically, and that holds whoever specified the coefficient.

- [ ] **Step 4: Report the contradiction left standing**

`studies/unified_dataset/report/unified_imputation_report.tex` states that the coefficient
"has never been calibrated for these instruments", in the paragraph after the column table
in section 1.3 and again in its Caveats section. That wording remains literally true after
the correction above. **Do not change it.** Report both locations to the user and let them
decide.

- [ ] **Step 5: Commit**

```bash
git add studies/data_exploration/README.md docs/raw-data-format.md
git commit -m "docs: data_exploration README, and the compensation coefficient's provenance"
```

---

## Final verification

- [ ] **Run the notebook from clean and confirm every artefact**

```bash
conda activate neuralprophet_env
cd studies/data_exploration
rm -f outputs/DE_*
jupytext --to ipynb data_exploration_study.py
jupyter nbconvert --to notebook --execute --inplace data_exploration_study.ipynb
ls -1 outputs/ | sort
```
Expected, 18 files: four CSVs, four `.tex` bodies, and five PNG/SVG pairs.

- [ ] **Rebuild the report from those artefacts**

```bash
cd report
pdflatex -interaction=nonstopmode data_exploration_report.tex >/dev/null
pdflatex -interaction=nonstopmode data_exploration_report.tex | tail -5
```
Expected: `Output written on data_exploration_report.pdf`.

- [ ] **Confirm the scope boundaries held**

```bash
cd ../../..
git status --short
```
Expected: changes confined to `studies/data_exploration/` and `docs/raw-data-format.md`,
plus the two documents under `docs/superpowers/`. `studies/unified_dataset/ud_lib.py` must
not appear. `data_exploration_study.ipynb` appears only if the user has synced it.
