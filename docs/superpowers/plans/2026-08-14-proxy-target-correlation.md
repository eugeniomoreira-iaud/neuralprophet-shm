# Proxy-to-Target Correlation and Solar-Radiation Comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the proxy-comparison study's pooled-span correlation against the calibrated inclination with the block-and-tier treatment used by the sibling studies, and extend the whole target-relationship analysis to solar radiation on the same footing as air temperature.

**Architecture:** New library code goes into `studies/proxy_comparison/pc_lib.py` and is limited to four new functions plus two signature extensions; everything else composes functions that already exist in `ip_lib` and `tc_lib`, which `pc_lib` already imports. The notebook `proxy_comparison_study.py` gains one step before the existing Step 5 and six after it, each writing numbered artefacts to `outputs/`. The report `report/proxy_comparison_report.tex` is then rewritten from those artefacts.

**Tech Stack:** Python 3.10, pandas 2.x, numpy <2, statsmodels, matplotlib, seaborn, jupytext. Conda environment `neuralprophet_env`. No new dependencies.

## Global Constraints

- **Environment:** every command runs with `conda activate neuralprophet_env` first, from `studies/proxy_comparison/`.
- **Never edit `.ipynb` files.** All notebook work happens in `proxy_comparison_study.py`. The user syncs the pair.
- **The target is `inc_comp` and nothing else.** No alternative target, no re-estimation of the compensation coefficient, and no discussion anywhere in code or report of the on-structure temperature appearing inside the calibrated reading. The compensation is ground truth.
- **No statistic is computed on the pooled ragged span** except the one retained contrast, `P_08`. No window crosses the era boundary `2025-02-21`.
- **A first difference is never taken across a block boundary.** Levels use the union of a tier's blocks; differences are computed inside each block and concatenated.
- **Operator scans run on a tier's single longest block**, never the union, because delays, recursive filters and rolling means all read across adjacent rows.
- **Block parameters are the sibling defaults:** `min_days=20`, `max_gap_hours=6`.
- **Artefact numbering:** new tables `P_13` … `P_22`, new figures `P_F06` … `P_F08`. Nothing existing is renumbered. `P_09` and `P_F04` are retired in Task 9.
- **There is no pytest suite in this repository and pytest is not in `environment.yml`.** Tests are plain assert scripts run with `python`, following the existing `verify_env.py` idiom. They live in `studies/proxy_comparison/tests/` and build their own synthetic frames, so no test depends on the archive or on `data/`.
- **Repository state:** the repo is on branch `main` with a dirty working tree. Task 0 branches before anything else.
- **Commits:** each task ends with a commit step. Run it only if the user has authorised commits for this branch; otherwise `git add` the files and stop there, leaving the commit to the user.
- **Prose:** code comments, docstrings and all report text are written in full prose, in complete sentences.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `studies/proxy_comparison/pc_lib.py` | Solar geometry, daylight mask, tier machinery, and the two extended functions | 1–4 |
| `studies/proxy_comparison/tests/test_solar.py` | Solar elevation and daylight mask | 2 |
| `studies/proxy_comparison/tests/test_tiers.py` | Block selection, tier frames, block-safe differencing | 3 |
| `studies/proxy_comparison/tests/test_target_relationship.py` | Backward compatibility and block behaviour of the extended function | 4 |
| `studies/proxy_comparison/proxy_comparison_study.py` | Notebook steps 4a and 5a–5f, and the extended Step 6 | 5–12 |
| `studies/proxy_comparison/report/proxy_comparison_report.tex` | Report sections 3, 4, 5, 6, 7 and the artefact inventory | 13–15 |
| `studies/proxy_comparison/README.md` | Findings summary | 15 |
| `docs/proxy-data-dictionary.md` | Generated; gains a Site block | 1 |

---

## Task 0: Branch

**Files:**
- Modify: none

- [ ] **Step 1: Create the working branch**

The working tree is dirty on `main`. Branching carries the existing modifications across without committing them.

```bash
cd "$(git rev-parse --show-toplevel)"
git checkout -b feat/proxy-target-correlation
git branch --show-current
```

Expected output: `feat/proxy-target-correlation`

- [ ] **Step 2: Confirm the study's inputs are present**

```bash
ls data/interim/unified/unified_st02_1h.csv \
   data/raw/proxies/meteosystem_gubbio.csv \
   data/raw/proxies/oikolab_weather.csv
```

Expected: all three paths listed, no `No such file` error. If any is missing, stop — `studies/unified_dataset/build_unified_dataset.py` must run first.

---

## Task 1: Site coordinate provenance

**Files:**
- Modify: `studies/proxy_comparison/pc_lib.py` (the `SITE_LONGITUDE` / `SITE_LATITUDE` constants near line 837, and `write_data_dictionary` near line 1079)

**Interfaces:**
- Consumes: nothing.
- Produces: `pc.SITE_LATITUDE` and `pc.SITE_LONGITUDE` unchanged in value, now documented. `pc.write_data_dictionary(path, report=None, extents=None, notes=None)` unchanged in signature, with a Site block added to its output.

- [ ] **Step 1: Document the constants**

Find the two constants in `pc_lib.py`. They currently read:

```python
SITE_LONGITUDE = 12.582047

#: Site latitude, degrees north.
SITE_LATITUDE = 43.35343
```

Replace that whole region, including the comment line above `SITE_LONGITUDE`, with:

```python
# The site coordinates originate in ``auxiliary/oiko.py``, lines 84 and 85,
# where they are the ``LAT`` and ``LON`` arguments of the Oikolab API request
# that produced ``data/raw/proxies/oikolab_weather.csv``. They therefore fix two
# things at once: the point at which the ERA5 grid cell was extracted, and the
# solar geometry used by the clock tests and by the daylight mask. Changing one
# without the other would silently compare a reanalysis series against the sun
# seen from somewhere else, so the two must always be edited together.

#: Site longitude, degrees east. See the provenance note above.
SITE_LONGITUDE = 12.582047

#: Site latitude, degrees north. See the provenance note above.
SITE_LATITUDE = 43.35343
```

- [ ] **Step 2: Emit a Site block into the generated data dictionary**

Open `write_data_dictionary`. It builds a Markdown document and writes it to `path`. Locate the point where the document's opening lines are assembled, immediately before the first channel table is appended, and insert the Site block there:

```python
    lines.append('## Site')
    lines.append('')
    lines.append('The coordinates below fix both the point at which the ERA5 '
                 'grid cell was extracted and the solar geometry used by the '
                 'clock tests and the daylight mask.')
    lines.append('')
    lines.append('| Field | Value |')
    lines.append('|---|---|')
    lines.append(f'| Latitude | {SITE_LATITUDE:.6f} degrees north |')
    lines.append(f'| Longitude | {SITE_LONGITUDE:.6f} degrees east |')
    lines.append('| Origin | `auxiliary/oiko.py`, lines 84 and 85 |')
    lines.append('')
```

If the local accumulator is not named `lines`, use whatever name the function already uses; do not rename it. If the function builds one long f-string rather than a list, insert the equivalent Markdown block at the same position.

- [ ] **Step 3: Verify the block renders**

```bash
cd studies/proxy_comparison
python -c "
import sys; sys.path.insert(0, '.'); sys.path.insert(0, '../..')
import pandas as pd, pc_lib as pc
report = pd.DataFrame([{'channel': 'tair_str', 'quantity': 'tair', 'source': 'str',
                        'unit': 'degC', 'first': '2018-07-26 00:00:00',
                        'last': '2026-08-13 00:00:00', 'coverage_%': 65.5,
                        'min': -5.17, 'max': 38.93, 'mean': 13.93,
                        'outside_plausible': 0}])
out = pc.write_data_dictionary('/tmp/dict_check.md', report=report,
                               extents={'str': '2018-07-26 to 2026-08-13'})
text = open('/tmp/dict_check.md').read()
assert '## Site' in text, 'Site block missing'
assert '43.353430' in text, 'latitude missing'
assert '12.582047' in text, 'longitude missing'
assert 'auxiliary/oiko.py' in text, 'provenance missing'
print('Site block OK')
"
```

Expected: `Site block OK`

- [ ] **Step 4: Regenerate the real dictionary**

The dictionary is generated by the notebook, so it will be refreshed in Task 12 when the notebook is re-run end to end. No action here beyond confirming Step 3 passed.

- [ ] **Step 5: Commit**

```bash
git add studies/proxy_comparison/pc_lib.py
git commit -m "docs: record site coordinate provenance in pc_lib and the data dictionary"
```

---

## Task 2: Solar elevation and the daylight mask

**Files:**
- Modify: `studies/proxy_comparison/pc_lib.py` (add after `solar_noon_utc`, which ends around line 875)
- Create: `studies/proxy_comparison/tests/test_solar.py`

**Interfaces:**
- Consumes: `pc.SITE_LATITUDE`, `pc.SITE_LONGITUDE` from Task 1.
- Produces:
  - `pc.solar_elevation(index, latitude=SITE_LATITUDE, longitude=SITE_LONGITUDE) -> pd.Series` of floats in degrees, indexed by `index`.
  - `pc.daylight_mask(index, latitude=SITE_LATITUDE, longitude=SITE_LONGITUDE, min_elevation=0.0) -> pd.Series` of bools, indexed by `index`.

- [ ] **Step 1: Write the failing test**

Create `studies/proxy_comparison/tests/test_solar.py`:

```python
"""
Tests for the solar geometry added to pc_lib.

Run with:
    conda activate neuralprophet_env
    cd studies/proxy_comparison
    python tests/test_solar.py

These tests build their own timestamps and never read the archive or data/.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pc_lib as pc


def test_elevation_peaks_near_solar_noon():
    """The daily maximum elevation must fall within an hour of solar noon."""
    index = pd.date_range('2025-06-21', periods=24, freq='h')
    elevation = pc.solar_elevation(index)
    peak_hour = int(elevation.idxmax().hour)
    solar_noon = float(pc.solar_noon_utc(pd.DatetimeIndex(['2025-06-21'])).iloc[0])
    assert abs(peak_hour - solar_noon) <= 1.0, (
        f'peak at {peak_hour} h, solar noon at {solar_noon:.2f} h')


def test_summer_noon_higher_than_winter_noon():
    """At this latitude the June sun must stand well above the December sun."""
    june = pc.solar_elevation(pd.DatetimeIndex(['2025-06-21 12:00'])).iloc[0]
    december = pc.solar_elevation(pd.DatetimeIndex(['2025-12-21 12:00'])).iloc[0]
    assert june > december + 30.0, f'June {june:.1f}, December {december:.1f}'


def test_elevation_within_physical_bounds():
    """Elevation is an angle and cannot leave the range it is defined on."""
    index = pd.date_range('2025-01-01', periods=24 * 365, freq='h')
    elevation = pc.solar_elevation(index)
    assert elevation.between(-90.0, 90.0).all()
    assert elevation.notna().all()


def test_daylight_fraction_is_plausible():
    """Averaged over a year the sun is up for roughly half the hours."""
    index = pd.date_range('2025-01-01', periods=24 * 365, freq='h')
    fraction = float(pc.daylight_mask(index).mean())
    assert 0.45 < fraction < 0.55, f'daylight fraction {fraction:.3f}'


def test_daylight_longer_in_june_than_december():
    """Day length must vary with the season, which a fixed clock window cannot do."""
    june = float(pc.daylight_mask(
        pd.date_range('2025-06-21', periods=24, freq='h')).sum())
    december = float(pc.daylight_mask(
        pd.date_range('2025-12-21', periods=24, freq='h')).sum())
    assert june - december >= 5.0, f'June {june} h, December {december} h'


def test_mask_is_boolean_and_aligned():
    """The mask must be usable directly as a selector on the analysis frame."""
    index = pd.date_range('2025-03-01', periods=48, freq='h')
    mask = pc.daylight_mask(index)
    assert mask.dtype == bool
    assert mask.index.equals(index)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('all solar tests passed')
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd studies/proxy_comparison
python tests/test_solar.py
```

Expected: `AttributeError: module 'pc_lib' has no attribute 'solar_elevation'`

- [ ] **Step 3: Implement the two functions**

Insert into `pc_lib.py` immediately after `solar_noon_utc` and before `radiation_phase`:

```python
def solar_elevation(index, latitude=SITE_LATITUDE, longitude=SITE_LONGITUDE):
    """
    Solar elevation angle at each timestamp, by the NOAA formulation.

    The elevation is the angle of the sun above the horizon. It is computed from
    the date and the site coordinates alone, so it is independent of every
    measured channel and is defined at timestamps where a radiation sensor is
    missing. That independence is the reason it is preferred to a data-derived
    daylight indicator: the same mask can then be applied to all three sources
    without any of them helping to define it.

    The calculation takes no account of cloud, of the local horizon formed by
    the surrounding terrain, or of atmospheric refraction near the horizon. It
    answers where the sun is, not how much radiation reached the wall.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Timestamps to evaluate. Interpreted as UTC, consistent with
        :func:`solar_noon_utc`.
    latitude : float, optional
        Degrees north. Default :data:`SITE_LATITUDE`.
    longitude : float, optional
        Degrees east. Default :data:`SITE_LONGITUDE`.

    Returns
    -------
    pd.Series
        Elevation in degrees, indexed by ``index``.
    """
    index = pd.DatetimeIndex(index)
    doy = index.dayofyear.to_numpy(dtype=float)
    gamma = 2.0 * np.pi / 365.0 * (doy - 1.0)

    # The equation of time, in minutes, identical to the term used by
    # solar_noon_utc so that the two cannot drift apart.
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )

    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.001480 * np.sin(3 * gamma)
    )

    minutes = (index.hour.to_numpy(dtype=float) * 60.0
               + index.minute.to_numpy(dtype=float)
               + index.second.to_numpy(dtype=float) / 60.0)
    true_solar_time = minutes + eqtime + 4.0 * longitude
    hour_angle = np.radians(true_solar_time / 4.0 - 180.0)

    lat = np.radians(latitude)
    cos_zenith = (np.sin(lat) * np.sin(declination)
                  + np.cos(lat) * np.cos(declination) * np.cos(hour_angle))
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    return pd.Series(90.0 - np.degrees(np.arccos(cos_zenith)), index=index)


def daylight_mask(index, latitude=SITE_LATITUDE, longitude=SITE_LONGITUDE,
                  min_elevation=0.0):
    """
    True where the sun stands above the horizon.

    Roughly half of every solar-radiation record is a structural night-time
    zero. Those zeros are real measurements and are never removed from the
    record, but they distort every agreement statistic computed over the full
    day: correlations are inflated by a large block of samples on which all
    sources trivially agree, limits of agreement are computed over a bimodal
    distribution, and the trough hour of a mean diurnal cycle is an arbitrary
    point inside the night. This mask allows those statistics to be reported a
    second time over the daylight hours alone, beside the all-hours figures
    rather than in place of them.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Timestamps to evaluate.
    latitude : float, optional
        Degrees north. Default :data:`SITE_LATITUDE`.
    longitude : float, optional
        Degrees east. Default :data:`SITE_LONGITUDE`.
    min_elevation : float, optional
        Elevation threshold in degrees. Default ``0.0``, the geometric horizon.

    Returns
    -------
    pd.Series
        Boolean, indexed by ``index``.
    """
    return solar_elevation(index, latitude=latitude,
                           longitude=longitude) > min_elevation
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd studies/proxy_comparison
python tests/test_solar.py
```

Expected: six `PASS` lines then `all solar tests passed`.

- [ ] **Step 5: Commit**

```bash
git add studies/proxy_comparison/pc_lib.py studies/proxy_comparison/tests/test_solar.py
git commit -m "feat: add solar elevation and daylight mask to pc_lib"
```

---

## Task 3: Tier machinery

**Files:**
- Modify: `studies/proxy_comparison/pc_lib.py` (add after `daylight_mask`)
- Create: `studies/proxy_comparison/tests/test_tiers.py`

**Interfaces:**
- Consumes: `ip.contiguous_blocks(df, cols, min_days=20, max_gap_hours=6)` returning a frame with `start`, `end`, `days`, `coverage_%`, sorted longest first.
- Produces:
  - `pc.ERA_BOUNDARY` — `pd.Timestamp('2025-02-21')`.
  - `pc.tier_blocks(df, tiers, era_boundary=ERA_BOUNDARY, min_days=20, max_gap_hours=6) -> pd.DataFrame` with columns `tier`, `era`, `block`, `start`, `end`, `days`, `coverage_%`.
  - `pc.tier_index(inventory, tier) -> pd.DatetimeIndex` — the union of that tier's blocks.
  - `pc.tier_frame(df, inventory, tier) -> pd.DataFrame` — `df` restricted to that union.
  - `pc.tier_differences(df, inventory, tier, cols=None) -> pd.DataFrame` — first differences taken inside each block and concatenated, boundary rows dropped.
  - `pc.longest_block(inventory, tier) -> pd.Series` — the single longest block row of that tier.

- [ ] **Step 1: Write the failing test**

Create `studies/proxy_comparison/tests/test_tiers.py`:

```python
"""
Tests for the tier and block machinery added to pc_lib.

Run with:
    conda activate neuralprophet_env
    cd studies/proxy_comparison
    python tests/test_tiers.py

The frames here are synthetic. A deliberate multi-day hole splits the record so
that block-boundary behaviour is exercised rather than assumed.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pc_lib as pc


def build_frame():
    """
    A synthetic frame with one long hole and one short one.

    The record runs from 1 January 2025 to 1 January 2026 on the hourly grid.
    A 10-day hole in March splits it into two blocks. A 3-hour hole in May is
    short enough to be absorbed, because max_gap_hours is 6.
    """
    index = pd.date_range('2025-01-01', '2026-01-01', freq='h')
    df = pd.DataFrame({
        'inc_comp': np.linspace(0.0, 100.0, len(index)),
        'tair_gs': np.linspace(0.0, 50.0, len(index)),
        'sr_gs': np.linspace(0.0, 900.0, len(index)),
        'tair_str': np.linspace(0.0, 50.0, len(index)),
        'sr_str': np.linspace(0.0, 900.0, len(index)),
    }, index=index)
    df.loc['2025-03-01':'2025-03-11', :] = np.nan
    df.loc['2025-05-01 00:00':'2025-05-01 02:00', :] = np.nan
    # The on-structure radiation channel exists only in the current era.
    df.loc[:'2025-02-20 23:00', 'sr_str'] = np.nan
    return df


def test_short_hole_absorbed_long_hole_splits():
    """A 3-hour interruption stays inside a block; a 10-day one ends it."""
    df = build_frame()
    inventory = pc.tier_blocks(
        df, {'T1': ['inc_comp', 'tair_gs', 'sr_gs']},
        era_boundary=pd.Timestamp('2025-02-21'))
    current = inventory[inventory['era'] == 'current']
    assert len(current) == 2, f'expected 2 blocks, got {len(current)}'
    assert current['days'].max() > 250, 'the post-March block should be long'


def test_no_block_crosses_the_era_boundary():
    """Every block lies wholly inside one era."""
    df = build_frame()
    boundary = pd.Timestamp('2025-02-21')
    inventory = pc.tier_blocks(df, {'T1': ['inc_comp', 'tair_gs', 'sr_gs']},
                               era_boundary=boundary)
    for _, row in inventory.iterrows():
        if row['era'] == 'legacy':
            assert row['end'] < boundary
        else:
            assert row['start'] >= boundary


def test_tier_requiring_sr_str_is_current_era_only():
    """A tier requiring the short channel cannot reach the legacy era."""
    df = build_frame()
    inventory = pc.tier_blocks(
        df, {'P0': ['inc_comp', 'sr_str'], 'P1': ['inc_comp', 'sr_gs']},
        era_boundary=pd.Timestamp('2025-02-21'))
    p0 = inventory[inventory['tier'] == 'P0']
    assert (p0['era'] == 'current').all(), 'P0 leaked into the legacy era'
    assert 'legacy' in set(inventory[inventory['tier'] == 'P1']['era'])


def test_tier_index_is_the_union_of_its_blocks():
    """The union index covers every block and nothing outside them."""
    df = build_frame()
    inventory = pc.tier_blocks(df, {'T1': ['inc_comp', 'tair_gs']},
                               era_boundary=pd.Timestamp('2025-02-21'))
    index = pc.tier_index(inventory, 'T1')
    assert index.is_monotonic_increasing
    assert index.is_unique
    # No timestamp from the long March hole survives.
    assert not index.isin(pd.date_range('2025-03-03', '2025-03-09',
                                        freq='h')).any()


def test_differences_are_not_taken_across_a_block_boundary():
    """
    The step manufactured by differencing across the March hole must not appear.

    The synthetic series rises by a constant amount per hour, so every honest
    first difference is that same constant. A difference taken across the
    10-day hole would be 241 times larger, and this test fails if one survives.
    """
    df = build_frame()
    inventory = pc.tier_blocks(df, {'T1': ['inc_comp', 'tair_gs']},
                               era_boundary=pd.Timestamp('2025-02-21'))
    diffs = pc.tier_differences(df, inventory, 'T1', cols=['inc_comp'])
    step = diffs['inc_comp'].dropna()
    assert len(step) > 1000, 'too few differences survived'
    assert step.max() < step.median() * 2.0, (
        f'a boundary step survived: max {step.max():.4f} '
        f'against median {step.median():.4f}')


def test_longest_block_is_the_longest():
    """The scan window selector returns the longest block, not the first."""
    df = build_frame()
    inventory = pc.tier_blocks(df, {'T1': ['inc_comp', 'tair_gs']},
                               era_boundary=pd.Timestamp('2025-02-21'))
    block = pc.longest_block(inventory, 'T1')
    tier_rows = inventory[inventory['tier'] == 'T1']
    assert block['days'] == tier_rows['days'].max()


def test_missing_tier_returns_empty_rather_than_raising():
    """A tier whose channels never coexist is reported, not an exception."""
    df = build_frame()
    df['never'] = np.nan
    inventory = pc.tier_blocks(df, {'T1': ['inc_comp', 'never']},
                               era_boundary=pd.Timestamp('2025-02-21'))
    assert len(inventory) == 0
    assert list(inventory.columns) == ['tier', 'era', 'block', 'start', 'end',
                                       'days', 'coverage_%']


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('all tier tests passed')
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd studies/proxy_comparison
python tests/test_tiers.py
```

Expected: `AttributeError: module 'pc_lib' has no attribute 'tier_blocks'`

- [ ] **Step 3: Implement the tier machinery**

Insert into `pc_lib.py` after `daylight_mask`:

```python
#: The instrument at station st02 was physically reinstalled on this date. The
#: legacy package and the current one share no baseline, so no window and no
#: statistic in this study may span the boundary.
ERA_BOUNDARY = pd.Timestamp('2025-02-21')

#: Column order of the block inventory, fixed so that an empty inventory has the
#: same shape as a populated one and downstream code needs no special case.
BLOCK_COLUMNS = ['tier', 'era', 'block', 'start', 'end', 'days', 'coverage_%']


def tier_blocks(df, tiers, era_boundary=ERA_BOUNDARY, min_days=20,
                max_gap_hours=6):
    """
    Contiguous blocks per tier, within each instrument era.

    A tier is a set of columns that must be simultaneously present. The
    binding channel of this study is the on-structure pyranometer, which
    exists only from the era boundary onwards, so a tier requiring it can never
    reach the legacy era. That asymmetry is not worked around: it is what the
    tiers exist to express.

    Blocks are found within an era rather than across the whole record, because
    the instrument was physically reinstalled at the boundary and the two
    packages share no baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame on a regular hourly index.
    tiers : dict
        Tier name to the list of columns that tier requires.
    era_boundary : pd.Timestamp, optional
        First timestamp of the current era. Default :data:`ERA_BOUNDARY`.
    min_days : int, optional
        Shortest block reported, in days. Default ``20``, the value used by the
        sibling studies, so inventories are directly comparable.
    max_gap_hours : int, optional
        Longest interruption absorbed into a block. Default ``6``.

    Returns
    -------
    pd.DataFrame
        One row per block, with the columns of :data:`BLOCK_COLUMNS`. Empty with
        those same columns if no tier yields a block.
    """
    eras = {
        'legacy': df.loc[df.index < era_boundary],
        'current': df.loc[df.index >= era_boundary],
    }

    rows = []
    for tier, cols in tiers.items():
        present = [c for c in cols if c in df.columns]
        if len(present) < len(cols):
            continue
        for era, frame in eras.items():
            if frame.empty:
                continue
            found = ip.contiguous_blocks(frame, present, min_days=min_days,
                                         max_gap_hours=max_gap_hours)
            for n, (_, block) in enumerate(found.iterrows(), start=1):
                rows.append({
                    'tier': tier,
                    'era': era,
                    'block': f'{tier}-{era[:3]}{n}',
                    'start': block['start'],
                    'end': block['end'],
                    'days': block['days'],
                    'coverage_%': block['coverage_%'],
                })

    if not rows:
        return pd.DataFrame(columns=BLOCK_COLUMNS)
    return (pd.DataFrame(rows)[BLOCK_COLUMNS]
            .sort_values(['tier', 'days'], ascending=[True, False])
            .reset_index(drop=True))


def tier_index(inventory, tier):
    """
    The union of a tier's blocks, as a single sorted index.

    This is the window for every *level* statistic. Differenced statistics must
    not use it directly — see :func:`tier_differences`.

    Parameters
    ----------
    inventory : pd.DataFrame
        Output of :func:`tier_blocks`.
    tier : str
        Tier name.

    Returns
    -------
    pd.DatetimeIndex
        Sorted and unique. Empty if the tier has no block.
    """
    parts = [pd.date_range(row['start'], row['end'], freq='h')
             for _, row in inventory[inventory['tier'] == tier].iterrows()]
    if not parts:
        return pd.DatetimeIndex([])
    return parts[0].union_many(parts[1:]) if len(parts) > 1 else parts[0]


def tier_frame(df, inventory, tier):
    """
    ``df`` restricted to the union of a tier's blocks.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame.
    inventory : pd.DataFrame
        Output of :func:`tier_blocks`.
    tier : str
        Tier name.

    Returns
    -------
    pd.DataFrame
        Rows of ``df`` inside the tier's blocks, in index order.
    """
    index = tier_index(inventory, tier)
    return df.loc[df.index.intersection(index)]


def tier_differences(df, inventory, tier, cols=None):
    """
    First differences taken inside each block and concatenated.

    A first difference must never be taken across a block boundary. The two
    rows on either side of a boundary are both valid and both present, but they
    are not adjacent in time, and differencing them manufactures a step out of
    a gap. Differencing per block and concatenating afterwards is the only
    correct way to obtain a differenced frame over a tier.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame.
    inventory : pd.DataFrame
        Output of :func:`tier_blocks`.
    tier : str
        Tier name.
    cols : list of str or None, optional
        Columns to difference. Default ``None``, meaning every column of ``df``.

    Returns
    -------
    pd.DataFrame
        Differences, with the first row of every block dropped.
    """
    cols = list(df.columns) if cols is None else list(cols)
    parts = []
    for _, row in inventory[inventory['tier'] == tier].iterrows():
        block = df.loc[row['start']:row['end'], cols]
        if len(block) < 2:
            continue
        parts.append(block.diff().iloc[1:])
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts).sort_index()


def longest_block(inventory, tier):
    """
    The single longest block of a tier.

    Operator scans run on one contiguous block rather than on the union,
    because a transport delay is applied with a shift, a time constant with a
    recursive filter, and the band limit with a rolling mean. Every one of
    those reads across adjacent rows, so a union would let a filter draw values
    from the far side of a multi-month interruption.

    Parameters
    ----------
    inventory : pd.DataFrame
        Output of :func:`tier_blocks`.
    tier : str
        Tier name.

    Returns
    -------
    pd.Series or None
        The inventory row of the longest block, or ``None`` if the tier is
        empty.
    """
    rows = inventory[inventory['tier'] == tier]
    if rows.empty:
        return None
    return rows.loc[rows['days'].idxmax()]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd studies/proxy_comparison
python tests/test_tiers.py
```

Expected: seven `PASS` lines then `all tier tests passed`.

If `union_many` raises a deprecation error on the installed pandas, replace the body of `tier_index` with the loop form:

```python
    index = parts[0]
    for part in parts[1:]:
        index = index.union(part)
    return index
```

and re-run.

- [ ] **Step 5: Commit**

```bash
git add studies/proxy_comparison/pc_lib.py studies/proxy_comparison/tests/test_tiers.py
git commit -m "feat: add era-aware tier and block machinery to pc_lib"
```

---

## Task 4: Extend `target_relationship` with tier blocks

**Files:**
- Modify: `studies/proxy_comparison/pc_lib.py:673-745` (the `target_relationship` function)
- Create: `studies/proxy_comparison/tests/test_target_relationship.py`

**Interfaces:**
- Consumes: `pc.tier_frame`, `pc.tier_differences` from Task 3.
- Produces: `pc.target_relationship(df, quantity, target=TARGET, sources=SOURCES, hac_lags=24, min_days=30, dt_hours=1.0, blocks=None, tier=None)`. With `blocks=None` the behaviour and output columns are byte-for-byte what they are today. With `blocks` and `tier` given, the output gains a `tier` column.

- [ ] **Step 1: Write the failing test**

Create `studies/proxy_comparison/tests/test_target_relationship.py`:

```python
"""
Tests for the block-aware extension of pc.target_relationship.

Run with:
    conda activate neuralprophet_env
    cd studies/proxy_comparison
    python tests/test_target_relationship.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pc_lib as pc


def build_frame(seed=0):
    """
    Two clean blocks separated by a long hole, with a known linear relationship.

    Inside each block the target is -3.0 times the driver plus noise, so a
    correct fit must recover a slope near -3.0 in both domains.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range('2025-03-01', periods=24 * 400, freq='h')
    driver = pd.Series(
        10.0 + 8.0 * np.sin(2 * np.pi * np.arange(len(index)) / 24.0)
        + rng.normal(0.0, 0.5, len(index)), index=index)
    target = -3.0 * driver + rng.normal(0.0, 0.5, len(index))
    df = pd.DataFrame({'inc_comp': target, 'tair_gs': driver})
    df.loc['2025-07-01':'2025-08-15', :] = np.nan
    return df


def test_default_call_is_unchanged():
    """Omitting blocks must reproduce today's behaviour and columns exactly."""
    df = build_frame()
    out = pc.target_relationship(df, 'tair', sources=('gs',))
    expected = ['quantity', 'source', 'domain', 'n', 'overlap_days', 'r',
                'slope_mdeg_per_unit', 'slope_se', 't', 'r2']
    assert list(out.columns) == expected, list(out.columns)
    assert set(out['domain']) == {'levels', 'differences'}
    assert 'tier' not in out.columns


def test_block_call_recovers_the_known_slope():
    """On clean blocks the fit must return the slope that built the data."""
    df = build_frame()
    inventory = pc.tier_blocks(df, {'T1': ['inc_comp', 'tair_gs']},
                               era_boundary=pd.Timestamp('2025-02-21'))
    out = pc.target_relationship(df, 'tair', sources=('gs',),
                                 blocks=inventory, tier='T1')
    levels = out[out['domain'] == 'levels'].iloc[0]
    assert abs(levels['slope_mdeg_per_unit'] + 3.0) < 0.1, levels
    assert levels['tier'] == 'T1'


def test_block_call_reports_fewer_rows_than_the_pooled_call():
    """Restricting to blocks must drop the hours inside the hole."""
    df = build_frame()
    inventory = pc.tier_blocks(df, {'T1': ['inc_comp', 'tair_gs']},
                               era_boundary=pd.Timestamp('2025-02-21'))
    pooled = pc.target_relationship(df, 'tair', sources=('gs',))
    tiered = pc.target_relationship(df, 'tair', sources=('gs',),
                                    blocks=inventory, tier='T1')
    pooled_n = int(pooled[pooled['domain'] == 'levels']['n'].iloc[0])
    tiered_n = int(tiered[tiered['domain'] == 'levels']['n'].iloc[0])
    assert tiered_n <= pooled_n


def test_differences_do_not_cross_the_hole():
    """
    A difference across the 45-day hole would wreck the differenced fit.

    With per-block differencing the differenced slope must still be near -3.0.
    """
    df = build_frame()
    inventory = pc.tier_blocks(df, {'T1': ['inc_comp', 'tair_gs']},
                               era_boundary=pd.Timestamp('2025-02-21'))
    out = pc.target_relationship(df, 'tair', sources=('gs',),
                                 blocks=inventory, tier='T1')
    diff = out[out['domain'] == 'differences'].iloc[0]
    assert abs(diff['slope_mdeg_per_unit'] + 3.0) < 0.2, diff


def test_empty_tier_returns_empty_frame():
    """A tier with no block yields no rows rather than an exception."""
    df = build_frame()
    empty = pd.DataFrame(columns=pc.BLOCK_COLUMNS)
    out = pc.target_relationship(df, 'tair', sources=('gs',),
                                 blocks=empty, tier='T1')
    assert len(out) == 0


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('all target-relationship tests passed')
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd studies/proxy_comparison
python tests/test_target_relationship.py
```

Expected: `TypeError: target_relationship() got an unexpected keyword argument 'blocks'`

- [ ] **Step 3: Extend the function**

Change the signature of `target_relationship` from:

```python
def target_relationship(df, quantity, target=TARGET, sources=SOURCES,
                        hac_lags=24, min_days=30, dt_hours=1.0):
```

to:

```python
def target_relationship(df, quantity, target=TARGET, sources=SOURCES,
                        hac_lags=24, min_days=30, dt_hours=1.0,
                        blocks=None, tier=None):
```

Add to the docstring's `Parameters` section, after `dt_hours`:

```
    blocks : pd.DataFrame or None, optional
        Block inventory from :func:`tier_blocks`. When given together with
        ``tier``, every statistic is computed inside that tier's blocks: levels
        on the union, differences taken within each block and concatenated. The
        default ``None`` computes on the whole frame, which pools both
        instrument eras and every ragged stretch between them, and is retained
        only as the contrast case.
    tier : str or None, optional
        Tier name to select from ``blocks``. Required when ``blocks`` is given.
```

Replace the loop body. The current code is:

```python
    rows = []
    for source in sources:
        col = f'{quantity}_{source}'
        if col not in df.columns:
            continue
        for domain, frame in (
                ('levels', df[[target, col]]),
                ('differences', df[[target, col]].diff().where(
                    df[[target, col]].notna()
                    & df[[target, col]].shift().notna()))):
```

Replace those lines with:

```python
    if blocks is not None and tier is None:
        raise ValueError('tier must be given when blocks is given')

    rows = []
    for source in sources:
        col = f'{quantity}_{source}'
        if col not in df.columns:
            continue

        if blocks is None:
            levels = df[[target, col]]
            differences = df[[target, col]].diff().where(
                df[[target, col]].notna()
                & df[[target, col]].shift().notna())
        else:
            levels = tier_frame(df, blocks, tier)[[target, col]]
            differences = tier_differences(df, blocks, tier,
                                           cols=[target, col])

        for domain, frame in (('levels', levels),
                              ('differences', differences)):
```

Then, inside the `rows.append({...})` call, add the tier field as the second entry so the column order stays readable:

```python
            rows.append({
                'quantity': quantity,
                'tier': tier,
                'source': source,
                ...
```

Finally, replace the closing `return pd.DataFrame(rows)` with:

```python
    out = pd.DataFrame(rows)
    if blocks is None and 'tier' in out.columns:
        out = out.drop(columns='tier')
    if not len(out):
        return pd.DataFrame(columns=(
            ['quantity'] + (['tier'] if blocks is not None else [])
            + ['source', 'domain', 'n', 'overlap_days', 'r',
               'slope_mdeg_per_unit', 'slope_se', 't', 'r2']))
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd studies/proxy_comparison
python tests/test_target_relationship.py
```

Expected: five `PASS` lines then `all target-relationship tests passed`.

- [ ] **Step 5: Re-run the earlier suites to confirm nothing regressed**

```bash
cd studies/proxy_comparison
python tests/test_solar.py && python tests/test_tiers.py
```

Expected: both suites pass.

- [ ] **Step 6: Commit**

```bash
git add studies/proxy_comparison/pc_lib.py \
        studies/proxy_comparison/tests/test_target_relationship.py
git commit -m "feat: compute the target relationship within tier blocks"
```

---

## Task 5: Notebook Step 4a — the daylight variant

**Files:**
- Modify: `studies/proxy_comparison/proxy_comparison_study.py` (insert after the `P_07` difference-agreement cell, before the `plot_diurnal_profiles` cell)

**Interfaces:**
- Consumes: `pc.daylight_mask` from Task 2.
- Produces: `outputs/P_21_sr_daylight_agreement.csv`, and the in-memory frames `daylight_agreement`, `daylight_differences`, `daylight_diurnal`.

- [ ] **Step 1: Add the Markdown header cell**

Insert:

```python
# %% [markdown]
# ### Solar radiation, over the daylight hours alone
#
# Roughly half of every radiation record is a structural night-time zero. Those
# zeros are real measurements and are never removed, but they distort every
# statistic computed over the whole day: the correlation is inflated by a large
# block of samples on which all three sources trivially agree, the limits of
# agreement are computed over a bimodal distribution, and the trough hour of the
# mean diurnal cycle is an arbitrary point inside the night — the all-hours
# table above reports 23, 1 and 2 for the three sources, which are not findings.
#
# The mask is geometric: the sun's elevation above the horizon, computed from
# the site coordinates alone. It is therefore independent of all three sources,
# it is defined at timestamps where a radiation sensor is missing, and the same
# mask applies to every channel so none of them helps define it. It takes no
# account of cloud or of the local horizon formed by the surrounding terrain.
#
# These figures are reported **beside** the all-hours figures, never in place of
# them.
#
# ### Parameter Tuning Guidance
#
# | Parameter | Purpose | Default | Effect |
# |---|---|---|---|
# | `MIN_ELEVATION` | Solar elevation above which an hour counts as daylight | `0.0` | Raising it to 5-10 degrees drops the shallow-sun hours near sunrise and sunset, where a wall's self-shading and the terrain horizon matter most; it also shortens the usable day and reduces `n`. |
```

- [ ] **Step 2: Add the code cell**

```python
# %%
MIN_ELEVATION = 0.0

daylight = pc.daylight_mask(df.index, min_elevation=MIN_ELEVATION)
df_day = df.loc[daylight]
print(f'daylight hours: {int(daylight.sum())} of {len(daylight)} '
      f'({100.0 * daylight.mean():.1f} %)')

daylight_agreement = pc.pairwise_agreement(
    df_day, 'sr', min_days=MIN_OVERLAP_DAYS, dt_hours=SAMPLING_HOURS)
daylight_differences = pc.difference_agreement(
    df_day, 'sr', min_days=MIN_OVERLAP_DAYS, dt_hours=SAMPLING_HOURS)
daylight_diurnal = pc.diurnal_comparison(df_day, 'sr')

all_hours = agreement[agreement['quantity'] == 'sr'].set_index(
    ['reference', 'compared'])
day_hours = daylight_agreement.set_index(['reference', 'compared'])
contrast = pd.DataFrame({
    'r_all_hours': all_hours['r'],
    'r_daylight': day_hours['r'],
    'MAE_all_hours': all_hours['MAE'],
    'MAE_daylight': day_hours['MAE'],
})
print('\nSolar radiation, all hours against daylight only:')
display(contrast)

pd.concat({'agreement': daylight_agreement.set_index(['reference', 'compared']),
           'differences': daylight_differences.set_index(['reference',
                                                          'compared'])},
          names=['statistic']).to_csv(
    f'{OUTPUT_DIR}/P_21_sr_daylight_agreement.csv')
daylight_diurnal.to_csv(f'{OUTPUT_DIR}/P_21b_sr_daylight_diurnal.csv')

assert 0.40 < float(daylight.mean()) < 0.60, \
    'daylight guard: the mask does not select roughly half the record'
print('\nDaylight guard passed.')
```

- [ ] **Step 3: Run the notebook up to this cell**

```bash
cd studies/proxy_comparison
rm -f proxy_comparison_study.ipynb
jupytext --to ipynb proxy_comparison_study.py
jupyter nbconvert --to notebook --execute --inplace proxy_comparison_study.ipynb
```

Expected: the notebook runs to completion, and `Daylight guard passed.` appears in the output.

- [ ] **Step 4: Verify the artefacts exist and are non-trivial**

```bash
cd studies/proxy_comparison
python -c "
import pandas as pd
a = pd.read_csv('outputs/P_21_sr_daylight_agreement.csv')
d = pd.read_csv('outputs/P_21b_sr_daylight_diurnal.csv')
assert len(a) >= 4, a
assert len(d) == 3, d
print(a.to_string()); print(); print(d.to_string())
"
```

Expected: both tables print with populated rows. Record the daylight correlations — they are needed for the report in Task 13.

- [ ] **Step 5: Commit**

```bash
git add studies/proxy_comparison/proxy_comparison_study.py
git commit -m "feat: report solar radiation agreement over daylight hours"
```

---

## Task 6: Notebook Step 5a — target verification, eras, tiers and blocks

**Files:**
- Modify: `studies/proxy_comparison/proxy_comparison_study.py` (insert after the existing Step 5 `P_08` cell and its target guard)

**Interfaces:**
- Consumes: `pc.tier_blocks`, `pc.ERA_BOUNDARY` from Task 3.
- Produces: `outputs/P_13_block_inventory.csv`, and the in-memory `TIERS` dict and `inventory_blocks` frame used by Tasks 7–11.

- [ ] **Step 1: Add the Markdown header cell**

```python
# %% [markdown]
# ## Step 5a · The windows the comparison can actually use
#
# Every statistic above this point is computed on the joined frame, which spans
# July 2018 to August 2026, contains both instrument eras, and is ragged. The
# sibling studies never correlate against the target on a span of that kind.
# They cut contiguous blocks of near-complete coverage, group them into tiers
# according to which channels a tier requires, and report every correlation,
# slope and operator inside a tier.
#
# The difference is not cosmetic. `inclination_prediction` reports the
# calibrated inclination against on-structure air temperature at r = -0.956 in
# levels and -0.886 in first differences on its Tier 1 blocks; the pooled
# computation in `P_08` above returns -0.524 and -0.155 for the same pair. For
# solar radiation the pooled computation reverses the sign outright.
#
# The radiation record settles the shape of the tiers by itself. `sr_str` begins
# on 2025-02-21, which is the era boundary, so the three-source radiation
# comparison lives entirely inside the current era, and the two proxies
# additionally reach into a legacy era the on-structure channel can never enter.
#
# ### Parameter Tuning Guidance
#
# | Parameter | Purpose | Default | Effect |
# |---|---|---|---|
# | `BLOCK_MIN_DAYS` | Shortest block reported | `20` | The sibling studies' value, kept so block inventories are comparable across studies. Raising it discards short usable windows; lowering it admits blocks too short for a seasonal statistic. |
# | `BLOCK_MAX_GAP_H` | Longest interruption absorbed into a block | `6` | Short interruptions are bridged because the statistics score only at observed timestamps. Raising it merges genuinely separate windows; lowering it fragments the record. |
# | `TIERS` | Which channels each tier requires | see cell | A tier requiring an on-structure channel cannot reach the legacy era. Adding a channel to a tier can only shorten its blocks. |
```

- [ ] **Step 2: Add the target verification cell**

```python
# %%
# The discrepancy with the sibling study is attributed to the analysis window
# only if the two studies are measuring the same target. inclination_prediction
# builds inc_comp through ip.add_target inside its own current-era loader; this
# study takes it from ud.load_unified. The two are compared directly before any
# conclusion rests on the difference.
import ip_lib as ip

sensor_current = sensor_raw.loc[sensor_raw.index >= pc.ERA_BOUNDARY].copy()
rebuilt = ip.add_target(sensor_current)
common = df.index.intersection(rebuilt.index)
delta = (df.loc[common, TARGET] - rebuilt.loc[common, TARGET]).dropna()

if len(delta):
    print(f'target check over {len(delta)} common hours: '
          f'max |difference| {delta.abs().max():.6f} mdeg, '
          f'mean {delta.mean():+.6f} mdeg')
else:
    print('target check: no common hours, the two constructions cannot be '
          'compared on this frame')

TARGET_TOLERANCE_MDEG = 1e-6
targets_identical = bool(len(delta)) and delta.abs().max() < TARGET_TOLERANCE_MDEG
print(f'targets identical: {targets_identical}')
```

If `ip.add_target` requires arguments this call does not supply, read its signature at `studies/inclination_prediction/ip_lib.py:128` and pass the defaults it documents. If the two targets are **not** identical, do not stop: record the magnitude of the difference, and note it in the report's caveats in Task 15 instead of attributing the whole discrepancy to windowing.

- [ ] **Step 3: Add the tier and block cell**

```python
# %%
BLOCK_MIN_DAYS = 20
BLOCK_MAX_GAP_H = 6

TIERS = {
    'P0': ['inc_comp', 'tair_str', 'sr_str',
           'tair_gs', 'sr_gs', 'tair_era5', 'sr_era5'],
    'P1': ['inc_comp', 'tair_gs', 'sr_gs', 'tair_era5', 'sr_era5'],
    'P2': ['inc_comp', 'tair_gs', 'sr_gs', 'tair_era5', 'sr_era5'],
}

inventory_blocks = pc.tier_blocks(
    df, TIERS, era_boundary=pc.ERA_BOUNDARY,
    min_days=BLOCK_MIN_DAYS, max_gap_hours=BLOCK_MAX_GAP_H)

# P0 and P1 are current-era tiers; P2 is the legacy-era window of the same
# proxy-only channel set. tier_blocks finds blocks in both eras for every tier,
# so each tier is reduced to the era it is defined for.
ERA_OF_TIER = {'P0': 'current', 'P1': 'current', 'P2': 'legacy'}
inventory_blocks = inventory_blocks[
    [ERA_OF_TIER[t] == e
     for t, e in zip(inventory_blocks['tier'], inventory_blocks['era'])]
].reset_index(drop=True)

display(inventory_blocks)
inventory_blocks.to_csv(f'{OUTPUT_DIR}/P_13_block_inventory.csv', index=False)

TIER_LABEL = {
    'P0': 'all three sources, current era',
    'P1': 'proxies only, current era',
    'P2': 'proxies only, legacy era',
}
for tier in TIERS:
    rows = inventory_blocks[inventory_blocks['tier'] == tier]
    print(f'{tier} ({TIER_LABEL[tier]}): {len(rows)} block(s), '
          f'{rows["days"].sum():.1f} days total, '
          f'longest {rows["days"].max() if len(rows) else 0:.1f} days')

for tier in TIERS:
    rows = inventory_blocks[inventory_blocks['tier'] == tier]
    assert len(rows) and rows['days'].max() >= MIN_OVERLAP_DAYS, \
        f'tier guard: {tier} has no block of at least {MIN_OVERLAP_DAYS} days'
print(f'\nTier guard passed: every tier holds a block of at least '
      f'{MIN_OVERLAP_DAYS} days.')

for _, row in inventory_blocks.iterrows():
    if row['era'] == 'legacy':
        assert row['end'] < pc.ERA_BOUNDARY, f'{row["block"]} crosses the era boundary'
    else:
        assert row['start'] >= pc.ERA_BOUNDARY, f'{row["block"]} crosses the era boundary'
print('Era guard passed: no block spans the 2025-02-21 reinstallation.')
```

- [ ] **Step 4: Run the notebook and confirm both guards pass**

```bash
cd studies/proxy_comparison
rm -f proxy_comparison_study.ipynb
jupytext --to ipynb proxy_comparison_study.py
jupyter nbconvert --to notebook --execute --inplace proxy_comparison_study.ipynb
```

Expected: `Tier guard passed` and `Era guard passed` both appear.

If the tier guard fails for `P0`, the on-structure channels do not coexist with the target for 30 contiguous days anywhere. That is a finding, not a bug: report it, lower `BLOCK_MIN_DAYS` to `10` for `P0` only, and record the change in the report's caveats.

- [ ] **Step 5: Read the inventory**

```bash
cd studies/proxy_comparison && python -c "
import pandas as pd
print(pd.read_csv('outputs/P_13_block_inventory.csv').to_string())"
```

Record this table — Task 14 writes it into the report.

- [ ] **Step 6: Commit**

```bash
git add studies/proxy_comparison/proxy_comparison_study.py
git commit -m "feat: define instrument-era tiers and their contiguous blocks"
```

---

## Task 7: Notebook Step 5b — correlation structure

**Files:**
- Modify: `studies/proxy_comparison/proxy_comparison_study.py` (after Step 5a)

**Interfaces:**
- Consumes: `inventory_blocks`, `TIERS`, `TIER_LABEL` from Task 6; `pc.tier_frame`, `pc.tier_differences` from Task 3.
- Produces: `outputs/P_14_corr_levels.csv`, `outputs/P_15_corr_differences.csv`, `outputs/P_F06_correlations.png` and `.svg`.

- [ ] **Step 1: Add the Markdown header cell**

```python
# %% [markdown]
# ## Step 5b · Correlation structure, per tier
#
# The correlation matrix over the target and every source channel the tier
# carries, computed twice: on levels and on first differences.
#
# Correlations between two trending series are inflated by the shared trend and
# say little about a driver-response relationship. The differenced matrix is the
# honest one for a signal with drift. Both are reported so the gap between them
# is visible, which is the convention the sibling studies use.
#
# Differences are taken **inside each block** and concatenated. A difference
# across a block boundary would manufacture a step out of a gap.
```

- [ ] **Step 2: Add the code cell**

```python
# %%
CHANNEL_ORDER = ['inc_comp',
                 'tair_str', 'tair_gs', 'tair_era5',
                 'sr_str', 'sr_gs', 'sr_era5']

corr_levels, corr_differences, scan_frames = {}, {}, {}
for tier in TIERS:
    frame = pc.tier_frame(df, inventory_blocks, tier)
    cols = [c for c in CHANNEL_ORDER
            if c in frame.columns and frame[c].notna().any()]
    scan_frames[tier] = (frame, cols)

    corr_levels[tier] = frame[cols].corr()
    corr_differences[tier] = pc.tier_differences(
        df, inventory_blocks, tier, cols=cols).corr()

    print(f'\n{tier} ({TIER_LABEL[tier]}) — {len(cols)} channels, '
          f'{len(frame)} hourly slots')
    print('  against the target, levels:      ', ', '.join(
        f'{c} {corr_levels[tier].loc["inc_comp", c]:+.3f}'
        for c in cols if c != 'inc_comp'))
    print('  against the target, differences: ', ', '.join(
        f'{c} {corr_differences[tier].loc["inc_comp", c]:+.3f}'
        for c in cols if c != 'inc_comp'))

pd.concat(corr_levels, names=['tier']).to_csv(
    f'{OUTPUT_DIR}/P_14_corr_levels.csv')
pd.concat(corr_differences, names=['tier']).to_csv(
    f'{OUTPUT_DIR}/P_15_corr_differences.csv')
```

- [ ] **Step 3: Add the figure cell**

```python
# %%
fig, axes = plt.subplots(len(TIERS), 2, figsize=pc.figsize(11, 4.0 * len(TIERS)),
                         squeeze=False)
for row, tier in enumerate(TIERS):
    for col, (matrix, label) in enumerate(
            ((corr_levels[tier], 'levels'),
             (corr_differences[tier], 'first differences'))):
        ax = axes[row][col]
        sns.heatmap(matrix, ax=ax, vmin=-1.0, vmax=1.0, cmap='coolwarm',
                    annot=True, fmt='.2f', annot_kws={'size': 6},
                    cbar=(col == 1), square=True)
        ax.set_title(f'{tier} — {label}', fontsize=9)
        ax.tick_params(labelsize=6)
fig.suptitle('Correlation structure by tier, on levels and on first differences')
fig.tight_layout()
pc._finish(fig, save_path=OUTPUT_DIR, filename='P_F06_correlations')
plt.show()
```

- [ ] **Step 4: Run the notebook and verify**

```bash
cd studies/proxy_comparison
rm -f proxy_comparison_study.ipynb
jupytext --to ipynb proxy_comparison_study.py
jupyter nbconvert --to notebook --execute --inplace proxy_comparison_study.ipynb
ls -la outputs/P_14_corr_levels.csv outputs/P_15_corr_differences.csv \
       outputs/P_F06_correlations.png
```

Expected: all three files present and non-empty.

- [ ] **Step 5: Read the target row of each matrix**

```bash
cd studies/proxy_comparison && python -c "
import pandas as pd
for f in ('P_14_corr_levels.csv', 'P_15_corr_differences.csv'):
    m = pd.read_csv(f'outputs/{f}', index_col=[0, 1])
    print(f'--- {f} ---')
    print(m.xs('inc_comp', level=1).to_string())
"
```

This is the headline comparison against the pooled `P_08`. Record it.

- [ ] **Step 6: Commit**

```bash
git add studies/proxy_comparison/proxy_comparison_study.py
git commit -m "feat: report per-tier correlation structure on levels and differences"
```

---

## Task 8: Notebook Step 5c — the target relationship per tier

**Files:**
- Modify: `studies/proxy_comparison/proxy_comparison_study.py` (after Step 5b)

**Interfaces:**
- Consumes: `pc.target_relationship(..., blocks=, tier=)` from Task 4; `inventory_blocks` from Task 6.
- Produces: `outputs/P_16_target_relationship_tiers.csv`, `outputs/P_F07_relationship_tiers.png` and `.svg`.

- [ ] **Step 1: Add the Markdown header cell**

```python
# %% [markdown]
# ## Step 5c · Relationship with the inclination, per tier
#
# Each source against the one target, in levels and in first differences, inside
# each tier's blocks. The slope describes the residual response that survives
# the calibration. **It is not a candidate compensation coefficient** — the
# coefficient is the manufacturer's and is not re-estimated anywhere in this
# project.
#
# Relative humidity is reported here where it is present, but it never defines a
# tier: the tiers are cut on air temperature and solar radiation, which are the
# quantities this section analyses and the ones whose availability constrains
# the windows.
#
# Standard errors are Newey-West at 24 lags, because the residuals of an hourly
# thermal regression are strongly autocorrelated and ordinary standard errors
# would be far too small.
```

- [ ] **Step 2: Add the code cell**

```python
# %%
relationship_tiers = pd.concat(
    [pc.target_relationship(df, q, target=TARGET, hac_lags=24,
                            min_days=MIN_OVERLAP_DAYS, dt_hours=SAMPLING_HOURS,
                            blocks=inventory_blocks, tier=tier)
     for tier in TIERS for q in QUANTITIES],
    ignore_index=True)

display(relationship_tiers)
relationship_tiers.to_csv(f'{OUTPUT_DIR}/P_16_target_relationship_tiers.csv',
                          index=False)

pooled = relationship.set_index(['quantity', 'source', 'domain'])['r']
tiered = relationship_tiers[relationship_tiers['tier'] == 'P0'].set_index(
    ['quantity', 'source', 'domain'])['r']
contrast_tiers = pd.DataFrame({'r_pooled_span': pooled,
                               'r_tier_P0': tiered}).dropna()
contrast_tiers['change'] = (contrast_tiers['r_tier_P0']
                            - contrast_tiers['r_pooled_span']).round(4)
print('\nWhat the pooled span cost, tier P0 against the whole record:')
display(contrast_tiers)

assert TARGET in df.columns, 'target guard: the calibrated target is missing'
print(f'\nTarget guard passed: every relationship is measured against '
      f'{TARGET}, unmodified.')
```

- [ ] **Step 3: Add the figure cell**

```python
# %%
fig, axes = plt.subplots(1, len(TIERS), figsize=pc.figsize(11, 3.6),
                         squeeze=False, sharey=True)
for ax, tier in zip(axes[0], TIERS):
    subset = relationship_tiers[(relationship_tiers['tier'] == tier)
                                & (relationship_tiers['quantity'].isin(
                                    ['tair', 'sr']))]
    offsets = {'levels': -0.15, 'differences': +0.15}
    for domain, marker in (('levels', 'o'), ('differences', 's')):
        rows = subset[subset['domain'] == domain]
        if not len(rows):
            continue
        positions = np.arange(len(rows)) + offsets[domain]
        ax.errorbar(rows['slope_mdeg_per_unit'], positions,
                    xerr=rows['slope_se'], fmt=marker, capsize=3,
                    label=domain)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels([f'{q}·{pc.SOURCE_LABEL[s]}' for q, s
                            in zip(rows['quantity'], rows['source'])],
                           fontsize=7)
    ax.axvline(0.0, color='0.4', lw=0.8)
    ax.set_title(f'{tier} — {TIER_LABEL[tier]}', fontsize=9)
    ax.set_xlabel('slope [mdeg per unit]')
axes[0][0].legend(fontsize=7)
fig.suptitle('Slope against the calibrated inclination, by tier and domain')
fig.tight_layout()
pc._finish(fig, save_path=OUTPUT_DIR, filename='P_F07_relationship_tiers')
plt.show()
```

- [ ] **Step 4: Run and verify**

```bash
cd studies/proxy_comparison
rm -f proxy_comparison_study.ipynb
jupytext --to ipynb proxy_comparison_study.py
jupyter nbconvert --to notebook --execute --inplace proxy_comparison_study.ipynb
python -c "
import pandas as pd
t = pd.read_csv('outputs/P_16_target_relationship_tiers.csv')
assert set(t['tier']) == {'P0', 'P1', 'P2'}, set(t['tier'])
assert set(t['domain']) == {'levels', 'differences'}
print(t.to_string())"
```

Expected: rows for all three tiers, both domains. `Target guard passed` in the notebook output.

- [ ] **Step 5: Commit**

```bash
git add studies/proxy_comparison/proxy_comparison_study.py
git commit -m "feat: measure the target relationship inside each tier"
```

---

## Task 9: Notebook Step 5d — operator scans on two bands

**Files:**
- Modify: `studies/proxy_comparison/proxy_comparison_study.py` (after Step 5c; also **delete** the existing `P_09`/`P_F04` cells, which are the `pc.operator_by_source` call and the `imshow` grid that follows it)

**Interfaces:**
- Consumes: `pc.longest_block` from Task 3; `ip.operator_table`, `ip.plot_operator_heatmaps`.
- Produces: `outputs/P_17_operators_fullband.csv`, `outputs/P_18_operators_diurnal.csv`, `outputs/P_F08_operators.png` and `.svg`. Removes `outputs/P_09_operators.csv` and `outputs/P_F04_operators.png`.

- [ ] **Step 1: Delete the superseded cells**

In `proxy_comparison_study.py`, delete the cell containing `operators, scans = pc.operator_by_source(` together with its `operators.to_csv(...P_09...)` line, and delete the following `if scans:` cell that builds `P_F04_operators.png`. Keep the `pc.plot_target_relationship` cell that produces `P_F05` and the `slope_table` cell that produces `P_10`.

- [ ] **Step 2: Delete the stale artefacts**

```bash
cd studies/proxy_comparison
rm -f outputs/P_09_operators.csv outputs/P_F04_operators.png outputs/P_F04_operators.svg
```

- [ ] **Step 3: Add the Markdown header cell**

```python
# %% [markdown]
# ## Step 5d · Transport delay and thermal inertia, on two bands
#
# Before a driver is compared it is given the chance to act through the operator
# that suits it. Two are scanned jointly, because scanning either alone can
# attribute to one what belongs to the other: a **transport delay** shifts the
# driver in time without changing its shape, and a **thermal inertia** low-passes
# it through a single-pole filter. The instantaneous case stays in the grid and
# competes rather than being assumed away.
#
# The scan is run twice, on the series as they are and on the diurnal band alone
# with a one-week centred rolling mean removed. `inclination_prediction` found
# that this matters: on the full band, solar radiation's optimum pins at the last
# time constant in the grid, 168 hours, lifting explained variance from 0.229 to
# 0.662. A time constant of one week is not a thermal property of masonry; it is
# the width of filter needed to turn a daily radiation cycle into a seasonal
# envelope. The diurnal-band operators are the ones carried forward.
#
# Delays are bounded at twelve hours per `docs/raw-data-format.md` §7.5. Air
# temperature and solar radiation are external forcings, and beyond half a
# diurnal cycle a delay is indistinguishable from a lead.
#
# **The scan runs on each tier's longest single block, not on the union.** A
# delay is applied with a shift, a time constant with a recursive filter, and the
# band limit with a rolling mean; all three read across adjacent rows, so a union
# would let a filter draw values from the far side of a months-long gap.
```

- [ ] **Step 4: Add the code cell**

```python
# %%
operator_rows, operator_scans = [], {}
for tier in TIERS:
    block = pc.longest_block(inventory_blocks, tier)
    if block is None:
        continue
    window = df.loc[block['start']:block['end']]
    drivers = [c for c in CHANNEL_ORDER
               if c != TARGET and c in window.columns
               and window[c].notna().sum() > 100]

    for band, detrend in (('full', None), ('diurnal', DETREND_HOURS)):
        summary, scans = ip.operator_table(
            window, TARGET, drivers, delays=DELAYS, taus=TAUS,
            dt_hours=SAMPLING_HOURS, detrend_hours=detrend)
        summary = summary.reset_index()
        summary.insert(0, 'tier', tier)
        summary.insert(1, 'band', band)
        summary.insert(2, 'block', block['block'])
        summary.insert(3, 'block_days', block['days'])
        operator_rows.append(summary)
        if band == 'diurnal':
            operator_scans[tier] = scans

operators_all = pd.concat(operator_rows, ignore_index=True)
operators_fullband = operators_all[operators_all['band'] == 'full']
operators_diurnal = operators_all[operators_all['band'] == 'diurnal']

display(operators_fullband)
display(operators_diurnal)
operators_fullband.to_csv(f'{OUTPUT_DIR}/P_17_operators_fullband.csv',
                          index=False)
operators_diurnal.to_csv(f'{OUTPUT_DIR}/P_18_operators_diurnal.csv',
                         index=False)

boundary = operators_fullband[operators_fullband['tau_h'] >= max(TAUS)]
if len(boundary):
    print('\nFull-band optima sitting at the last time constant in the grid — '
          'these are fitting a component slower than the grid contains:')
    display(boundary[['tier', 'driver', 'tau_h', 'r2', 'r2_instantaneous']])
```

- [ ] **Step 5: Add the figure cell**

```python
# %%
for tier, scans in operator_scans.items():
    fig = ip.plot_operator_heatmaps(
        scans,
        title=f'Operator scan against the calibrated inclination, '
              f'diurnal band, tier {tier}',
        save_path=OUTPUT_DIR, filename=f'P_F08_operators_{tier}')
    plt.show()
```

- [ ] **Step 6: Run and verify**

```bash
cd studies/proxy_comparison
rm -f proxy_comparison_study.ipynb
jupytext --to ipynb proxy_comparison_study.py
jupyter nbconvert --to notebook --execute --inplace proxy_comparison_study.ipynb
python -c "
import pandas as pd
full = pd.read_csv('outputs/P_17_operators_fullband.csv')
diur = pd.read_csv('outputs/P_18_operators_diurnal.csv')
assert (full['band'] == 'full').all() and (diur['band'] == 'diurnal').all()
assert diur['delay_h'].max() <= 12, 'delay bound violated'
print(full.to_string()); print(); print(diur.to_string())"
ls outputs/P_F08_operators_*.png
```

Expected: both tables print, no delay above 12 hours, one figure per tier.

- [ ] **Step 7: Commit**

```bash
git add studies/proxy_comparison/proxy_comparison_study.py
git commit -m "feat: scan operators on the full and diurnal bands per tier"
```

---

## Task 10: Notebook Step 5e — the signed-delay control on radiation

**Files:**
- Modify: `studies/proxy_comparison/proxy_comparison_study.py` (after Step 5d)

**Interfaces:**
- Consumes: `pc.longest_block`, `ip.band_limit`, `tc.cross_correlation`, `tc.peak_lags`.
- Produces: `outputs/P_19_signed_delay_sr.csv`.

- [ ] **Step 1: Add the Markdown header cell**

```python
# %% [markdown]
# ## Step 5e · The signed-delay control on solar radiation
#
# The scan above is bounded at non-negative delays, because solar radiation is an
# external forcing and a forcing must precede the response it causes. That bound
# is right for *use* — a lead cannot be applied by a forecasting system, which
# would need the driver's future values at the moment the forecast is issued —
# but it also hides a diagnostic. If the optimum for a channel genuinely sits at
# a negative delay, the bound reports the boundary instead of the anomaly.
#
# Scanning signed delays makes the anomaly visible. A negative optimum for a
# radiation channel is not a physical lead; it is a signature of a defect in that
# channel. This is the third independent test of the on-structure pyranometer,
# after its non-zero night-time floor and its impossible solar-noon phase, and it
# is run on tier P0 where all three sources coexist — which removes the analysis
# window as an alternative explanation.
```

- [ ] **Step 2: Add the code cell**

```python
# %%
signed_rows = []
for tier in TIERS:
    block = pc.longest_block(inventory_blocks, tier)
    if block is None:
        continue
    window = df.loc[block['start']:block['end']].copy()
    drivers = [c for c in ('sr_str', 'sr_gs', 'sr_era5')
               if c in window.columns and window[c].notna().sum() > 100]
    if not drivers:
        continue

    banded = window[[TARGET] + drivers].apply(
        lambda s: ip.band_limit(s, DETREND_HOURS, SAMPLING_HOURS))
    ccf = tc.cross_correlation(banded, TARGET, drivers,
                               max_lag_steps=MAX_DELAY_H, differenced=False)
    peaks = tc.peak_lags(ccf, sampling_hours=SAMPLING_HOURS, match_sign=True)
    peaks.insert(0, 'tier', tier)
    peaks.insert(1, 'block', block['block'])
    signed_rows.append(peaks)

signed_delays = pd.concat(signed_rows, ignore_index=True)
display(signed_delays)
signed_delays.to_csv(f'{OUTPUT_DIR}/P_19_signed_delay_sr.csv', index=False)

proxies = signed_delays[signed_delays['driver'].isin(['sr_gs', 'sr_era5'])]
assert (proxies['peak_lag_hours'] >= 0).all(), (
    'forcing guard: a proxy radiation channel peaks at a negative delay, '
    'which points at the harmonisation or the clock correction rather than at '
    'the wall\n' + proxies.to_string())
print('\nForcing guard passed: both proxy radiation channels peak at a '
      'non-negative delay.')

on_structure = signed_delays[signed_delays['driver'] == 'sr_str']
if len(on_structure) and (on_structure['peak_lag_hours'] < 0).any():
    print('\nThe on-structure radiation channel peaks at a negative delay. '
          'A forcing cannot lead its own response, so this is a further '
          'symptom of the defect already established from the night-time '
          'floor and the solar-noon phase.')
```

Note that `tc` must be importable in the notebook. `pc_lib` already imports it, so add `import tc_lib as tc` to the notebook's import cell if it is not already there.

- [ ] **Step 3: Run and verify**

```bash
cd studies/proxy_comparison
rm -f proxy_comparison_study.ipynb
jupytext --to ipynb proxy_comparison_study.py
jupyter nbconvert --to notebook --execute --inplace proxy_comparison_study.ipynb
python -c "
import pandas as pd
s = pd.read_csv('outputs/P_19_signed_delay_sr.csv')
print(s.to_string())"
```

Expected: `Forcing guard passed`. If it fails, the harmonisation or the seasonal clock correction is at fault, not the wall — stop and report before continuing.

- [ ] **Step 4: Commit**

```bash
git add studies/proxy_comparison/proxy_comparison_study.py
git commit -m "feat: add the signed-delay control on the radiation channels"
```

---

## Task 11: Notebook Step 5f — what radiation adds over temperature

**Files:**
- Modify: `studies/proxy_comparison/proxy_comparison_study.py` (after Step 5e)

**Interfaces:**
- Consumes: `ip.vif_table`, `tc.multivariate_fit`, `pc.tier_frame`, `pc.tier_differences`.
- Produces: `outputs/P_20_radiation_contribution.csv`.

- [ ] **Step 1: Add the Markdown header cell**

```python
# %% [markdown]
# ## Step 5f · What solar radiation adds once temperature is present
#
# Radiation heats the air, so air temperature and solar radiation are strongly
# coupled and a marginal correlation between radiation and the target largely
# re-measures the temperature relationship. Two steps separate them.
#
# The **variance inflation factor** quantifies how far the two duplicate one
# another before any coefficient is read for interpretation. The **joint fit**
# then enters both together and reports the increment in explained variance over
# the temperature-only fit, with Newey-West standard errors at 24 lags.
#
# This is the question the imputation study inherits: whether a radiation channel
# is worth carrying as a regressor at all once temperature is available.
```

- [ ] **Step 2: Add the code cell**

```python
# %%
contribution_rows = []
for tier in TIERS:
    for domain in ('levels', 'differences'):
        frame = (pc.tier_frame(df, inventory_blocks, tier) if domain == 'levels'
                 else pc.tier_differences(df, inventory_blocks, tier))
        for source in pc.SOURCES:
            tair_col, sr_col = f'tair_{source}', f'sr_{source}'
            if not {tair_col, sr_col, TARGET}.issubset(frame.columns):
                continue
            data = frame[[TARGET, tair_col, sr_col]].dropna()
            if len(data) < MIN_OVERLAP_DAYS * 24:
                continue

            vif = ip.vif_table(data, [tair_col, sr_col])
            alone = tc.multivariate_fit(data, TARGET, [tair_col], hac_lags=24)
            joint = tc.multivariate_fit(data, TARGET, [tair_col, sr_col],
                                        hac_lags=24)

            contribution_rows.append({
                'tier': tier,
                'domain': domain,
                'source': source,
                'n': int(len(data)),
                'VIF': float(vif.loc[tair_col, 'VIF']),
                'r_tair_sr': float(data[tair_col].corr(data[sr_col])),
                'r2_tair_only': round(float(alone.rsquared), 4),
                'r2_joint': round(float(joint.rsquared), 4),
                'delta_r2': round(float(joint.rsquared - alone.rsquared), 4),
                'slope_tair_joint': round(float(joint.params[tair_col]), 4),
                'slope_sr_joint': round(float(joint.params[sr_col]), 5),
                'se_sr_joint': round(float(joint.bse[sr_col]), 5),
                't_sr_joint': round(float(joint.tvalues[sr_col]), 2),
            })

contribution = pd.DataFrame(contribution_rows)
display(contribution)
contribution.to_csv(f'{OUTPUT_DIR}/P_20_radiation_contribution.csv', index=False)

print('\nWhere radiation earns its place — the joint fit gains at least one '
      'point of explained variance and the radiation term clears |t| = 2:')
earns = contribution[(contribution['delta_r2'] >= 0.01)
                     & (contribution['t_sr_joint'].abs() >= 2.0)]
display(earns if len(earns) else 'nowhere on these windows')
```

- [ ] **Step 3: Run and verify**

```bash
cd studies/proxy_comparison
rm -f proxy_comparison_study.ipynb
jupytext --to ipynb proxy_comparison_study.py
jupyter nbconvert --to notebook --execute --inplace proxy_comparison_study.ipynb
python -c "
import pandas as pd
c = pd.read_csv('outputs/P_20_radiation_contribution.csv')
assert len(c), 'no contribution rows produced'
print(c.to_string())"
```

- [ ] **Step 4: Commit**

```bash
git add studies/proxy_comparison/proxy_comparison_study.py
git commit -m "feat: measure what solar radiation adds beyond air temperature"
```

---

## Task 12: Notebook Step 6 — radiation availability, verdicts and the feature specification

**Files:**
- Modify: `studies/proxy_comparison/proxy_comparison_study.py` (the existing Step 6 availability cell, verdict cell, and feature-specification cell)

**Interfaces:**
- Consumes: `operators_diurnal` from Task 9, `relationship_tiers` from Task 8, `signed_delays` from Task 10.
- Produces: `outputs/P_22_sr_availability.csv`, and updated `outputs/P_11_verdicts.csv` and `outputs/P_12_feature_specification.csv`.

- [ ] **Step 1: Extend the availability cell**

Find the cell that builds the `availability` frame from `tair_{s}`. Replace the list comprehension that builds it with one covering both quantities:

```python
availability = pd.DataFrame([
    {'quantity': q,
     'source': s,
     'coverage_of_span_%': round(
         100.0 * on_span[f'{q}_{s}'].notna().mean(), 1),
     'present_during_target_gaps_%': round(
         100.0 * on_span.loc[absent, f'{q}_{s}'].notna().mean(), 1)}
    for q in ('tair', 'sr') for s in pc.SOURCES
    if f'{q}_{s}' in df.columns]).set_index(['quantity', 'source'])
display(availability)
availability.to_csv(f'{OUTPUT_DIR}/P_22_sr_availability.csv')
print(f'Measured over {len(span)} hourly slots of the archive span, '
      f'of which {int(absent.sum())} are missing the inclinometer.')
```

Every later reference to `availability.loc[s, ...]` must become `availability.loc[(q, s), ...]`. There are two, in the verdict cell and in the feature-specification cell; both are rewritten below.

- [ ] **Step 2: Rewrite the verdict cell**

Replace the `verdict_rows` list with:

```python
p0 = relationship_tiers[relationship_tiers['tier'] == 'P0']
tair_agree = agreement[(agreement['quantity'] == 'tair')
                       & (agreement['reference'] == 'str')]
sr_agree = agreement[(agreement['quantity'] == 'sr')]
diurnal_ops = operators_diurnal

verdict_rows = [
    {'question': 'do the sources agree in absolute terms?',
     'answer': '; '.join(
         f'{pc.SOURCE_LABEL[r["compared"]]}: bias {r["bias"]:+.2f} degC, '
         f'MAE {r["MAE"]:.2f}, limits of agreement '
         f'{r["loa_lower"]:+.1f} to {r["loa_upper"]:+.1f}'
         for _, r in tair_agree.iterrows())},
    {'question': 'do the radiation channels agree?',
     'answer': '; '.join(
         f'{r["reference"]}-{r["compared"]}: r {r["r"]:.3f}, '
         f'MAE {r["MAE"]:.1f} W/m2 over {r["overlap_days"]:.0f} days'
         for _, r in sr_agree.iterrows())},
    {'question': 'what does the daylight mask change for radiation?',
     'answer': '; '.join(
         f'{r["reference"]}-{r["compared"]}: r {r["r"]:.3f} in daylight'
         for _, r in daylight_agreement.iterrows())},
    {'question': 'how much agreement survives differencing?',
     'answer': '; '.join(
         f'{pc.SOURCE_LABEL[r["compared"]]}: r {r["r"]:.3f} in differences '
         f'against {level_r.get(("tair", r["reference"], r["compared"])):.3f} '
         f'in levels'
         for _, r in differences[(differences['quantity'] == 'tair')
                                 & (differences['reference'] == 'str')].iterrows())},
    {'question': 'what did the pooled span cost?',
     'answer': '; '.join(
         f'{q}·{s}·{d}: {row["r_pooled_span"]:+.3f} pooled against '
         f'{row["r_tier_P0"]:+.3f} in tier P0'
         for (q, s, d), row in contrast_tiers.iterrows()
         if q in ('tair', 'sr') and d == 'differences')},
    {'question': 'relationship with the target in tier P0, levels',
     'answer': '; '.join(
         f'{r["quantity"]}·{pc.SOURCE_LABEL[r["source"]]}: r {r["r"]:+.3f}, '
         f'slope {r["slope_mdeg_per_unit"]:+.3f}'
         for _, r in p0[(p0['domain'] == 'levels')
                        & (p0['quantity'].isin(['tair', 'sr']))].iterrows())},
    {'question': 'relationship with the target in tier P0, differences',
     'answer': '; '.join(
         f'{r["quantity"]}·{pc.SOURCE_LABEL[r["source"]]}: r {r["r"]:+.3f}, '
         f'slope {r["slope_mdeg_per_unit"]:+.3f}'
         for _, r in p0[(p0['domain'] == 'differences')
                        & (p0['quantity'].isin(['tair', 'sr']))].iterrows())},
    {'question': 'transport delay and inertia, diurnal band',
     'answer': '; '.join(
         f'{r["tier"]}·{r["driver"]}: delay {r["delay_h"]:g} h, '
         f'tau {r["tau_h"]:g} h, R2 {r["r2"]:.3f}'
         for _, r in diurnal_ops.iterrows())},
    {'question': 'does any radiation channel lead the response?',
     'answer': '; '.join(
         f'{r["tier"]}·{r["driver"]}: {r["peak_lag_hours"]:+.0f} h'
         for _, r in signed_delays.iterrows())},
    {'question': 'does radiation add anything beyond temperature?',
     'answer': '; '.join(
         f'{r["tier"]}·{r["domain"]}·{r["source"]}: delta R2 '
         f'{r["delta_r2"]:+.3f}, t {r["t_sr_joint"]:+.1f}'
         for _, r in contribution.iterrows())},
    {'question': 'availability when the target is missing',
     'answer': '; '.join(
         f'{q}·{pc.SOURCE_LABEL[s]}: '
         f'{availability.loc[(q, s), "present_during_target_gaps_%"]:.1f} %'
         for q, s in availability.index)},
]
verdicts = pd.DataFrame(verdict_rows).set_index('question')
display(verdicts)
verdicts.to_csv(f'{OUTPUT_DIR}/P_11_verdicts.csv')
```

- [ ] **Step 3: Rewrite the feature-specification cell**

```python
# %%
spec_rows = []
for _, row in operators_diurnal.iterrows():
    driver = row['driver']
    quantity, source = driver.rsplit('_', 1)
    if quantity not in ('tair', 'sr'):
        continue
    if (quantity, source) not in availability.index:
        continue
    rel = relationship_tiers[(relationship_tiers['tier'] == row['tier'])
                             & (relationship_tiers['quantity'] == quantity)
                             & (relationship_tiers['source'] == source)
                             & (relationship_tiers['domain'] == 'differences')]
    gaps = float(availability.loc[(quantity, source),
                                  'present_during_target_gaps_%'])
    spec_rows.append({
        'channel': driver,
        'tier': row['tier'],
        'source': pc.SOURCE_LABEL[source],
        'delay_h': row['delay_h'],
        'tau_h': row['tau_h'],
        'r2_diurnal_band': row['r2'],
        'r_differences': float(rel['r'].iloc[0]) if len(rel) else np.nan,
        'available_during_gaps_%': gaps,
        'usable_for_reconstruction': bool(gaps > 70.0),
    })
specification = pd.DataFrame(spec_rows).set_index(['channel', 'tier'])
display(specification)
specification.to_csv(f'{OUTPUT_DIR}/P_12_feature_specification.csv')

print('\nChannels usable for reconstruction (present when the target is not):')
for (channel, tier), row in specification.iterrows():
    if row['usable_for_reconstruction']:
        print(f'  {channel} ({tier}): delay {row["delay_h"]:g} h, '
              f'tau {row["tau_h"]:g} h, '
              f'{row["available_during_gaps_%"]:.1f} % available')
```

- [ ] **Step 4: Run the whole notebook end to end**

```bash
cd studies/proxy_comparison
rm -f proxy_comparison_study.ipynb
jupytext --to ipynb proxy_comparison_study.py
jupyter nbconvert --to notebook --execute --inplace proxy_comparison_study.ipynb
```

Expected: no exception, and every guard message present: `Clock guard passed`, `Daylight guard passed`, `Tier guard passed`, `Era guard passed`, `Target guard passed`, `Forcing guard passed`.

- [ ] **Step 5: Confirm the full artefact set**

```bash
cd studies/proxy_comparison
for f in P_13_block_inventory P_14_corr_levels P_15_corr_differences \
         P_16_target_relationship_tiers P_17_operators_fullband \
         P_18_operators_diurnal P_19_signed_delay_sr \
         P_20_radiation_contribution P_21_sr_daylight_agreement \
         P_22_sr_availability; do
  test -s "outputs/$f.csv" && echo "OK  $f" || echo "MISSING $f"
done
ls outputs/P_F06_correlations.png outputs/P_F07_relationship_tiers.png \
   outputs/P_F08_operators_*.png
test -e outputs/P_09_operators.csv && echo "STALE P_09 still present" || echo "OK  P_09 retired"
```

Expected: ten `OK` lines, the figures listed, and `OK  P_09 retired`.

- [ ] **Step 6: Confirm the data dictionary carries the Site block**

```bash
cd studies/proxy_comparison
grep -A6 '## Site' ../../docs/proxy-data-dictionary.md
```

Expected: the latitude, longitude and `auxiliary/oiko.py` provenance rows from Task 1.

- [ ] **Step 7: Commit**

```bash
git add studies/proxy_comparison/proxy_comparison_study.py docs/proxy-data-dictionary.md
git commit -m "feat: extend verdicts and the feature specification to radiation"
```

---

## Task 13: Report sections 3 and 4

**Files:**
- Modify: `studies/proxy_comparison/report/proxy_comparison_report.tex:199-378`

**Interfaces:**
- Consumes: `outputs/P_21_sr_daylight_agreement.csv`, `outputs/P_21b_sr_daylight_diurnal.csv`, `outputs/P_06_diurnal_comparison.csv`.

- [ ] **Step 1: Read the numbers**

```bash
cd studies/proxy_comparison && python -c "
import pandas as pd
print('--- daylight agreement ---')
print(pd.read_csv('outputs/P_21_sr_daylight_agreement.csv').to_string())
print('--- daylight diurnal ---')
print(pd.read_csv('outputs/P_21b_sr_daylight_diurnal.csv').to_string())
print('--- all-hours diurnal ---')
print(pd.read_csv('outputs/P_06_diurnal_comparison.csv').to_string())"
```

Every number written below comes from this output. Do not carry a number over from the current report text without checking it against these tables.

- [ ] **Step 2: Add the daylight block to the Section 3 agreement table**

In the table beginning `\multicolumn{8}{l}{\itshape solar radiation [W/m\textsuperscript{2}]}`, keep the three existing all-hours rows and add beneath them, before `\bottomrule`:

```latex
\addlinespace
\multicolumn{8}{l}{\itshape solar radiation, daylight hours only} \\
& str--gs   & DAYS & BIAS & MAE & SLOPE & R & LOA \\
& str--era5 & DAYS & BIAS & MAE & SLOPE & R & LOA \\
& gs--era5  & DAYS & BIAS & MAE & SLOPE & R & LOA \\
```

Substitute each placeholder token with the corresponding value from the `agreement` rows of `P_21_sr_daylight_agreement.csv`, formatted like the rows above them: bias and MAE to one decimal, slope to three, `$r$` to three, limits of agreement as `$-505$ to $+370$`.

- [ ] **Step 3: Add the explanatory paragraph after that table**

```latex
\panel{Half of a radiation record is night, and it flatters every statistic}{%
Roughly half of every solar-radiation series is a structural night-time zero.
Those zeros are real measurements and are not removed anywhere in this study,
but a statistic computed over the whole day is computed largely on a block of
samples on which all three sources trivially agree.

\medskip
Restricting to the hours the sun is above the horizon --- a geometric mask taken
from the site coordinates, independent of all three sources --- changes the
picture in the direction the objection predicts. [Write one sentence stating the
direction and size of the change, from the numbers read in Step 1.]

\medskip
The mask is geometry, not radiation. It takes no account of cloud, and none of
the local horizon formed by the terrain around the site, so it marks when the
sun is up rather than when the wall was lit.}
```

Replace the bracketed sentence with the actual finding. If the daylight correlations are lower than the all-hours ones, say so and say by how much; if they are higher, say that instead. Do not keep the bracket.

- [ ] **Step 4: Add the radiation rows to the Section 4 diurnal table**

After the `relative humidity` block and before `\bottomrule`:

```latex
\addlinespace
\multicolumn{5}{l}{\itshape solar radiation [W/m\textsuperscript{2}], daylight hours} \\
& on-structure   & AMP & 1.00 & PEAK \\
& ground station & AMP & RATIO & PEAK \\
& ERA5           & AMP & RATIO & PEAK \\
```

Fill from `P_21b_sr_daylight_diurnal.csv`. Then add this sentence immediately after the table:

```latex
The radiation rows are reported over the daylight hours and carry no trough
hour. Over the full day the trough of a radiation cycle is an arbitrary point
inside the night --- the all-hours computation returns hours 23, 1 and 2 for the
three sources --- and a number that arbitrary does not belong in a comparison.
```

- [ ] **Step 5: Verify the report still has matched delimiters**

```bash
cd studies/proxy_comparison/report
python - <<'PY'
text = open('proxy_comparison_report.tex').read()
assert text.count('\\begin{tabular}') == text.count('\\end{tabular}'), 'tabular mismatch'
assert text.count('\\begin{center}') == text.count('\\end{center}'), 'center mismatch'
assert text.count('{') == text.count('}'), 'brace mismatch'
assert '[Write one sentence' not in text, 'placeholder left in the report'
assert 'DAYS' not in text and 'RATIO' not in text, 'token left unsubstituted'
print('report structure OK')
PY
```

Expected: `report structure OK`

- [ ] **Step 6: Commit**

```bash
git add studies/proxy_comparison/report/proxy_comparison_report.tex
git commit -m "docs: report solar radiation over daylight hours in sections 3 and 4"
```

---

## Task 14: Report section 5

**Files:**
- Modify: `studies/proxy_comparison/report/proxy_comparison_report.tex:380-463` (replace the whole of `\section{Relationship with the inclination}`)

**Interfaces:**
- Consumes: `P_13`, `P_14`, `P_15`, `P_16`, `P_17`, `P_18`, `P_19`, `P_20`, `P_F06`, `P_F07`, `P_F08_operators_*`.

- [ ] **Step 1: Read every number the section needs**

```bash
cd studies/proxy_comparison && python -c "
import pandas as pd
for name in ('P_13_block_inventory', 'P_16_target_relationship_tiers',
             'P_17_operators_fullband', 'P_18_operators_diurnal',
             'P_19_signed_delay_sr', 'P_20_radiation_contribution'):
    print(f'=== {name} ===')
    print(pd.read_csv(f'outputs/{name}.csv').to_string())
    print()
for name in ('P_14_corr_levels', 'P_15_corr_differences'):
    m = pd.read_csv(f'outputs/{name}.csv', index_col=[0, 1])
    print(f'=== {name}, target row ===')
    print(m.xs('inc_comp', level=1).to_string())
    print()"
```

- [ ] **Step 2: Replace the section**

Delete everything from `\section{Relationship with the inclination}` up to but not including `% ===` before `\section{Verdict}`, and write the replacement with these seven subsections in this order. Each table is filled from the artefact named beside it; no number is invented and none is carried over from the old text.

1. `\subsection{The windows this comparison can use}` — the block inventory table from `P_13`, one row per block with tier, era, start, days and coverage. Prose states that no statistic below is computed on the pooled span, that no window crosses 21 February 2025, and that the on-structure pyranometer beginning at the era boundary is what forces this shape. State the P0/P1 overlap and that the two are therefore not independent samples.
2. `\subsection{Correlation structure}` — the target row of `P_14` and `P_15` side by side, per tier, and figure `P_F06`. Carry the standing reading rule: levels are trend-inflated, differences are the honest matrix for a signal with drift.
3. `\subsection{Relationship with the target, by tier}` — the main table from `P_16`: quantity, source, domain, *r*, slope, Newey-West standard error, *R²*, per tier. Repeat the standing sentence that the slope is not a candidate compensation coefficient.
4. `\subsection{Transport delay and thermal inertia, on two bands}` — the full-band table from `P_17` and the diurnal-band table from `P_18`, with the block each scan ran on and its length. Explain the boundary optimum if the full-band scan produced one, and state that the diurnal-band operators are the ones carried forward. Include figure `P_F08` for each tier that produced one.
5. `\subsection{Does any radiation channel lead the response?}` — the table from `P_19`. State the rule that a forcing must precede its response, so a negative optimum is a defect signature rather than a physical lead, and connect the result to the night-time floor and the solar-noon phase already established in Section~\ref{sec:clock}.
6. `\subsection{What radiation adds once temperature is present}` — the table from `P_20`: VIF, the correlation between the two drivers, *R²* alone and jointly, the increment, and the radiation term's *t*. Conclude explicitly whether a radiation channel earns a place in the feature set.
7. `\subsection{What the pooled span cost}` — the single contrast against `P_08`. Report the pooled figures the previous version of this report carried and the tier figures beside them, and state that the earlier conclusion of a uniformly weak differenced relationship was a property of the analysis window.

Use the existing `\panel{...}{...}` macro for the one or two findings that carry the section, matching how Sections 3 and 4 use it. Every table uses `\toprule`, `\midrule`, `\bottomrule` and the `R{...}` column type already used throughout the file.

- [ ] **Step 3: Verify no placeholder or stale reference survives**

```bash
cd studies/proxy_comparison/report
python - <<'PY'
import re
text = open('proxy_comparison_report.tex').read()
for token in ('P_F04', 'P\\_09', 'TODO', 'TBD', '[Write', 'XXX'):
    assert token not in text, f'stale token {token!r} in the report'
labels = set(re.findall(r'\\label\{([^}]*)\}', text))
refs = set(re.findall(r'\\ref\{([^}]*)\}', text))
missing = refs - labels
assert not missing, f'undefined references: {missing}'
assert text.count('\\begin{tabular}') == text.count('\\end{tabular}')
assert text.count('{') == text.count('}')
print('section 5 structure OK')
PY
```

- [ ] **Step 4: Verify every figure referenced exists**

```bash
cd studies/proxy_comparison/report
python - <<'PY'
import os, re
text = open('proxy_comparison_report.tex').read()
for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}', text):
    for ext in ('.png', '.pdf', '.svg', ''):
        if os.path.exists(os.path.join('..', 'outputs', name + ext)):
            break
    else:
        raise AssertionError(f'figure not found in outputs/: {name}')
print('every referenced figure exists')
PY
```

- [ ] **Step 5: Commit**

```bash
git add studies/proxy_comparison/report/proxy_comparison_report.tex
git commit -m "docs: rewrite the target-relationship section around tiers and blocks"
```

---

## Task 15: Report sections 6 and 7, the inventory, the abstract and the README

**Files:**
- Modify: `studies/proxy_comparison/report/proxy_comparison_report.tex:465-604` and `:24-66`
- Modify: `studies/proxy_comparison/README.md`

**Interfaces:**
- Consumes: `P_11`, `P_12`, `P_22`, and the findings established in Tasks 13 and 14.

- [ ] **Step 1: Read the verdict artefacts**

```bash
cd studies/proxy_comparison && python -c "
import pandas as pd
pd.set_option('display.width', 200); pd.set_option('display.max_colwidth', 400)
print(pd.read_csv('outputs/P_22_sr_availability.csv').to_string()); print()
print(pd.read_csv('outputs/P_12_feature_specification.csv').to_string()); print()
print(pd.read_csv('outputs/P_11_verdicts.csv').to_string())"
```

- [ ] **Step 2: Extend the three Section 6 tables**

The availability table gains its radiation rows from `P_22`, grouped under an italic sub-heading in the same style Section 3 uses for its quantities. The which-source-for-what table gains a row for radiation as a regressor, stating the recommendation the `P_20` result supports. The feature-specification table gains its radiation rows and a `tier` column, filled from `P_12`.

- [ ] **Step 3: Add the three new caveats**

Append to `\section{Caveats}`, in the file's existing `\textbf{...}` paragraph style:

```latex
\textbf{The legacy tier measures a different instrument.} Tier P2 lies before the
21 February 2025 reinstallation. The package installed on that date shares no
baseline with the one it replaced, so a result obtained on P2 describes the
earlier instrument and does not transfer across the boundary. No window in this
section spans it.

\textbf{The daylight mask is geometry, not radiation.} It marks the hours the sun
stands above the horizon at the site's coordinates. It takes no account of cloud,
and none of the local horizon formed by the terrain around the site, so it says
when the sun was up rather than when the wall was lit.

\textbf{Tiers P0 and P1 overlap.} P1 contains every hour of P0 and more.
Agreement between the two is therefore weaker evidence than agreement between
independent samples, and where they disagree the disagreement is the informative
part.
```

If the Task 6 target verification found the two constructions of `inc_comp` to differ, add a fourth caveat stating the magnitude of the difference and that part of the gap against the sibling study is attributable to it rather than to the window. If they were identical, add nothing.

- [ ] **Step 4: Update the artefact inventory**

Extend the inventory table with:

```latex
\texttt{P\_13} & Block inventory: the windows each tier can use \\
\texttt{P\_14}, \texttt{P\_15} & Correlation structure per tier, levels and differences \\
\texttt{P\_16} & Relationship with the target, by tier and domain \\
\texttt{P\_17}, \texttt{P\_18} & Operator scans, full band and diurnal band \\
\texttt{P\_19} & Signed-delay control on the radiation channels \\
\texttt{P\_20} & What radiation adds once temperature is present \\
\texttt{P\_21} & Radiation agreement over the daylight hours \\
\texttt{P\_22} & Radiation availability, including during target gaps \\
\addlinespace
\texttt{P\_F06} & Correlation heatmaps by tier \\
\texttt{P\_F07} & Slope against the target by tier and domain \\
\texttt{P\_F08} & Operator scans, diurnal band, one figure per tier \\
```

Delete the `\texttt{P\_F04}` row and change the `\texttt{P\_08} -- \texttt{P\_10}` row so it no longer claims `P_09` exists.

- [ ] **Step 5: Rewrite the abstract**

The abstract currently quotes pooled-span figures in its final paragraph — the level slopes $-4.90$, $-4.68$ and $-5.03$. Replace that paragraph with the tier figures from `P_16`, and add one sentence reporting the radiation result and one reporting what the pooled span cost. Keep every paragraph that Tasks 13 and 14 did not invalidate: the availability result, the temperature agreement, the diurnal-amplitude reversal and the triangulated radiation defect are all unchanged.

- [ ] **Step 6: Update the README findings list**

In `studies/proxy_comparison/README.md`, the "What it found" list ends with a bullet quoting the pooled level slopes. Replace it with the tier result, and add two bullets: one for the radiation relationship with the target, one stating that the pooled span understated the differenced relationship and that all target statistics are now computed inside blocks. Add a `tests/` row to the Contents table.

- [ ] **Step 7: Final static verification of the whole report**

```bash
cd studies/proxy_comparison/report
python - <<'PY'
import os, re
text = open('proxy_comparison_report.tex').read()
for token in ('TODO', 'TBD', 'XXX', '[Write', 'P_F04', 'DAYS', 'RATIO', 'AMP', 'PEAK'):
    assert token not in text, f'stale token {token!r}'
labels = set(re.findall(r'\\label\{([^}]*)\}', text))
refs = set(re.findall(r'\\ref\{([^}]*)\}', text))
assert not refs - labels, f'undefined references: {refs - labels}'
for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}', text):
    assert any(os.path.exists(os.path.join('..', 'outputs', name + e))
               for e in ('.png', '.pdf', '.svg', '')), f'missing figure {name}'
for env in ('tabular', 'tabularx', 'center', 'figure', 'abstract', 'document'):
    assert text.count(f'\\begin{{{env}}}') == text.count(f'\\end{{{env}}}'), env
assert text.count('{') == text.count('}')
print('full report verification passed')
PY
```

Expected: `full report verification passed`

- [ ] **Step 8: Re-run every test suite**

```bash
cd studies/proxy_comparison
python tests/test_solar.py && \
python tests/test_tiers.py && \
python tests/test_target_relationship.py
```

Expected: all three suites pass.

- [ ] **Step 9: Commit**

```bash
git add studies/proxy_comparison/report/proxy_comparison_report.tex \
        studies/proxy_comparison/README.md
git commit -m "docs: extend the verdict, caveats and abstract to the tiered analysis"
```

- [ ] **Step 10: Hand the build to the user**

Agents do not compile. Report to the user:

> The report source is updated and passes static verification. Build it with
> `cd studies/proxy_comparison/report && pdflatex proxy_comparison_report.tex`,
> run twice for the table of contents.

---

## Self-Review

**Spec coverage.** Section 3 of the spec (coordinate provenance) is Task 1. Section 4 (tiers) is Tasks 3 and 6. Section 4.1 (humidity does not define a tier, sources absent from P1/P2) is handled by Task 8's Markdown cell and by the `if not {...}.issubset(frame.columns)` guards in Tasks 7, 11 and 12. Section 5 (correlation structure) is Task 7. Section 6 (target relationship per tier) is Tasks 4 and 8. Section 7 (operators, two bands, longest block) is Task 9. Section 8 (signed delay) is Task 10. Section 9 (VIF and partial fit) is Task 11. Section 10 (availability and daylight) is Tasks 5 and 12. Section 11 (report) is Tasks 13, 14 and 15. Section 12 (notebook structure) is distributed across Tasks 5–12 in the spec's order. Section 13 (new pc_lib code) is Tasks 1–4. Section 15 (verification) is the guards in Tasks 6, 10 and 12 and the static checks in Tasks 13–15.

**One spec item deliberately not implemented:** the spec's section 14 lists the Oikolab API key as out of scope, and no task touches it.

**Placeholder scan.** The only bracketed instructions are in Task 13 Step 3 and Task 14 Step 2, where the prose depends on numbers that do not exist until the notebook runs. Both are accompanied by an explicit verification step that fails if the bracket survives.

**Type consistency.** `pc.tier_blocks` returns the columns in `pc.BLOCK_COLUMNS`, and Tasks 6–12 index it by `tier`, `era`, `block`, `start`, `end`, `days`, `coverage_%` only. `pc.longest_block` returns a row of that same frame and is indexed by `start`, `end`, `block`, `days` in Tasks 9 and 10. `pc.target_relationship` gains `tier` as its second column only when `blocks` is passed, which Task 8 relies on and Task 4's first test pins. `operators_diurnal` carries `tier`, `band`, `block`, `block_days`, `driver`, `delay_h`, `tau_h`, `r`, `r2`, `r2_instantaneous`, `gain_from_operator` — the `ip.operator_table` columns plus the four inserted in Task 9 — and Task 12 reads exactly those names.
