"""
Module: ud_lib.py

Support library for the unified-dataset and imputation study
(``studies/unified_dataset/``).

This study does two things the earlier ones deliberately did not. First, it
assembles **one dataset covering the whole archive** — both column eras, all
three legacy stations and the current-era package — cleaned, compensated, and
carrying enough provenance that no later analysis has to guess where a value
came from. Second, it asks what can be done about the holes in it: which gaps
are fillable, by what method, to what accuracy, with what uncertainty, and how
long a usable stretch that buys.

Three facts govern everything below and are not negotiable.

**The two eras are two instruments.** The package installed on 2025-02-21
replaced the legacy network rather than augmenting it, and its inclinometer
shares no baseline with legacy block b2. ``docs/raw-data-format.md`` Section 3.3
forbids concatenating the two ``inc`` series without an explicit offset
treatment. The unified table therefore keeps them separate and labelled, and
offers the joined series only as a clearly named derived column.

**The absolute level means nothing.** Only changes carry structural information
(Section 7.1 of the same document). Every level in this dataset is referred to
an arbitrary anchor, and the first difference is the only quantity invariant to
all of them.

**A filled value is not a measurement.** Every row carries a provenance flag,
and the flag survives into every product built from it.

Loading, sentinel handling and compensation come from ``tc_lib`` and ``lc_lib``;
the imputation benchmark generalises the one in ``lp_lib``. Only what is new to
this study is implemented here.

Nothing in this module writes to the raw archive.
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

_HERE = os.path.dirname(os.path.abspath(__file__))
for _sib in ('thermal_compensation', 'thermal_compensation_legacy',
             'inclination_prediction', 'inclination_prediction_legacy'):
    _p = os.path.abspath(os.path.join(_HERE, '..', _sib))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tc_lib as tc                                             # noqa: E402
import lc_lib as lc                                             # noqa: E402
import ip_lib as ip                                             # noqa: E402
import lp_lib as lp                                             # noqa: E402

from tc_lib import set_context, figsize, _finish                # noqa: E402,F401


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

#: First day of the archive.
ARCHIVE_START = '2018-07-26'

#: Last day of the legacy 14-column era, inclusive.
LEGACY_END = '2025-02-20'

#: First day of the current 20-column era.
CURRENT_START = '2025-02-21'

#: Legacy stations, in block order. The current era instruments ``st02`` only.
STATIONS = ('st01', 'st02', 'st03')

#: The station this project analyses.
TARGET_STATION = 'st02'

#: Channels every era carries, for both stations and eras.
COMMON_CHANNELS = ('batt', 'tair', 'rh', 'inc')

#: Channels only the current-era package carries.
CURRENT_ONLY_CHANNELS = ('sr', 'twall')

#: Instrument labels. The inclinometer baseline is constant within a label and
#: meaningless across labels.
INSTRUMENT = {'legacy': 'legacy_block', 'current': 'current_package'}

#: Analysis grid. Every external proxy the pipeline aligns against is hourly.
ANALYSIS_FREQ = tc.ANALYSIS_FREQ

#: Compensation coefficient, in mdeg · °C⁻¹ · 10⁻³. The documented value, which
#: has never been calibrated for these instruments; carried here for continuity
#: with every earlier study, and its influence is quantified rather than
#: trusted.
COMP_COEFF = tc.DOCUMENTED_COEFF

#: Days per year, for expressing a span in annual cycles.
YEAR_DAYS = lp.YEAR_DAYS

#: Cycles of a component a window must contain before that component may be
#: fitted from it. Two is the rule the legacy study applied when choosing a
#: decomposition window, and it is applied here to the harmonic filler so that a
#: short window cannot be asked to identify an annual term.
MIN_CYCLES_TO_FIT = 2.0


# ──────────────────────────────────────────────────────────────────────
# Loading the whole archive, every station, both eras
# ──────────────────────────────────────────────────────────────────────

def load_all_stations(archive_dir, cache_dir, start=ARCHIVE_START, end=None,
                      freq=ANALYSIS_FREQ, stations=STATIONS, verbose=True):
    """
    Load every station of both eras onto one common grid.

    The two eras are parsed by the loaders written for them —
    :func:`lc_lib.load_legacy` for the 14-column files and
    :func:`tc_lib.load_archive` for the 20-column ones — rather than by a third
    parser written here. Those two have been exercised against the archive's
    documented defects, and a unified reader would have to reproduce every one
    of them.

    Each station is returned on the **full archive grid**, so that a station
    which stopped recording in February 2025 is explicitly missing thereafter
    rather than simply absent. That distinction is what makes the completeness
    picture in this study honest.

    Parameters
    ----------
    archive_dir : str
        Read-only ``.adc`` archive.
    cache_dir : str
        Local working copy directory.
    start : str, optional
        First day to read. Default :data:`ARCHIVE_START`.
    end : str, optional
        Last day to read. Default ``None``, meaning the last file present.
    freq : str, optional
        Analysis grid. Default :data:`ANALYSIS_FREQ`.
    stations : tuple of str, optional
        Legacy stations to extract. Default :data:`STATIONS`.
    verbose : bool, optional
        Print a loading report. Default ``True``.

    Returns
    -------
    frames : dict
        Station identifier to DataFrame, each on the full archive grid.
    report : dict
        Extent, file counts and the era boundary actually observed.
    """
    if end is None:
        end = _last_archive_day(archive_dir)

    legacy_end = min(pd.Timestamp(LEGACY_END), pd.Timestamp(end))
    has_current = pd.Timestamp(end) >= pd.Timestamp(CURRENT_START)

    if verbose:
        print(f'Archive {start} to {end}')
        print(f'  legacy era : {start} to {legacy_end.date()}')
        print(f'  current era: '
              f'{CURRENT_START if has_current else "not reached"} to {end}')

    legacy = {}
    for station in stations:
        if verbose:
            print(f'\n[legacy] {station}')
        legacy[station] = lc.load_legacy(
            archive_dir, cache_dir, start, legacy_end,
            station=station, freq=freq, verbose=verbose)

    current = None
    if has_current:
        if verbose:
            print(f'\n[current] {TARGET_STATION}')
        current = tc.load_archive(archive_dir, cache_dir, CURRENT_START, end,
                                  freq=freq, verbose=verbose)

    grid = pd.date_range(pd.Timestamp(start).floor('D'),
                         pd.Timestamp(end).ceil('D'), freq=freq)

    frames = {}
    for station in stations:
        frames[station] = _assemble_station(
            station, legacy[station], current if station == TARGET_STATION
            else None, grid)

    report = {
        'start': str(grid.min()),
        'end': str(grid.max()),
        'freq': freq,
        'slots': int(len(grid)),
        'span_days': round(len(grid) * _step_hours(freq) / 24.0, 1),
        'legacy_end': str(legacy_end.date()),
        'current_start': CURRENT_START if has_current else None,
        'stations': list(stations),
        'current_era_station': TARGET_STATION if has_current else None,
    }
    return frames, report


def _last_archive_day(archive_dir):
    """
    Date of the most recent ``.adc`` file present in the archive.

    Parameters
    ----------
    archive_dir : str
        Read-only archive directory.

    Returns
    -------
    str
        Date in ``YYYY-MM-DD`` form.
    """
    stamps = []
    for name in os.listdir(archive_dir):
        if not name.lower().endswith('.adc'):
            continue
        digits = ''.join(ch for ch in name if ch.isdigit())
        if len(digits) >= 8:
            stamps.append(digits[:8])
    if not stamps:
        raise ValueError(f'no .adc files found in {archive_dir}')
    return pd.Timestamp(max(stamps)).strftime('%Y-%m-%d')


def _step_hours(freq):
    """
    Hours per step of a frequency string.

    Parameters
    ----------
    freq : str
        Pandas frequency string.

    Returns
    -------
    float
        Step length in hours.
    """
    return pd.Timedelta(
        pd.tseries.frequencies.to_offset(freq)).total_seconds() / 3600.0


def _assemble_station(station, legacy_df, current_df, grid):
    """
    Put one station's two eras on the common grid, labelled and compensated.

    The eras are placed side by side in time, never blended. Compensation is
    applied **within each era separately**, because the correction is anchored
    to the first record carrying both a signal and a temperature and that anchor
    belongs to one instrument only.

    Parameters
    ----------
    station : str
        Station identifier.
    legacy_df : pd.DataFrame
        Legacy-era record for this station.
    current_df : pd.DataFrame or None
        Current-era record, for the instrumented station only.
    grid : pd.DatetimeIndex
        Full archive grid.

    Returns
    -------
    pd.DataFrame
        One row per slot of ``grid``.
    """
    columns = list(COMMON_CHANNELS) + list(CURRENT_ONLY_CHANNELS)
    out = pd.DataFrame(index=grid, columns=columns, dtype=float)
    out.index.name = 'datetime'
    out['era'] = pd.Series(pd.NA, index=grid, dtype='object')
    out['instrument'] = pd.Series(pd.NA, index=grid, dtype='object')

    pieces = [('legacy', legacy_df)]
    if current_df is not None:
        pieces.append(('current', current_df))

    for era, frame in pieces:
        if frame is None or frame.empty:
            continue
        idx = frame.index.intersection(grid)
        for col in columns:
            if col in frame.columns:
                out.loc[idx, col] = frame.loc[idx, col].astype(float)
        out.loc[idx, 'era'] = era
        out.loc[idx, 'instrument'] = f'{INSTRUMENT[era]}_{station}'

    # Compensation, per era, on that era's own anchor.
    out['inc_comp'] = np.nan
    for era, _ in pieces:
        mask = out['era'] == era
        if not mask.any():
            continue
        sub = out.loc[mask, ['inc', 'tair']]
        out.loc[mask, 'inc_comp'] = tc.compensate(
            sub, temp_col='tair', coeff=COMP_COEFF, normalise=True).values

    # Provenance. Every value in this table is measured; the study fills nothing
    # here. The column exists so that products built downstream inherit it.
    out['inc_source'] = np.where(out['inc'].notna(), 'observed', 'missing')
    out['station'] = station
    return out


# ──────────────────────────────────────────────────────────────────────
# Per-station classification
# ──────────────────────────────────────────────────────────────────────

def classify_stations(frames, freq=ANALYSIS_FREQ):
    """
    Summarise what each station actually contributes to the archive.

    The question this answers is which stations are usable for what, and it is
    asked once here so that no later step has to rediscover that two of the
    three stopped recording in February 2025.

    Parameters
    ----------
    frames : dict
        Output of :func:`load_all_stations`.
    freq : str, optional
        Analysis grid, for converting slots to days.

    Returns
    -------
    pd.DataFrame
        One row per station.
    """
    step = _step_hours(freq)
    rows = []
    for station, df in frames.items():
        inc = df['inc']
        observed = inc.notna()
        first = inc.first_valid_index()
        last = inc.last_valid_index()
        eras = [e for e in ('legacy', 'current')
                if bool((df['era'] == e).any())]
        entry = {
            'station': station,
            'eras': '+'.join(eras),
            'first_reading': str(first) if first is not None else '',
            'last_reading': str(last) if last is not None else '',
            'span_days': round(((last - first).total_seconds() / 86400.0)
                               if first is not None and last is not None
                               else 0.0, 1),
            'observed_hours': int(observed.sum()),
            'observed_days': round(observed.sum() * step / 24.0, 1),
            'coverage_of_span_%': round(
                100.0 * observed.sum()
                / max(int(((last - first) / pd.Timedelta(hours=step)) + 1), 1), 1)
            if first is not None and last is not None else 0.0,
            'coverage_of_archive_%': round(100.0 * observed.mean(), 1),
        }
        for channel in COMMON_CHANNELS + CURRENT_ONLY_CHANNELS:
            entry[f'{channel}_%'] = round(100.0 * df[channel].notna().mean(), 1)
        entry['inc_min'] = round(float(inc.min()), 2) if observed.any() else np.nan
        entry['inc_max'] = round(float(inc.max()), 2) if observed.any() else np.nan
        rows.append(entry)
    return pd.DataFrame(rows).set_index('station')


def era_profile(df, freq=ANALYSIS_FREQ):
    """
    Split one station's record by era and profile each side.

    Parameters
    ----------
    df : pd.DataFrame
        One station's unified frame.
    freq : str, optional
        Analysis grid.

    Returns
    -------
    pd.DataFrame
        One row per era present.
    """
    step = _step_hours(freq)
    rows = []
    for era in ('legacy', 'current'):
        mask = df['era'] == era
        if not mask.any():
            continue
        sub = df.loc[mask]
        entry = {
            'era': era,
            'from': str(sub.index.min().date()),
            'to': str(sub.index.max().date()),
            'slots': int(len(sub)),
            'span_days': round(len(sub) * step / 24.0, 1),
            'annual_cycles': round(len(sub) * step / 24.0 / YEAR_DAYS, 2),
        }
        for channel in COMMON_CHANNELS + CURRENT_ONLY_CHANNELS:
            entry[f'{channel}_%'] = round(100.0 * sub[channel].notna().mean(), 1)
        entry['inc_mean'] = round(float(sub['inc'].mean()), 2)
        entry['inc_sd'] = round(float(sub['inc'].std()), 2)
        rows.append(entry)
    return pd.DataFrame(rows).set_index('era')


# ──────────────────────────────────────────────────────────────────────
# The era changeover
# ──────────────────────────────────────────────────────────────────────

def estimate_era_offset(df, window_days=30, deseason=True,
                        periods=None, target='inc_comp'):
    """
    Estimate the level step between the two instruments at the changeover.

    The naive difference between the last legacy observations and the first
    current-era ones confounds the instrument offset with the season, because
    the two windows sit at different points of the annual cycle. Subtracting a
    harmonic model fitted to the legacy era removes that contribution, and what
    survives is the part attributable to the instrument.

    The estimate is honest about what it cannot separate. A re-installed
    inclinometer can differ from its predecessor in gain as well as in zero, and
    a level comparison sees only the zero. The returned diagnostics therefore
    include the diurnal amplitude on each side, whose ratio is the sharpest
    available check on gain; a ratio far from one means the offset alone does
    not reconcile the two instruments.

    Parameters
    ----------
    df : pd.DataFrame
        Unified frame for one station, both eras present.
    window_days : float, optional
        Comparison window on each side. Default ``30``.
    deseason : bool, optional
        Remove a harmonic model fitted to the legacy era. Default ``True``.
    periods : dict, optional
        Harmonic components. Defaults to :data:`lp_lib.HARMONIC_PERIODS`.
    target : str, optional
        Column to align. Default ``'inc_comp'``.

    Returns
    -------
    dict
        ``offset`` and the diagnostics behind it.
    """
    if periods is None:
        periods = lp.HARMONIC_PERIODS

    legacy = df.loc[df['era'] == 'legacy', target].dropna()
    current = df.loc[df['era'] == 'current', target].dropna()
    if legacy.empty or current.empty:
        raise ValueError('both eras must carry the target to estimate an offset')

    left_end, right_start = legacy.index.max(), current.index.min()
    span = pd.Timedelta(days=window_days)
    before = legacy.loc[left_end - span:left_end]
    after = current.loc[right_start:right_start + span]

    raw_step = float(after.mean() - before.mean())

    result = {
        'left_end': str(left_end),
        'right_start': str(right_start),
        'interruption_days': round(
            (right_start - left_end).total_seconds() / 86400.0, 2),
        'n_before': int(len(before)),
        'n_after': int(len(after)),
        'mean_before': round(float(before.mean()), 3),
        'mean_after': round(float(after.mean()), 3),
        'raw_step': round(raw_step, 3),
    }

    offset = raw_step
    if deseason:
        table, fitted, _ = lp.fit_harmonics(legacy, periods, trend=False)
        seasonal = _extend_harmonics(table, legacy.index.min(),
                                     df.index, periods)
        adj_before = (before - seasonal.reindex(before.index)).mean()
        adj_after = (after - seasonal.reindex(after.index)).mean()
        offset = float(adj_after - adj_before)
        result['seasonal_step'] = round(raw_step - offset, 3)
        result['deseasoned_step'] = round(offset, 3)
        result['annual_amplitude'] = round(
            float(table.loc['annual', 'amplitude_mdeg']), 3)

    # Gain check: the diurnal amplitude either side of the changeover.
    amp_before = tc.diurnal_amplitude(before)
    amp_after = tc.diurnal_amplitude(after)
    result['diurnal_amp_before'] = round(float(amp_before), 3)
    result['diurnal_amp_after'] = round(float(amp_after), 3)
    result['gain_ratio'] = (round(float(amp_after / amp_before), 3)
                            if amp_before else np.nan)
    result['offset'] = round(offset, 3)
    return result


def _extend_harmonics(table, t0, index, periods):
    """
    Evaluate a fitted harmonic model on an arbitrary index.

    :func:`lp_lib.fit_harmonics` returns the fit on its own index only. The
    changeover comparison needs the same model evaluated on the other side of
    the boundary, which is a deliberate extrapolation of the seasonal terms and
    of nothing else — no trend is carried across.

    Parameters
    ----------
    table : pd.DataFrame
        Component table from :func:`lp_lib.fit_harmonics`.
    t0 : pd.Timestamp
        Time origin the phases are referred to.
    index : pd.DatetimeIndex
        Index to evaluate on.
    periods : dict
        Component periods in days.

    Returns
    -------
    pd.Series
        The seasonal expectation on ``index``.
    """
    days = (index - pd.Timestamp(t0)).total_seconds() / 86400.0
    out = np.zeros(len(index))
    for name, period in periods.items():
        if name not in table.index:
            continue
        amp = float(table.loc[name, 'amplitude_mdeg'])
        peak = float(table.loc[name, 'peak_day_after_t0'])
        out += amp * np.cos(2 * np.pi * (days - peak) / period)
    return pd.Series(out, index=index)


def apply_era_join(df, offset, target='inc_comp', out_col='inc_comp_joined'):
    """
    Add the offset-aligned continuous series as a separate, named column.

    The primary columns are left untouched. A consumer that wants one continuous
    level series across 2025-02-21 uses this column and thereby accepts an
    estimated constant; a consumer that does not, does not.

    Parameters
    ----------
    df : pd.DataFrame
        Unified frame for one station.
    offset : float
        Value to subtract from the current era, from
        :func:`estimate_era_offset`.
    target : str, optional
        Source column. Default ``'inc_comp'``.
    out_col : str, optional
        Name of the derived column. Default ``'inc_comp_joined'``.

    Returns
    -------
    pd.DataFrame
        A copy carrying the derived column.
    """
    out = df.copy()
    out[out_col] = out[target]
    current = out['era'] == 'current'
    out.loc[current, out_col] = out.loc[current, target] - offset
    return out


# ──────────────────────────────────────────────────────────────────────
# Persisting the dataset
# ──────────────────────────────────────────────────────────────────────

def save_unified(frames, out_dir, report=None, freq=ANALYSIS_FREQ):
    """
    Write one CSV per station plus a manifest describing the build.

    Parameters
    ----------
    frames : dict
        Station identifier to DataFrame.
    out_dir : str
        Destination directory. Created if absent.
    report : dict, optional
        Build report from :func:`load_all_stations`, stored in the manifest.
    freq : str, optional
        Analysis grid, recorded in the manifest.

    Returns
    -------
    dict
        Path written per station, plus ``manifest``.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for station, df in frames.items():
        path = os.path.join(out_dir, f'unified_{station}_{freq}.csv')
        df.to_csv(path)
        written[station] = path

    manifest = {
        'built': pd.Timestamp.now().isoformat(timespec='seconds'),
        'freq': freq,
        'compensation_coefficient': COMP_COEFF,
        'compensation_note': (
            'Documented value, never calibrated for these instruments. '
            'Applied per era on that era own anchor.'),
        'columns': {station: list(df.columns) for station, df in frames.items()},
        'rows': {station: int(len(df)) for station, df in frames.items()},
        'files': written,
    }
    if report:
        manifest['build_report'] = report

    manifest_path = os.path.join(out_dir, 'unified_manifest.json')
    with open(manifest_path, 'w') as fh:
        json.dump(manifest, fh, indent=2)
    written['manifest'] = manifest_path
    return written


def load_unified(out_dir, station=TARGET_STATION, freq=ANALYSIS_FREQ):
    """
    Read back one station's unified table.

    Parameters
    ----------
    out_dir : str
        Directory written by :func:`save_unified`.
    station : str, optional
        Station to read. Default :data:`TARGET_STATION`.
    freq : str, optional
        Analysis grid. Default :data:`ANALYSIS_FREQ`.

    Returns
    -------
    pd.DataFrame
        The station's table, datetime-indexed.
    """
    path = os.path.join(out_dir, f'unified_{station}_{freq}.csv')
    # low_memory=False reads each column in one pass. Without it pandas infers
    # dtypes chunk by chunk and warns on `era` and `instrument`, whose first
    # chunks are all-NaN at the stations that stopped recording early.
    return pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)


# ──────────────────────────────────────────────────────────────────────
# What the record is made of: completeness and the anatomy of the gaps
# ──────────────────────────────────────────────────────────────────────

def gap_table(series, dt_hours=1.0, min_hours=1.0):
    """
    Every interruption in a series, individually rather than by band.

    A band histogram says how much is missing; it does not say *when*, and the
    when is what decides whether a gap is fillable. A three-day hole in
    midwinter and a three-day hole in midsummer are the same length and not the
    same problem, because the diurnal amplitude a filler has to reproduce
    differs by a factor of several between them.

    Parameters
    ----------
    series : pd.Series
        Source series on a regular index.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.
    min_hours : float, optional
        Shortest interruption to report. Default ``1.0``.

    Returns
    -------
    pd.DataFrame
        One row per gap: its bounds, length, season, and the length of the
        observed runs flanking it.
    """
    missing = series.isna().to_numpy()
    if not missing.any():
        return pd.DataFrame(columns=['start', 'end', 'hours', 'days'])

    edges = np.flatnonzero(np.diff(np.r_[0, missing.astype(int), 0]))
    starts, stops = edges[::2], edges[1::2]
    index = series.index

    rows = []
    for k, (a, b) in enumerate(zip(starts, stops)):
        hours = (b - a) * dt_hours
        if hours < min_hours:
            continue
        # Observed run immediately before and after, which is the context any
        # filler has to work from.
        before = a - (stops[k - 1] if k > 0 else 0)
        after = (starts[k + 1] if k + 1 < len(starts) else len(missing)) - b
        rows.append({
            'start': index[a],
            'end': index[b - 1],
            'hours': float(hours),
            'days': round(hours / 24.0, 2),
            'month': int(index[a].month),
            'season': _season(index[a].month),
            'year': int(index[a].year),
            'context_before_h': float(before * dt_hours),
            'context_after_h': float(after * dt_hours),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values('hours', ascending=False).reset_index(drop=True)
    return out


def _season(month):
    """
    Meteorological season of a month, northern hemisphere.

    Parameters
    ----------
    month : int
        Month number, 1 to 12.

    Returns
    -------
    str
        ``'DJF'``, ``'MAM'``, ``'JJA'`` or ``'SON'``.
    """
    return {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
            6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON',
            11: 'SON'}[int(month)]


def gap_bands(gaps, edges=(0, 3, 6, 24, 72, 168, 720, np.inf),
              labels=None, total_missing=None):
    """
    Aggregate a gap table into length bands.

    Parameters
    ----------
    gaps : pd.DataFrame
        Output of :func:`gap_table`.
    edges : tuple, optional
        Band boundaries in hours, exclusive below and inclusive above.
    labels : list of str, optional
        Band names. Defaults to names derived from the edges.
    total_missing : float, optional
        Denominator for the share column. Defaults to the sum over the bands.

    Returns
    -------
    pd.DataFrame
        One row per band.
    """
    if labels is None:
        labels = ['<= 3 h', '3-6 h', '6-24 h', '1-3 d', '3-7 d', '7-30 d',
                  '> 30 d']
    total = total_missing if total_missing else float(gaps['hours'].sum())
    rows = []
    for (lo, hi), name in zip(zip(edges[:-1], edges[1:]), labels):
        sel = gaps[(gaps['hours'] > lo) & (gaps['hours'] <= hi)]
        rows.append({
            'band': name,
            'n_gaps': int(len(sel)),
            'hours_lost': float(sel['hours'].sum()),
            'share_of_missing_%': round(
                100.0 * sel['hours'].sum() / total, 1) if total else 0.0,
            'median_h': round(float(sel['hours'].median()), 1) if len(sel) else np.nan,
        })
    return pd.DataFrame(rows).set_index('band')


def completeness_calendar(series, dt_hours=1.0):
    """
    Daily completeness as a year-by-day-of-year matrix.

    The layout a calendar heat map needs, and the clearest way to see whether
    interruptions are scattered or seasonal.

    Parameters
    ----------
    series : pd.Series
        Source series on a regular index.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.

    Returns
    -------
    pd.DataFrame
        Rows are years, columns day-of-year, values the fraction of the day
        observed.
    """
    per_day = series.notna().resample('1D').mean()
    frame = pd.DataFrame({
        'year': per_day.index.year,
        'doy': per_day.index.dayofyear,
        'completeness': per_day.values,
    })
    return frame.pivot_table(index='year', columns='doy',
                             values='completeness')


def monthly_completeness(frames, column='inc'):
    """
    Fraction of each month observed, per station.

    Parameters
    ----------
    frames : dict
        Station identifier to DataFrame.
    column : str, optional
        Channel to measure. Default ``'inc'``.

    Returns
    -------
    pd.DataFrame
        Rows are months, columns stations.
    """
    out = {}
    for station, df in frames.items():
        out[station] = df[column].notna().resample('1MS').mean()
    return pd.DataFrame(out)


def channel_availability(df, columns=None):
    """
    Daily availability of every channel, for a stacked timeline.

    Parameters
    ----------
    df : pd.DataFrame
        One station's unified frame.
    columns : list of str, optional
        Channels to include. Defaults to every numeric channel present.

    Returns
    -------
    pd.DataFrame
        Rows are days, columns channels, values the observed fraction.
    """
    if columns is None:
        columns = [c for c in COMMON_CHANNELS + CURRENT_ONLY_CHANNELS
                   if c in df.columns]
    return df[columns].notna().resample('1D').mean()


def missingness_mechanism(df, target='inc', covariates=None, dt_hours=1.0):
    """
    Classify why the target is missing, against the covariates that are present.

    The taxonomy is a design input, not a label. A gap that is missing at random
    can be filled from its neighbours; one whose absence tracks a supply channel
    is an outage, and an outage removes the drivers along with the response, so
    nothing conditioned on them can reconstruct it.

    The test is deliberately simple: correlate a binary missingness indicator
    against each covariate over the rows where that covariate is observed. A
    covariate that itself vanishes with the target cannot be tested this way,
    and is reported as jointly missing instead — which is itself the finding.

    Parameters
    ----------
    df : pd.DataFrame
        One station's unified frame.
    target : str, optional
        Response channel. Default ``'inc'``.
    covariates : list of str, optional
        Channels to test against. Defaults to the other common channels.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.

    Returns
    -------
    pd.DataFrame
        One row per covariate.
    """
    from scipy import stats

    if covariates is None:
        covariates = [c for c in COMMON_CHANNELS + CURRENT_ONLY_CHANNELS
                      if c != target and c in df.columns]

    absent = df[target].isna()
    rows = []
    for name in covariates:
        col = df[name]
        both_missing = float((absent & col.isna()).sum())
        joint = (100.0 * both_missing / absent.sum()) if absent.sum() else 0.0
        testable = col.notna()
        if testable.sum() < 10 or absent[testable].nunique() < 2:
            # Nothing to correlate: the covariate is never present while the
            # target is absent. That is not a failed test, it is the strongest
            # possible answer — the two fail together, so the covariate can
            # never help reconstruct the target.
            rows.append({
                'covariate': name, 'joint_missing_%': round(joint, 1),
                'r': np.nan, 'p': np.nan, 'testable_rows': 0,
                'reading': ('always missing together — outage (MNAR), '
                            'unusable for reconstruction')
                if joint > 95.0 else 'never observed while target is absent'})
            continue
        r, p = stats.pointbiserialr(absent[testable].astype(int),
                                    col[testable].astype(float))
        if joint > 95.0:
            reading = 'missing together — outage (MNAR)'
        elif p < 0.05:
            reading = 'missingness tracks this channel (MAR)'
        else:
            reading = 'no association (MCAR-consistent)'
        rows.append({'covariate': name, 'joint_missing_%': round(joint, 1),
                     'r': round(float(r), 4), 'p': float(p),
                     'testable_rows': int(testable.sum()),
                     'reading': reading})
    out = pd.DataFrame(rows).set_index('covariate')
    out.attrs['missing_hours'] = float(absent.sum() * dt_hours)
    out.attrs['missing_%'] = round(100.0 * float(absent.mean()), 2)
    return out


# ──────────────────────────────────────────────────────────────────────
# Fillers
# ──────────────────────────────────────────────────────────────────────
#
# Six fillers of increasing ambition. The point of the range is that no single
# method is right at every gap length: interpolation is unbeatable across an
# hour and meaningless across a month, while a seasonal model knows nothing
# useful about an hour and is the only thing left after a week.
#
# Each takes the series with the gap already removed and returns a filled
# series. None of them sees the withheld values.

def fill_interpolate(series, **kwargs):
    """
    Linear interpolation in time between the flanking observations.

    Parameters
    ----------
    series : pd.Series
        Series with the gap present as NaN.

    Returns
    -------
    pd.Series
        Filled series.
    """
    return series.interpolate(method='time', limit_direction='both')


def fill_seasonal_naive(series, period_h=24, **kwargs):
    """
    Carry the value from one period earlier, falling back to one period later.

    The cheapest filler that knows the signal has a daily cycle.

    Parameters
    ----------
    series : pd.Series
        Series with the gap present as NaN.
    period_h : int, optional
        Cycle length in steps. Default ``24``.

    Returns
    -------
    pd.Series
        Filled series.
    """
    out = series.copy()
    for _ in range(14):                       # up to a fortnight of carrying
        if out.notna().all():
            break
        out = out.fillna(out.shift(period_h)).fillna(out.shift(-period_h))
    return out.fillna(series.interpolate(method='time', limit_direction='both'))


def fill_harmonic(series, periods=None, **kwargs):
    """
    Evaluate a least-squares harmonic model fitted to the observed part.

    Knows the annual and diurnal cycles and nothing else, so its error does not
    grow with gap length — which makes it the only sensible candidate for the
    longest holes, and a poor one for short ones.

    Parameters
    ----------
    series : pd.Series
        Series with the gap present as NaN.
    periods : dict, optional
        Component periods in days. Defaults to
        :data:`lp_lib.HARMONIC_PERIODS`.

    Returns
    -------
    pd.Series
        Filled series.
    """
    if periods is None:
        periods = lp.HARMONIC_PERIODS
    observed = series.dropna()
    if len(observed) < 4 * len(periods):
        return fill_interpolate(series)

    # A component can only be fitted from a window that contains it several
    # times over. An annual basis evaluated across five weeks is very nearly a
    # straight line, so including it turns the fit into an unconstrained
    # extrapolation — which is how this filler produced errors of several
    # hundred millidegrees before the restriction was added.
    span_days = (series.index.max() - series.index.min()).total_seconds() / 86400.0
    usable = {name: period for name, period in periods.items()
              if period * MIN_CYCLES_TO_FIT <= span_days}
    if not usable:
        return fill_interpolate(series)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # No trend term either. A straight line fitted on the window flanking a
        # gap and extrapolated across it is the single most damaging term
        # available on this signal — the legacy study priced it at 85 to
        # 93 mdeg over the whole record.
        _, fitted, _ = lp.fit_harmonics(series, usable, trend=False)
    return series.fillna(fitted)



def fill_drivers(series, drivers=None, **kwargs):
    """
    Ridge regression of the target on the environmental channels.

    Parameters
    ----------
    series : pd.Series
        Series with the gap present as NaN.
    drivers : pd.DataFrame, optional
        Driver columns on the same index. Without them this degrades to
        interpolation, which is the honest outcome: an outage removes the
        drivers too.

    Returns
    -------
    pd.Series
        Filled series.
    """
    from sklearn.linear_model import Ridge

    if drivers is None or drivers.empty:
        return fill_interpolate(series)

    X = drivers.reindex(series.index)
    train = X.notna().all(axis=1) & series.notna()
    predict = X.notna().all(axis=1) & series.isna()
    if train.sum() < 24 or not predict.any():
        return fill_interpolate(series)

    model = Ridge(alpha=1.0).fit(X[train].to_numpy(), series[train].to_numpy())
    out = series.copy()
    out.loc[predict] = model.predict(X[predict].to_numpy())
    return fill_interpolate(out)


def fill_kalman(series, period_h=24, max_context=4000, **kwargs):
    """
    Local-linear-trend state-space model with a daily seasonal, smoothed.

    The only filler here with a native notion of its own uncertainty: the
    smoother returns a variance for every interpolated state, which grows with
    distance into the gap because the model knows it is extrapolating. That is
    the behaviour uncertainty quantification wants, and none of the others have
    it.

    Fitted on a context window rather than the whole record, because the
    likelihood is evaluated by a filter pass whose cost is linear in length and
    the state is local anyway.

    Parameters
    ----------
    series : pd.Series
        Series with the gap present as NaN.
    period_h : int, optional
        Seasonal period in steps. Default ``24``.
    max_context : int, optional
        Longest window to fit on. Default ``4000``.

    Returns
    -------
    pd.Series
        Filled series.
    """
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    work = series
    if len(work) > max_context:
        centre = int(np.mean(np.flatnonzero(series.isna().to_numpy()))) \
            if series.isna().any() else len(series) // 2
        lo = max(0, centre - max_context // 2)
        work = series.iloc[lo:lo + max_context]

    if work.notna().sum() < 3 * period_h:
        return fill_interpolate(series)

    smoothed, _ = _kalman_smooth(work, period_h)
    if smoothed is None:
        return fill_interpolate(series)

    out = series.copy()
    out.loc[work.index] = work.fillna(smoothed)
    return fill_interpolate(out)


def _kalman_smooth(series, period_h=24, harmonics=2, maxiter=50):
    """
    Smoothed signal and its standard deviation from a structural model.

    ``MLEResults.smoothed_forecasts`` is ``None`` for this model class, so the
    signal is reconstructed from the smoothed state directly as
    :math:`Z\\alpha_{t|n}`, and its variance as :math:`Z P_{t|n} Z'`. That
    variance is the reason this filler is here: it widens inside a gap, because
    the smoother knows it is interpolating rather than observing.

    Parameters
    ----------
    series : pd.Series
        Series with gaps present as NaN.
    period_h : int, optional
        Seasonal period in steps. Default ``24``.
    harmonics : int, optional
        Fourier harmonics for the seasonal. Default ``2``.
    maxiter : int, optional
        Optimiser iterations. Default ``50``.

    Returns
    -------
    signal : pd.Series or None
        Smoothed signal, or ``None`` if the fit failed.
    sd : pd.Series or None
        Standard deviation of the smoothed signal.
    """
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = UnobservedComponents(
                series.to_numpy(dtype=float), level='local linear trend',
                freq_seasonal=[{'period': period_h, 'harmonics': harmonics}])
            res = model.fit(disp=False, maxiter=maxiter)

        design = np.asarray(model['design'])
        z = design[0, :, 0] if design.ndim == 3 else design[0]
        signal = z @ res.smoothed_state
        var = np.einsum('i,ijt,j->t', z, res.smoothed_state_cov, z)
        return (pd.Series(signal, index=series.index),
                pd.Series(np.sqrt(np.clip(var, 0, None)), index=series.index))
    except Exception:
        return None, None


def fill_neuralprophet(series, freq=ANALYSIS_FREQ, epochs=12, **kwargs):
    """
    Fill from the series' own NeuralProphet trend-plus-seasonal fit.

    Included because the project's framework is built on this decomposition, so
    what it does as a filler is worth knowing rather than assuming. It is fitted
    without autoregression, since a model carrying lags cannot produce a value
    inside a gap: the lags it needs are themselves missing.

    Parameters
    ----------
    series : pd.Series
        Series with the gap present as NaN.
    freq : str, optional
        Sampling frequency. Default :data:`ANALYSIS_FREQ`.
    epochs : int, optional
        Training epochs. Default ``12``.

    Returns
    -------
    pd.Series
        Filled series.
    """
    if series.notna().sum() < 200:
        return fill_interpolate(series)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            components, _ = lc.decompose_series(
                series, freq=freq, epochs=epochs, yearly=True, weekly=False,
                daily=True, verbose=False)
        return series.fillna(components['yhat'])
    except Exception:
        return fill_interpolate(series)


#: The filler registry. Order is roughly increasing ambition.
FILLERS = {
    'interpolate': fill_interpolate,
    'seasonal_naive': fill_seasonal_naive,
    'kalman': fill_kalman,
    'harmonic': fill_harmonic,
    'drivers': fill_drivers,
    'neuralprophet': fill_neuralprophet,
}

#: Fillers cheap enough to run on every trial of a large benchmark.
FAST_FILLERS = ('interpolate', 'seasonal_naive', 'kalman', 'harmonic',
                'drivers')


# ──────────────────────────────────────────────────────────────────────
# Benchmarking the fillers, and the uncertainty that goes with them
# ──────────────────────────────────────────────────────────────────────

def benchmark_fillers(df, target, gap_lengths=(1, 3, 6, 12, 24, 72, 168, 720),
                      methods=FAST_FILLERS, drivers=None, n_trials=40, seed=0,
                      min_context_hours=48, dt_hours=1.0, keep_errors=True):
    """
    Measure each filler against gaps of known length, injected where the data
    is complete.

    This is the generalisation of the benchmark in ``lp_lib``: more methods, and
    it retains **every individual error** rather than only their mean. The
    per-trial errors are what makes a distribution-free interval possible in
    :func:`conformal_bands`; a mean absolute error on its own cannot say how
    wrong a filled value might be, only how wrong it is on average.

    Parameters
    ----------
    df : pd.DataFrame
        Source frame carrying the target and any drivers.
    target : str
        Column to fill.
    gap_lengths : tuple of int, optional
        Gap lengths to test, in steps.
    methods : tuple of str, optional
        Keys of :data:`FILLERS` to test.
    drivers : list of str, optional
        Driver columns, passed to the driver-based filler.
    n_trials : int, optional
        Injection sites per gap length. Default ``40``.
    seed : int, optional
        Random seed. Default ``0``.
    min_context_hours : float, optional
        Observed context required either side of an injection site.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.
    keep_errors : bool, optional
        Retain the per-trial errors on the result's ``attrs``. Default ``True``.

    Returns
    -------
    pd.DataFrame
        One row per gap length and method.
    """
    rng = np.random.default_rng(seed)
    series = df[target]
    observed = series.notna().to_numpy()
    driver_frame = df[drivers] if drivers else None

    context = int(round(min_context_hours / dt_hours))
    rows, error_store = [], {}

    for length in gap_lengths:
        need = length + 2 * context
        # Candidate sites: every window of the required width that is complete.
        run = np.convolve(observed.astype(int), np.ones(need, dtype=int),
                          mode='valid')
        candidates = np.flatnonzero(run == need)
        if len(candidates) == 0:
            continue
        picks = rng.choice(candidates,
                           size=min(n_trials, len(candidates)), replace=False)

        per_method = {m: [] for m in methods}
        for start in picks:
            lo = start
            hi = start + need
            window = series.iloc[lo:hi].copy()
            gap_lo = context
            gap_hi = context + length
            truth = window.iloc[gap_lo:gap_hi].to_numpy(dtype=float)

            holed = window.copy()
            holed.iloc[gap_lo:gap_hi] = np.nan
            sub_drivers = (driver_frame.iloc[lo:hi]
                           if driver_frame is not None else None)

            for name in methods:
                filled = FILLERS[name](holed, drivers=sub_drivers)
                got = filled.iloc[gap_lo:gap_hi].to_numpy(dtype=float)
                if np.isnan(got).any():
                    continue
                per_method[name].append(got - truth)

        for name in methods:
            if not per_method[name]:
                continue
            errors = np.concatenate(per_method[name])
            rows.append({
                'gap_hours': length * dt_hours,
                'method': name,
                'MAE_mdeg': round(float(np.abs(errors).mean()), 4),
                'RMSE_mdeg': round(float(np.sqrt((errors ** 2).mean())), 4),
                'bias_mdeg': round(float(errors.mean()), 4),
                'p90_abs_mdeg': round(float(np.percentile(np.abs(errors), 90)), 4),
                'max_abs_mdeg': round(float(np.abs(errors).max()), 4),
                'n_trials': int(len(per_method[name])),
                'n_values': int(errors.size),
            })
            if keep_errors:
                error_store[(length * dt_hours, name)] = errors

    out = pd.DataFrame(rows)
    if keep_errors:
        out.attrs['errors'] = error_store
    return out


def conformal_bands(bench, level=0.90):
    """
    Distribution-free interval half-width for each method and gap length.

    The benchmark's per-trial errors are treated as a calibration sample: the
    interval that covers ``level`` of them is, by construction, the interval
    that would have covered that fraction of the errors actually observed. No
    distributional assumption is made, which matters because these errors are
    neither Gaussian nor independent.

    The guarantee is marginal and holds under exchangeability between the
    injected gaps and the real ones. Real outages are not exchangeable with
    injected ones — they happen for reasons — so this is a floor on the
    uncertainty rather than a complete account of it, and the study says so.

    Parameters
    ----------
    bench : pd.DataFrame
        Output of :func:`benchmark_fillers`, with errors retained.
    level : float, optional
        Nominal coverage. Default ``0.90``.

    Returns
    -------
    pd.DataFrame
        One row per gap length and method, with the half-width and the coverage
        it achieves in sample.
    """
    errors = bench.attrs.get('errors')
    if not errors:
        raise ValueError('benchmark carries no per-trial errors; '
                         'run benchmark_fillers with keep_errors=True')
    rows = []
    for (gap_hours, method), sample in errors.items():
        half = float(np.quantile(np.abs(sample), level))
        rows.append({
            'gap_hours': gap_hours,
            'method': method,
            'half_width_mdeg': round(half, 4),
            'nominal_%': round(100 * level, 1),
            'empirical_%': round(
                100.0 * float((np.abs(sample) <= half).mean()), 1),
            'n': int(sample.size),
        })
    return (pd.DataFrame(rows)
            .sort_values(['gap_hours', 'method'])
            .reset_index(drop=True))


#: Fillers the benchmark measures but a policy must never select.
#:
#: The driver-based filler needs the environmental channels over the interval it
#: is reconstructing. The benchmark injects its gaps where the record is
#: complete, so those channels are present and the filler scores well. In a real
#: gap they are absent — every channel of this station fails together — so the
#: method cannot run at all, and would silently degrade to interpolation while
#: still being credited with the accuracy it showed on injected gaps.
#:
#: Measuring it and then refusing it is deliberate: the gap between its
#: benchmark score and its real-world availability is the clearest statement
#: this study can make about why an availability criterion belongs in the
#: framework beside an accuracy one.
UNAVAILABLE_IN_REAL_GAPS = ('drivers',)


def imputation_policy(bench, tolerance_mdeg, bands=None,
                      exclude=UNAVAILABLE_IN_REAL_GAPS):
    """
    Choose one filler per gap-length band, and refuse the bands nothing can do.

    A policy is a decision, so it is stated as one: for each band, the method
    with the lowest error among those meeting the tolerance, or an explicit
    refusal. A band with no admissible method is the useful output, not a
    failure — it says which holes have to stay holes.

    Selection is over the methods that can actually run on a real gap. A method
    excluded by ``exclude`` is still reported in the ``best_available`` columns,
    so the cost of excluding it is visible rather than hidden.

    Parameters
    ----------
    bench : pd.DataFrame
        Output of :func:`benchmark_fillers`.
    tolerance_mdeg : float
        Largest mean absolute error accepted from a filled value.
    bands : list of tuple, optional
        ``(label, upper_bound_hours)`` pairs. Defaults to the benchmark's own
        gap lengths.
    exclude : tuple of str, optional
        Methods that may not be selected. Defaults to
        :data:`UNAVAILABLE_IN_REAL_GAPS`.

    Returns
    -------
    pd.DataFrame
        One row per band.
    """
    if bands is None:
        bands = [(f'<= {int(h)} h', h)
                 for h in sorted(bench['gap_hours'].unique())]

    rows = []
    for label, upper in bands:
        at_length = bench[bench['gap_hours'] == upper]
        if at_length.empty:
            continue

        # What the benchmark says is best, before availability is considered.
        unrestricted = at_length.loc[at_length['MAE_mdeg'].idxmin()]
        usable = at_length[~at_length['method'].isin(exclude)]
        admissible = usable[usable['MAE_mdeg'] <= tolerance_mdeg]

        entry = {
            'band': label,
            'max_gap_h': upper,
            'benchmark_best': unrestricted['method'],
            'benchmark_best_MAE': unrestricted['MAE_mdeg'],
        }
        if admissible.empty:
            best_usable = (usable.loc[usable['MAE_mdeg'].idxmin()]
                           if len(usable) else None)
            entry.update({
                'method': 'none',
                'MAE_mdeg': np.nan,
                'best_usable': (best_usable['method'] if best_usable is not None
                                else ''),
                'best_usable_MAE': (best_usable['MAE_mdeg']
                                    if best_usable is not None else np.nan),
                'decision': 'leave missing — no available method meets the '
                            'tolerance',
            })
        else:
            chosen = admissible.loc[admissible['MAE_mdeg'].idxmin()]
            entry.update({
                'method': chosen['method'],
                'MAE_mdeg': chosen['MAE_mdeg'],
                'best_usable': chosen['method'],
                'best_usable_MAE': chosen['MAE_mdeg'],
                'decision': f'fill with {chosen["method"]}',
            })
        entry['cost_of_availability_mdeg'] = round(
            float(entry['best_usable_MAE'] - entry['benchmark_best_MAE']), 4) \
            if not pd.isna(entry.get('best_usable_MAE', np.nan)) else np.nan
        rows.append(entry)

    out = pd.DataFrame(rows).set_index('band')
    out.attrs['excluded'] = list(exclude)
    out.attrs['tolerance_mdeg'] = tolerance_mdeg
    return out


# ──────────────────────────────────────────────────────────────────────
# The stretch: how much continuous record a policy actually buys
# ──────────────────────────────────────────────────────────────────────

def apply_policy(df, target, policy, drivers=None, dt_hours=1.0,
                 conformal=None, level=0.90):
    """
    Fill a series according to a policy, keeping provenance and uncertainty.

    Each gap is filled by the method its length band assigns, or left alone if
    the band refuses. Every filled value carries the method that produced it and
    the interval half-width calibrated for that method at that gap length, so
    that a consumer can weight or exclude it.

    Parameters
    ----------
    df : pd.DataFrame
        Source frame.
    target : str
        Column to fill.
    policy : pd.DataFrame
        Output of :func:`imputation_policy`.
    drivers : list of str, optional
        Driver columns for the driver-based filler.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.
    conformal : pd.DataFrame, optional
        Output of :func:`conformal_bands`, for the per-value half-width.
    level : float, optional
        Nominal coverage of the interval. Default ``0.90``.

    Returns
    -------
    out : pd.DataFrame
        ``value``, ``source``, ``method`` and ``half_width`` per row.
    summary : dict
        What was filled, by what, and at what cost.
    """
    series = df[target]
    gaps = gap_table(series, dt_hours=dt_hours)

    value = series.copy()
    method_col = pd.Series(pd.NA, index=series.index, dtype='object')
    half_col = pd.Series(np.nan, index=series.index, dtype=float)
    method_col[series.notna()] = 'observed'

    ladder = policy.sort_values('max_gap_h')
    filled_hours, refused_hours, per_method = 0.0, 0.0, {}

    for _, gap in gaps.iterrows():
        band = ladder[ladder['max_gap_h'] >= gap['hours']]
        if band.empty or band.iloc[0]['method'] == 'none':
            refused_hours += gap['hours']
            continue
        row = band.iloc[0]
        name = row['method']

        # Fill on a window of the gap plus context, not the whole record.
        context = int(max(48, 3 * gap['hours']) / dt_hours)
        lo = series.index.get_loc(gap['start'])
        hi = series.index.get_loc(gap['end']) + 1
        a, b = max(0, lo - context), min(len(series), hi + context)
        window = value.iloc[a:b]
        sub_drivers = df[drivers].iloc[a:b] if drivers else None

        filled = FILLERS[name](window, drivers=sub_drivers)
        segment = filled.iloc[lo - a:hi - a]
        if segment.isna().any():
            refused_hours += gap['hours']
            continue

        value.iloc[lo:hi] = segment.to_numpy()
        method_col.iloc[lo:hi] = name
        if conformal is not None:
            half_col.iloc[lo:hi] = _half_width_for(conformal, name,
                                                   gap['hours'])
        filled_hours += gap['hours']
        per_method[name] = per_method.get(name, 0.0) + gap['hours']

    source = np.where(series.notna(), 'observed',
                      np.where(value.notna(), 'imputed', 'missing'))
    out = pd.DataFrame({'value': value, 'source': source,
                        'method': method_col, 'half_width': half_col},
                       index=series.index)

    summary = {
        'observed_hours': float(series.notna().sum() * dt_hours),
        'filled_hours': filled_hours,
        'refused_hours': refused_hours,
        'still_missing_hours': float(out['value'].isna().sum() * dt_hours),
        'per_method_hours': per_method,
        'coverage_before_%': round(100.0 * float(series.notna().mean()), 2),
        'coverage_after_%': round(100.0 * float(out['value'].notna().mean()), 2),
        'nominal_level_%': round(100 * level, 1),
    }
    return out, summary


def _half_width_for(conformal, method, gap_hours):
    """
    Interval half-width calibrated for one method at the nearest tested length.

    Parameters
    ----------
    conformal : pd.DataFrame
        Output of :func:`conformal_bands`.
    method : str
        Filler name.
    gap_hours : float
        Length of the gap being filled.

    Returns
    -------
    float
        Half-width in the target's units, or NaN if the method was not
        calibrated.
    """
    rows = conformal[conformal['method'] == method]
    if rows.empty:
        return np.nan
    # The nearest calibrated length at or above this gap, else the longest.
    at_or_above = rows[rows['gap_hours'] >= gap_hours]
    pick = (at_or_above.iloc[at_or_above['gap_hours'].argmin()]
            if len(at_or_above)
            else rows.iloc[rows['gap_hours'].argmax()])
    return float(pick['half_width_mdeg'])


def longest_stretch(series, dt_hours=1.0, boundary=None):
    """
    The longest run of consecutive present values.

    Presence is not the same as meaning. A run that spans the instrument
    changeover is continuous in the sense that no slot is empty, and it is *not*
    one series unless the offset has been applied: the per-era column carries a
    step of over a hundred millidegrees at that point. ``boundary`` forbids a run
    from crossing a stated instant, which is how the question "what is available
    without accepting the estimated offset?" is asked.

    Parameters
    ----------
    series : pd.Series
        Series on a regular index.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.
    boundary : str or pd.Timestamp, optional
        Instant a run may not cross. ``None`` places no restriction.

    Returns
    -------
    dict
        Bounds and length of the longest admissible run.
    """
    present = series.notna().to_numpy()
    if not present.any():
        return {'start': None, 'end': None, 'hours': 0.0, 'days': 0.0,
                'annual_cycles': 0.0, 'crosses_boundary': False}

    edges = np.flatnonzero(np.diff(np.r_[0, present.astype(int), 0]))
    starts, stops = edges[::2], edges[1::2]

    if boundary is not None:
        cut = series.index.searchsorted(pd.Timestamp(boundary))
        split_starts, split_stops = [], []
        for a, b in zip(starts, stops):
            if a < cut < b:
                split_starts += [a, cut]
                split_stops += [cut, b]
            else:
                split_starts.append(a)
                split_stops.append(b)
        starts = np.array(split_starts)
        stops = np.array(split_stops)

    k = int(np.argmax(stops - starts))
    hours = float((stops[k] - starts[k]) * dt_hours)
    start, end = series.index[starts[k]], series.index[stops[k] - 1]
    crosses = (boundary is None
               and start < pd.Timestamp(CURRENT_START) <= end)
    return {
        'start': start,
        'end': end,
        'hours': hours,
        'days': round(hours / 24.0, 1),
        'annual_cycles': round(hours / 24.0 / YEAR_DAYS, 2),
        'crosses_boundary': bool(crosses),
    }


def stretch_frontier(df, target, policies, drivers=None, dt_hours=1.0,
                     conformal=None, boundary=None):
    """
    Longest continuous stretch against fabricated fraction, policy by policy.

    The question the study exists to answer, posed as a trade rather than a
    single number: every extra hour of continuous record costs invented data,
    and the frontier shows the exchange rate. A policy that buys a year for one
    per cent fabrication is worth taking; one that buys a week for fifteen is
    not.

    Parameters
    ----------
    df : pd.DataFrame
        Source frame.
    target : str
        Column to fill.
    policies : dict
        Label to policy DataFrame, each from :func:`imputation_policy`.
    drivers : list of str, optional
        Driver columns.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.
    conformal : pd.DataFrame, optional
        Calibration table, so each row carries the mean half-width applied.
    boundary : str or pd.Timestamp, optional
        Instant a stretch may not cross, passed to :func:`longest_stretch`. Use
        it to ask what is available without accepting the estimated instrument
        offset.

    Returns
    -------
    frontier : pd.DataFrame
        One row per policy.
    filled : dict
        Label to the filled frame, for the winner to be used directly.
    """
    rows, filled = [], {}
    base = longest_stretch(df[target], dt_hours=dt_hours, boundary=boundary)

    for label, policy in policies.items():
        out, summary = apply_policy(df, target, policy, drivers=drivers,
                                    dt_hours=dt_hours, conformal=conformal)
        stretch = longest_stretch(out['value'], dt_hours=dt_hours,
                                  boundary=boundary)
        filled[label] = out

        # The fabricated share that matters is the one *inside* the winning
        # stretch. Counting fills elsewhere in the record would understate a
        # policy that bridges one long hole and overstate one that tidies many
        # short ones far away.
        if stretch['start'] is not None:
            inside = out.loc[stretch['start']:stretch['end']]
            inside_filled = float((inside['source'] == 'imputed').sum()
                                  * dt_hours)
            half = inside.loc[inside['source'] == 'imputed', 'half_width']
        else:
            inside_filled, half = 0.0, pd.Series(dtype=float)

        rows.append({
            'policy': label,
            'stretch_days': stretch['days'],
            'annual_cycles': stretch['annual_cycles'],
            'stretch_from': str(stretch['start']),
            'stretch_to': str(stretch['end']),
            'crosses_changeover': stretch['crosses_boundary'],
            'gain_vs_observed_days': round(stretch['days'] - base['days'], 1),
            'filled_in_stretch_h': inside_filled,
            'fabricated_%': round(
                100.0 * inside_filled / max(stretch['hours'], 1.0), 2),
            'filled_total_h': summary['filled_hours'],
            'refused_h': summary['refused_hours'],
            'coverage_after_%': summary['coverage_after_%'],
            'mean_half_width': round(float(half.mean()), 3)
            if len(half) else np.nan,
            'max_half_width': round(float(half.max()), 3)
            if len(half) else np.nan,
        })

    frontier = pd.DataFrame(rows).set_index('policy')
    frontier.attrs['observed_stretch_days'] = base['days']
    frontier.attrs['observed_stretch_from'] = str(base['start'])
    frontier.attrs['observed_stretch_to'] = str(base['end'])
    return frontier, filled


def band_policies(bench, tolerances, bands=None):
    """
    Build one policy per tolerance, for sweeping the frontier.

    Parameters
    ----------
    bench : pd.DataFrame
        Output of :func:`benchmark_fillers`.
    tolerances : iterable of float
        Accepted mean absolute errors, in the target's units.
    bands : list of tuple, optional
        Passed to :func:`imputation_policy`.

    Returns
    -------
    dict
        Label to policy DataFrame.
    """
    return {f'tolerance {t:g} mdeg': imputation_policy(bench, t, bands=bands)
            for t in tolerances}


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────

def plot_station_completeness(frames, column='inc', title='', save_path=None,
                              filename=None):
    """
    Daily availability of one channel, one row per station.

    Parameters
    ----------
    frames : dict
        Station identifier to DataFrame.
    column : str, optional
        Channel to show. Default ``'inc'``.
    title : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    stations = list(frames)
    fig, axes = plt.subplots(len(stations), 1,
                             figsize=figsize(11, 0.9 * len(stations) + 1.4),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, station in zip(axes, stations):
        daily = frames[station][column].notna().resample('1D').mean()
        ax.fill_between(daily.index, 0, daily.values, step='mid',
                        color='#1B6B3A', lw=0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.set_ylabel(station, rotation=0, ha='right', va='center')
    axes[-1].set_xlabel('')
    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)


def plot_completeness_calendar(calendar, title='', save_path=None,
                               filename=None):
    """
    Year-by-day-of-year heat map of daily completeness.

    Parameters
    ----------
    calendar : pd.DataFrame
        Output of :func:`completeness_calendar`.
    title : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize(11, 0.42 * len(calendar) + 1.6))
    sns.heatmap(calendar, cmap='YlGn', vmin=0, vmax=1, ax=ax,
                cbar_kws={'label': 'fraction of the day observed'})
    ax.set_xlabel('day of year')
    ax.set_ylabel('')
    starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    ax.set_xticks(starts)
    ax.set_xticklabels(names, rotation=0)
    if title:
        ax.set_title(title)
    return _finish(fig, save_path, filename)


def plot_channel_availability(avail, title='', save_path=None, filename=None):
    """
    Stacked daily availability of every channel.

    Parameters
    ----------
    avail : pd.DataFrame
        Output of :func:`channel_availability`.
    title : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize(11, 3.2))
    palette = sns.color_palette('colorblind', len(avail.columns))
    for k, col in enumerate(avail.columns):
        ax.fill_between(avail.index, len(avail.columns) - k - 1,
                        len(avail.columns) - k - 1 + avail[col].values,
                        step='mid', color=palette[k], lw=0)
    ax.set_yticks(np.arange(len(avail.columns)) + 0.5)
    ax.set_yticklabels(list(avail.columns)[::-1])
    ax.set_xlabel('')
    if title:
        ax.set_title(title)
    return _finish(fig, save_path, filename)


def plot_gap_anatomy(gaps, bands=None, title='', save_path=None,
                     filename=None):
    """
    Where the missing hours are: by length band, by season, and along the record.

    Parameters
    ----------
    gaps : pd.DataFrame
        Output of :func:`gap_table`.
    bands : pd.DataFrame, optional
        Output of :func:`gap_bands`.
    title : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize(11, 3.4))

    if bands is not None:
        labels = bands.index.tolist()
        sns.barplot(x=bands['share_of_missing_%'].values, y=labels,
                    ax=axes[0], hue=labels, palette='colorblind',
                    legend=False, orient='h')
        for i, n in enumerate(bands['n_gaps']):
            axes[0].text(bands['share_of_missing_%'].iloc[i] + 0.5, i,
                         f'{int(n)}', va='center', fontsize='small')
    axes[0].set_xlabel('share of missing hours [%]')
    axes[0].set_ylabel('')

    by_season = gaps.groupby('season')['hours'].sum().reindex(
        ['DJF', 'MAM', 'JJA', 'SON']).fillna(0)
    sns.barplot(x=by_season.index.tolist(), y=by_season.values, ax=axes[1],
                hue=by_season.index.tolist(), palette='colorblind',
                legend=False)
    axes[1].set_xlabel('season of onset')
    axes[1].set_ylabel('hours lost')

    axes[2].scatter(gaps['start'], gaps['hours'], s=14, alpha=0.65,
                    color='#A5202B')
    axes[2].set_yscale('log')
    axes[2].set_ylabel('gap length [h]')
    axes[2].set_xlabel('')
    axes[2].tick_params(axis='x', rotation=30)

    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)


def plot_filler_benchmark(bench, conformal=None, title='', save_path=None,
                          filename=None):
    """
    Error against gap length per filler, with the calibrated interval beside it.

    Parameters
    ----------
    bench : pd.DataFrame
        Output of :func:`benchmark_fillers`.
    conformal : pd.DataFrame, optional
        Output of :func:`conformal_bands`.
    title : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_panels = 2 if conformal is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=figsize(11, 3.4),
                             squeeze=False)
    palette = dict(zip(sorted(bench['method'].unique()),
                       sns.color_palette('colorblind',
                                         bench['method'].nunique())))

    ax = axes[0, 0]
    for name, grp in bench.groupby('method'):
        grp = grp.sort_values('gap_hours')
        ax.plot(grp['gap_hours'], grp['MAE_mdeg'], marker='o', ms=4, lw=1.3,
                color=palette[name], label=name)
    ax.set_xscale('log')
    ax.set_xlabel('gap length [h]')
    ax.set_ylabel('MAE [mdeg]')
    ax.legend(fontsize='small')

    if conformal is not None:
        ax = axes[0, 1]
        for name, grp in conformal.groupby('method'):
            grp = grp.sort_values('gap_hours')
            ax.plot(grp['gap_hours'], grp['half_width_mdeg'], marker='s',
                    ms=4, lw=1.3, color=palette.get(name), label=name)
        ax.set_xscale('log')
        ax.set_xlabel('gap length [h]')
        ax.set_ylabel(f'{conformal["nominal_%"].iloc[0]:.0f} % half-width [mdeg]')

    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)


def plot_stretch_frontier(frontier, title='', save_path=None, filename=None):
    """
    Continuous stretch bought against fabricated fraction paid.

    Parameters
    ----------
    frontier : pd.DataFrame
        Output of :func:`stretch_frontier`.
    title : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize(11, 3.4))
    labels = frontier.index.tolist()
    palette = sns.color_palette('colorblind', len(labels))

    axes[0].barh(labels, frontier['stretch_days'].values, color=palette)
    base = frontier.attrs.get('observed_stretch_days')
    if base:
        axes[0].axvline(base, color='0.3', ls='--', lw=1.0)
        axes[0].text(base, -0.6, f' observed: {base:.0f} d', fontsize='small',
                     color='0.3')
    axes[0].set_xlabel('longest continuous stretch [days]')
    for i, c in enumerate(frontier['annual_cycles']):
        axes[0].text(frontier['stretch_days'].iloc[i], i, f'  {c:.2f} cyc',
                     va='center', fontsize='small')

    axes[1].scatter(frontier['fabricated_%'], frontier['stretch_days'],
                    s=60, color=palette)
    for label, row in frontier.iterrows():
        axes[1].annotate(label.replace('tolerance ', ''),
                         (row['fabricated_%'], row['stretch_days']),
                         fontsize='small', xytext=(4, 4),
                         textcoords='offset points')
    axes[1].set_xlabel('fabricated share of the stretch [%]')
    axes[1].set_ylabel('stretch [days]')

    for ax in axes:
        ax.set_ylabel(ax.get_ylabel())
    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)
