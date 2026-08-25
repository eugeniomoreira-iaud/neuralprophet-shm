"""
Module: tc_lib.py

Support library for the thermal-compensation study
(``studies/thermal_compensation/``).

The study asks one question: does the temperature compensation currently
implemented in Notebook 00 — a fixed coefficient applied instantaneously to the
station's own air temperature — help or hurt a prediction system built on the
inclinometer signal?

Everything here is deliberately self-contained. The study reads the raw archive
directly rather than going through ``heritageshm.dataloader``, because that
loader assumes the 14-column legacy era and cannot read the 20-column files that
carry the solar-radiation and wall-temperature channels this study needs. The
parsing rules implemented in :func:`load_archive` follow the contract documented
in ``docs/raw-data-format.md``.

Nothing in this module writes to the raw archive.
"""

import os
import re
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# ──────────────────────────────────────────────────────────────────────
# Archive constants — see docs/raw-data-format.md
# ──────────────────────────────────────────────────────────────────────

#: Number of fields in a current-era record (2 timestamp + 18 measurement).
N_COLS_CURRENT = 20

#: Number of fields in a legacy-era record (2 timestamp + 12 measurement).
N_COLS_LEGACY = 14

#: Date on which the 20-column instrument package replaced the legacy network.
CHANGEOVER = pd.Timestamp('2025-02-21')

#: Conditioner output zero point, in the millivolt units the logger records.
#: Sensitivity is 1000 mV per degree, so millivolts and millidegrees share a
#: scale and only this offset separates the logged number from a true angle.
INCLINOMETER_ZERO_MV = 2500.0

#: Full-scale range of the calibrated inclinometer, in millidegrees.
INCLINOMETER_RANGE_MDEG = 2000.0

#: Wall-temperature probe failure sentinel.
TWALL_SENTINEL = -55.0

#: Physical ceiling for incident solar radiation at this latitude, in W/m².
#: Records above this are the unsigned wrap-around artefact, not measurements.
SR_MAX_PHYSICAL = 1400.0

#: Nominal sampling interval of the raw archive.
SAMPLING = '20min'

#: Analysis sampling interval. Every environmental proxy the pipeline aligns
#: against — ERA5-Land and the other reanalysis products — is published hourly,
#: so the study aggregates to one hour immediately after parsing and every
#: correlation, slope, lag and forecast below is computed on that grid. Working
#: at the native 20-minute rate and resampling later would make the study's
#: conclusions depend on a resolution the production pipeline never sees.
ANALYSIS_FREQ = '1h'

#: Minimum number of valid raw samples required to accept an aggregated hour.
#: The raw rate is three samples per hour; requiring two keeps an hour that lost
#: a single record and rejects one built from an isolated reading.
MIN_SAMPLES_PER_HOUR = 2

#: Current-era column names, in file order. Fields 3-14 are the three legacy
#: blocks, which are identically zero in this era and are dropped after parsing.
CURRENT_COLUMNS = [
    'b1_batt', 'b1_tair', 'b1_rh', 'b1_i',
    'b2_batt', 'b2_tair', 'b2_rh', 'b2_i',
    'b3_batt', 'b3_tair', 'b3_rh', 'b3_i',
    'batt', 'tair', 'rh', 'i_mv', 'sr', 'twall',
]

#: Channels in which 0.000 is a missing-data sentinel rather than a reading.
#: Solar radiation is deliberately absent: its night-time zeros are real.
ZERO_IS_SENTINEL = ['batt', 'tair', 'rh', 'i_mv']

#: The compensation coefficient hard-coded in Notebook 00, in
#: mdeg · °C⁻¹ · 10⁻³. Multiplied by 1000 it becomes 5 mdeg per °C.
DOCUMENTED_COEFF = 0.005


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────

def _to_float(token):
    """
    Parse one numeric field, accepting either decimal separator.

    The archive mixes ``3.500`` and ``3,500`` forms, sometimes within a single
    file, so the separator is normalised per field rather than per file.

    Parameters
    ----------
    token : str
        Raw field text.

    Returns
    -------
    float
        Parsed value, or ``nan`` if the field is not numeric.
    """
    try:
        return float(token.replace(',', '.'))
    except (ValueError, AttributeError):
        return np.nan


def sync_cache(archive_dir, cache_dir, start, end, verbose=True,
               progress=False):
    """
    Copy the ``.adc`` files covering a date range to a local cache.

    The archive is served by Google Drive streaming at roughly ten files per
    minute, which makes repeated in-place reads impractical. This copies each
    required file once; subsequent runs reuse the cache. The archive itself is
    only ever read.

    Parameters
    ----------
    archive_dir : str
        Path to the read-only ``.adc`` archive.
    cache_dir : str
        Local directory for the working copy. Created if absent.
    start, end : str or pd.Timestamp
        Inclusive date bounds.
    verbose : bool, optional
        Print a one-line summary. Default ``True``.
    progress : bool, optional
        Draw a progress bar over the files that still have to be copied.
        Default ``False``. Worth enabling on a cold cache: copying is the one
        slow step here, at roughly ten files per minute, while a warm cache
        returns immediately and draws no bar at all.

    Returns
    -------
    list of str
        Absolute paths of the cached files, sorted by date.
    """
    os.makedirs(cache_dir, exist_ok=True)
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    wanted = []
    for name in sorted(os.listdir(archive_dir)):
        m = re.fullmatch(r'GUBBIO_(\d{8})\.adc', name)
        if not m:
            continue
        day = pd.Timestamp(m.group(1))
        if start <= day <= end:
            wanted.append(name)

    missing = [n for n in wanted
               if not os.path.exists(os.path.join(cache_dir, n))]
    copied = len(missing)

    pending = missing
    if progress and missing:
        # Imported here rather than at module scope so that the four studies
        # importing this library keep working if tqdm is ever absent.
        from tqdm.auto import tqdm
        pending = tqdm(missing, desc='  caching', unit='file')

    for name in pending:
        shutil.copy2(os.path.join(archive_dir, name),
                     os.path.join(cache_dir, name))

    if verbose:
        print(f'  cache: {len(wanted)} files in range, {copied} newly copied, '
              f'{len(wanted) - copied} already present')
    return [os.path.join(cache_dir, n) for n in wanted]


def parse_file(path):
    """
    Parse one ``.adc`` file into records, applying the documented contract.

    Handles the mixed decimal separator, takes the date from the record rather
    than from the filename (files bleed one record into the previous day), and
    branches on the record's own field count so that a file whose content
    disagrees with its expected era is skipped rather than mangled.

    Parameters
    ----------
    path : str
        Path to a ``.adc`` file.

    Returns
    -------
    pd.DataFrame
        Indexed by timestamp, columns as :data:`CURRENT_COLUMNS`. Empty if the
        file holds no current-era records.
    """
    rows, stamps = [], []
    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != N_COLS_CURRENT:
                continue
            try:
                ts = pd.to_datetime(parts[0] + ' ' + parts[1],
                                    format='%d/%m/%y %H:%M:%S')
            except ValueError:
                continue
            stamps.append(ts)
            rows.append([_to_float(p) for p in parts[2:]])

    if not rows:
        return pd.DataFrame(columns=CURRENT_COLUMNS)

    return pd.DataFrame(rows, index=pd.DatetimeIndex(stamps),
                        columns=CURRENT_COLUMNS)


def load_archive(archive_dir, cache_dir, start, end, verbose=True,
                 freq=ANALYSIS_FREQ):
    """
    Load the current-era station 02 record over a date range.

    Applies, in order: parsing per :func:`parse_file`; field-wise merging of
    duplicate timestamps; sentinel conversion; conversion of the inclinometer
    column from conditioner millivolts to millidegrees referred to the
    instrument's calibrated zero; reindexing onto the regular 20-minute grid
    with absent slots present as ``NaN``; and aggregation to ``freq``.

    Duplicate timestamps are merged by taking the first non-null value in each
    field, never by dropping whole records, because conflicting copies of one
    timestamp routinely differ in which block carries real values.

    Parameters
    ----------
    archive_dir : str
        Read-only ``.adc`` archive.
    cache_dir : str
        Local working copy directory.
    start, end : str or pd.Timestamp
        Inclusive date bounds.
    verbose : bool, optional
        Print a loading report. Default ``True``.
    freq : str or None, optional
        Aggregation interval, applied by :func:`to_hourly`. Default
        :data:`ANALYSIS_FREQ`. Pass ``None`` to obtain the native 20-minute
        record, which is needed only for sub-hourly lag checks.

    Returns
    -------
    pd.DataFrame
        Columns ``batt``, ``tair``, ``rh``, ``inc``, ``sr``, ``twall`` on a
        regular index at ``freq``. ``inc`` is in millidegrees.
    """
    paths = sync_cache(archive_dir, cache_dir, start, end, verbose=verbose)

    frames = [parse_file(p) for p in paths]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise ValueError('No current-era records found in the requested range.')
    raw = pd.concat(frames).sort_index()
    n_records = len(raw)

    # Duplicate timestamps: count genuine conflicts, then merge field-wise.
    dup_mask = raw.index.duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    conflicts = 0
    if n_dup:
        grouped = raw[dup_mask].groupby(level=0)
        conflicts = int(grouped.nunique().gt(1).sum().sum())
    merged = raw.groupby(level=0).first()

    # Keep only the current station package; the legacy blocks are all zero.
    df = merged[['batt', 'tair', 'rh', 'i_mv', 'sr', 'twall']].copy()

    # Sentinels.
    n_zero = 0
    for col in ZERO_IS_SENTINEL:
        if col == 'i_mv':
            hit = df['i_mv'] == 0.0
            n_zero += int(hit.sum())
            df.loc[hit, 'i_mv'] = np.nan
        else:
            hit = df[col] == 0.0
            n_zero += int(hit.sum())
            df.loc[hit, col] = np.nan

    n_twall_bad = int((df['twall'] == TWALL_SENTINEL).sum())
    df.loc[df['twall'] == TWALL_SENTINEL, 'twall'] = np.nan

    n_sr_bad = int((df['sr'] > SR_MAX_PHYSICAL).sum())
    df.loc[df['sr'] > SR_MAX_PHYSICAL, 'sr'] = np.nan

    # Millivolts to millidegrees about the calibrated zero.
    df['inc'] = df['i_mv'] - INCLINOMETER_ZERO_MV
    df = df.drop(columns=['i_mv'])

    # Out-of-range inclinometer values cannot be measurements: the certificates
    # cover -2 to +2 degrees and the conditioner cannot represent more.
    n_range = int((df['inc'].abs() > INCLINOMETER_RANGE_MDEG).sum())
    df.loc[df['inc'].abs() > INCLINOMETER_RANGE_MDEG, 'inc'] = np.nan

    # Regular grid.
    grid = pd.date_range(df.index.min().floor('D'),
                         df.index.max().ceil('D'), freq=SAMPLING)
    df = df.reindex(grid)
    df.index.name = 'datetime'

    if verbose:
        print(f'  parsed  : {n_records} records from {len(paths)} files')
        print(f'  duplicates: {n_dup} records share a timestamp, '
              f'{conflicts} field-level conflicts merged')
        print(f'  sentinels : {n_zero} zeros, {n_twall_bad} Twall = -55, '
              f'{n_sr_bad} SR > {SR_MAX_PHYSICAL:.0f} W/m2, '
              f'{n_range} inclinometer out of range')
        print(f'  grid      : {len(df)} slots from {df.index.min()} '
              f'to {df.index.max()}')
    if freq:
        df = to_hourly(df, freq=freq, verbose=verbose)
    return df


def to_hourly(df, freq=ANALYSIS_FREQ, min_count=MIN_SAMPLES_PER_HOUR,
              verbose=True):
    """
    Aggregate the native record onto the analysis grid.

    Every channel is averaged over the interval. Averaging is the correct
    aggregation for all of them: the temperatures, humidity and inclination are
    instantaneous states whose hourly mean is the natural hourly value, and
    solar radiation is a flux whose hourly mean is the quantity reanalysis
    products publish. Taking a single sample per hour instead would keep the
    20-minute noise and throw away two thirds of the information.

    An interval is kept only if at least ``min_count`` raw samples in it were
    valid, so that an hour reconstructed from one surviving reading is not
    presented as equivalent to a complete one. Intervals below the threshold
    become ``NaN`` and remain visible to the gap analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Native-resolution record on a regular index.
    freq : str, optional
        Target interval. Default :data:`ANALYSIS_FREQ`.
    min_count : int, optional
        Minimum valid raw samples per interval. Default
        :data:`MIN_SAMPLES_PER_HOUR`.
    verbose : bool, optional
        Print an aggregation report. Default ``True``.

    Returns
    -------
    pd.DataFrame
        Aggregated record on a regular index at ``freq``.
    """
    grouped = df.resample(freq)
    means = grouped.mean()
    counts = grouped.count()
    out = means.where(counts >= min_count)
    if verbose:
        kept = int(out['inc'].notna().sum()) if 'inc' in out else 0
        thin = int(((counts['inc'] > 0) & (counts['inc'] < min_count)).sum()) \
            if 'inc' in counts else 0
        print(f'  resampled : {len(df)} native slots to {len(out)} at {freq}; '
              f'{kept} intervals carry the inclinometer, {thin} rejected for '
              f'holding fewer than {min_count} raw samples')
    return out


def sampling_hours(df):
    """
    Hours per step of a regular index, inferred from the data.

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Object with a regular ``DatetimeIndex``.

    Returns
    -------
    float
        Interval length in hours.
    """
    idx = df.index
    if len(idx) < 2:
        raise ValueError('cannot infer a sampling interval from fewer than '
                         'two timestamps')
    return (idx[1] - idx[0]).total_seconds() / 3600.0


def screen_spikes(df, col='inc', window=3, k=8.0, verbose=True):
    """
    Remove isolated outliers with a rolling-median context filter.

    This mirrors the filter in ``heritageshm.preprocessing.clean_signal_robust``
    — deviation from the local median of the surrounding samples — but sets the
    threshold from the data rather than from a fixed constant, so that it
    transfers between the two eras without retuning.

    The scale is the robust spread of **the signal itself**, not of the
    local-median residual. That distinction is load-bearing. This signal is very
    smooth at any sampling rate the archive supports, so the residual from a
    centred median has a robust sigma of well under a millidegree; a multiple of
    *that* would flag most of the legitimate diurnal variation. The robust
    spread of the series is two orders of magnitude larger and separates genuine
    excursions from ordinary movement.

    The filter is deliberately a gross-outlier screen and nothing more. Its
    purpose here is to remove the instrument-settling transients that follow a
    long outage — values more than a thousand millidegrees outside the working
    band — because a single such sample would dominate every variance
    comparison in this study.

    Parameters
    ----------
    df : pd.DataFrame
        Source data with a regular index.
    col : str, optional
        Column to screen. Default ``'inc'``.
    window : int, optional
        Half-width of the centred median window, in samples. Default ``3``,
        which is three hours either side on the hourly analysis grid.
    k : float, optional
        Threshold as a multiple of the signal's robust standard deviation.
        Default ``8.0``.
    verbose : bool, optional
        Print how much was removed. Default ``True``.

    Returns
    -------
    out : pd.DataFrame
        Copy of ``df`` with flagged samples of ``col`` set to ``NaN``.
    flagged : pd.Series
        Boolean mask of the removed samples.
    """
    out = df.copy()
    s = out[col]
    med = s.rolling(2 * window + 1, center=True, min_periods=3).median()
    mad = (s - s.median()).abs().median()
    sigma = 1.4826 * mad if mad > 0 else s.std()
    threshold = k * sigma
    flagged = (s - med).abs() > threshold
    out.loc[flagged, col] = np.nan
    if verbose:
        print(f'  spike screen on "{col}": robust sigma of signal '
              f'{sigma:.2f} mdeg, threshold {threshold:.1f} mdeg, '
              f'{int(flagged.sum())} of {int(s.notna().sum())} samples removed '
              f'({100 * flagged.sum() / max(s.notna().sum(), 1):.2f} %)')
    return out, flagged


def coverage_table(df):
    """
    Summarise per-channel availability on the regular grid.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_archive`.

    Returns
    -------
    pd.DataFrame
        One row per channel with counts, availability, and observed range.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            'channel': col,
            'present': int(s.notna().sum()),
            'missing': int(s.isna().sum()),
            'available_%': 100 * s.notna().mean(),
            'min': s.min(),
            'max': s.max(),
            'mean': s.mean(),
            'std': s.std(),
        })
    return pd.DataFrame(rows).set_index('channel').round(3)


def valid_windows(df, cols, min_days=5):
    """
    Find contiguous runs where every requested channel is available.

    A run is broken by any grid slot in which one of ``cols`` is missing, then
    runs shorter than ``min_days`` are discarded. This is what defines the
    period on which the study can be run at all: the comparison needs the
    inclinometer, both temperatures and solar radiation simultaneously.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_archive`.
    cols : list of str
        Channels that must all be present.
    min_days : int, optional
        Shortest run to report. Default ``5``.

    Returns
    -------
    pd.DataFrame
        One row per run with start, end, duration and sample count.
    """
    ok = df[cols].notna().all(axis=1)
    grp = (~ok).cumsum()[ok]
    rows = []
    for _, idx in ok[ok].groupby(grp):
        block = idx.index
        days = (block[-1] - block[0]).total_seconds() / 86400
        if days >= min_days:
            rows.append({'start': block[0], 'end': block[-1],
                         'days': round(days, 1), 'samples': len(block)})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Compensation
# ──────────────────────────────────────────────────────────────────────

def compensate(df, temp_col='tair', coeff=DOCUMENTED_COEFF, normalise=True):
    """
    Apply the compensation exactly as Notebook 00 implements it.

    The correction is ``inc - (T - T_ref) * coeff * 1000`` with ``T_ref`` taken
    from the first record carrying both a signal and a temperature, followed
    optionally by a shift so that the series starts at zero.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``inc`` and ``temp_col``.
    temp_col : str, optional
        Temperature channel driving the correction. Default ``'tair'``.
    coeff : float, optional
        Coefficient in mdeg · °C⁻¹ · 10⁻³. Default :data:`DOCUMENTED_COEFF`.
    normalise : bool, optional
        Subtract the first valid value. Default ``True``.

    Returns
    -------
    pd.Series
        Compensated signal, same index as ``df``.
    """
    valid = df[['inc', temp_col]].dropna()
    if valid.empty:
        return pd.Series(np.nan, index=df.index, name='inc_comp')
    t_ref = valid[temp_col].iloc[0]
    out = df['inc'] - (df[temp_col] - t_ref) * coeff * 1000.0
    if normalise:
        first = out.dropna()
        if len(first):
            out = out - first.iloc[0]
    return out.rename('inc_comp')


def compensation_term(df, temp_col='tair', coeff=DOCUMENTED_COEFF):
    """
    Return the quantity that :func:`compensate` subtracts, without the shift.

    Needed to score predictions of a compensated target on the raw scale.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``temp_col``.
    temp_col : str, optional
        Temperature channel. Default ``'tair'``.
    coeff : float, optional
        Coefficient in mdeg · °C⁻¹ · 10⁻³. Default :data:`DOCUMENTED_COEFF`.

    Returns
    -------
    pd.Series
        The correction term in millidegrees.
    """
    valid = df[['inc', temp_col]].dropna()
    t_ref = valid[temp_col].iloc[0] if len(valid) else np.nan
    return ((df[temp_col] - t_ref) * coeff * 1000.0).rename('comp_term')


# ──────────────────────────────────────────────────────────────────────
# Correlation and lag structure
# ──────────────────────────────────────────────────────────────────────

def correlation_matrix(df, cols, method='pearson', differenced=False):
    """
    Correlation matrix on levels or on first differences.

    Levels correlations between two trending series are inflated by the shared
    trend and say little about a driver-response relationship. The differenced
    matrix is the honest one for a signal with drift, and both are reported so
    the difference between them is visible.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    cols : list of str
        Columns to include.
    method : str, optional
        ``'pearson'`` or ``'spearman'``. Default ``'pearson'``.
    differenced : bool, optional
        Correlate one-step differences instead of levels. Default ``False``.

    Returns
    -------
    pd.DataFrame
        Square correlation matrix.
    """
    data = df[cols].diff() if differenced else df[cols]
    return data.corr(method=method)


def cross_correlation(df, target, drivers, max_lag_steps, differenced=True):
    """
    Cross-correlation of a target against several drivers over a lag range.

    A positive lag means the driver leads the target: the value of the driver
    ``k`` steps earlier is correlated with the target now. A thermal response
    with inertia peaks at a positive lag, which is precisely what an
    instantaneous correction cannot represent.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    target : str
        Response column.
    drivers : list of str
        Candidate driver columns.
    max_lag_steps : int
        Largest lag, in sampling intervals, evaluated in both directions.
    differenced : bool, optional
        Operate on first differences. Default ``True``.

    Returns
    -------
    pd.DataFrame
        Index of lags in steps, one column per driver.
    """
    data = df.diff() if differenced else df
    y = data[target]
    lags = range(-max_lag_steps, max_lag_steps + 1)
    out = {}
    for d in drivers:
        out[d] = [y.corr(data[d].shift(k)) for k in lags]
    res = pd.DataFrame(out, index=pd.Index(lags, name='lag_steps'))
    return res


def peak_lags(ccf, sampling_hours=1.0, match_sign=False, positive_only=False):
    """
    Extract the strongest lag for each driver from a cross-correlation table.

    On levels, both series carry strong daily and seasonal cycles, and the
    cross-correlation of two near-periodic signals is itself near-periodic: the
    largest *absolute* correlation is then just as likely to sit at an
    anti-phase lag half a cycle away as at the physical one. Selecting the
    extremum blindly reports that anti-phase lag as the response time of the
    structure, which is nonsense. ``match_sign`` restricts the search to lags
    whose correlation has the same sign as the contemporaneous one, and
    ``positive_only`` restricts it to lags at which the driver leads, which is
    the only direction a forcing can act in.

    Parameters
    ----------
    ccf : pd.DataFrame
        Output of :func:`cross_correlation`.
    sampling_hours : float, optional
        Hours per step. Default ``1.0`` for the hourly analysis grid; pass
        ``1/3`` when scanning the native 20-minute record.
    match_sign : bool, optional
        Consider only lags whose correlation shares the sign of the lag-0
        correlation. Default ``False``.
    positive_only : bool, optional
        Consider only lags at which the driver leads the response. Default
        ``False``.

    Returns
    -------
    pd.DataFrame
        Peak correlation, its lag in steps and in hours, and the correlation at
        zero lag for comparison.
    """
    rows = []
    for col in ccf.columns:
        s = ccf[col]
        candidate = s
        if positive_only:
            candidate = candidate.loc[candidate.index >= 0]
        if match_sign:
            sign = np.sign(s.loc[0])
            if sign != 0:
                same = candidate[np.sign(candidate) == sign]
                if len(same):
                    candidate = same
        k = candidate.abs().idxmax()
        rows.append({
            'driver': col,
            'peak_corr': round(s.loc[k], 4),
            'peak_lag_steps': int(k),
            'peak_lag_hours': round(k * sampling_hours, 2),
            'corr_at_lag0': round(s.loc[0], 4),
        })
    return pd.DataFrame(rows).set_index('driver')


# ──────────────────────────────────────────────────────────────────────
# Thermal inertia: is the response instantaneous?
# ──────────────────────────────────────────────────────────────────────
#
# The correction in Notebook 00 assumes the structure responds to air
# temperature at the instant it is measured. The physical hypothesis says
# otherwise: solar radiation heats the exposed face of the wall, the heat
# diffuses into a large thermal mass, and the resulting differential expansion
# tilts the structure. A body with thermal mass cannot respond instantaneously
# to anything, so the assumption is a property of the correction, not of the
# wall, and it has to be tested rather than assumed.
#
# Two tests are provided. The first is empirical: shift each driver and see
# which shift explains the signal best. The second is physical: filter each
# driver through a first-order thermal lag before fitting, and see which time
# constant explains the signal best. A single-pole low-pass is the standard
# lumped-capacitance model of a body exchanging heat with its surroundings,
# and its step response is the familiar 1 - exp(-t/tau). Its output is the
# effective temperature of the mass, which is what an expansion actually
# follows.


def thermal_lag_filter(series, tau_hours, dt_hours=1.0):
    """
    Apply a first-order thermal lag to a driver series.

    Implements the lumped-capacitance response of a body with time constant
    ``tau_hours`` to the forcing in ``series``, discretised as a one-pole
    exponential filter with smoothing factor ``alpha = dt / (tau + dt)``. With
    ``tau_hours = 0`` the driver is returned unchanged, which makes the
    instantaneous assumption the zero-inertia special case of this family
    rather than a separate model.

    Gaps are bridged for the purpose of running the filter and restored
    afterwards, so that a missing hour does not silently reset the state or
    propagate a fill into the output.

    Parameters
    ----------
    series : pd.Series
        Driver on a regular index.
    tau_hours : float
        Thermal time constant in hours. ``0`` disables the filter.
    dt_hours : float, optional
        Sampling interval in hours. Default ``1.0``.

    Returns
    -------
    pd.Series
        Filtered driver, ``NaN`` wherever the input was ``NaN``.
    """
    if tau_hours <= 0:
        return series.copy()
    alpha = dt_hours / (tau_hours + dt_hours)
    filled = series.interpolate(method='time', limit_direction='both')
    out = filled.ewm(alpha=alpha, adjust=False).mean()
    return out.where(series.notna())


def inertia_scan(df, target, drivers, taus, dt_hours=None,
                 detrend_hours=None):
    """
    Scan thermal time constants and report how well each explains the signal.

    For every driver and every candidate ``tau`` the driver is filtered by
    :func:`thermal_lag_filter` and regressed on the target. The reported
    quantities are the ordinary-least-squares slope, the coefficient of
    determination and the residual variance, all computed on levels, because
    the question is whether the *state* of the structure follows the effective
    temperature of its mass.

    Parameters
    ----------
    df : pd.DataFrame
        Source data on a regular index.
    target : str
        Response column.
    drivers : list of str
        Candidate driver columns.
    taus : sequence of float
        Time constants to evaluate, in hours. Include ``0`` to keep the
        instantaneous case in the comparison.
    dt_hours : float or None, optional
        Sampling interval. Inferred from the index when ``None``.
    detrend_hours : float or None, optional
        When given, subtract a centred rolling mean of this width from both the
        target and the filtered driver before fitting, which restricts the
        comparison to variation faster than that width. A window that spans a
        season is dominated by the seasonal march of temperature, and a fit on
        it can look excellent while saying nothing about how quickly the
        structure answers a change in forcing. Running the scan twice — once
        unfiltered, once band-limited — separates the two timescales instead of
        letting the slower one decide the answer.

    Returns
    -------
    pd.DataFrame
        One row per driver and tau, with columns ``driver``, ``tau_hours``,
        ``slope``, ``r2`` and ``residual_var``.
    """
    if dt_hours is None:
        dt_hours = sampling_hours(df)
    window = None
    if detrend_hours:
        window = max(int(round(detrend_hours / dt_hours)), 3)

    def _band(s):
        if window is None:
            return s
        return s - s.rolling(window, center=True,
                             min_periods=max(3, window // 4)).mean()

    rows = []
    y_all = _band(df[target])
    for d in drivers:
        for tau in taus:
            x = _band(thermal_lag_filter(df[d], tau, dt_hours))
            both = pd.concat([y_all, x], axis=1).dropna()
            if len(both) < 100:
                continue
            y, xx = both.iloc[:, 0], both.iloc[:, 1]
            if xx.std() == 0:
                continue
            slope = float(np.polyfit(xx, y, 1)[0])
            resid = y - (slope * xx + (y.mean() - slope * xx.mean()))
            rows.append({
                'driver': d,
                'tau_hours': float(tau),
                'slope': round(slope, 4),
                'r2': round(float(1 - resid.var() / y.var()), 4),
                'residual_var': round(float(resid.var()), 3),
                'n': len(both),
            })
    return pd.DataFrame(rows)


def best_inertia(scan):
    """
    Pick the best time constant per driver from an :func:`inertia_scan` table.

    Parameters
    ----------
    scan : pd.DataFrame
        Output of :func:`inertia_scan`.

    Returns
    -------
    pd.DataFrame
        One row per driver: the winning ``tau``, its slope and ``r2``, the
        ``r2`` of the instantaneous case, and the improvement between them.
    """
    rows = []
    for d, g in scan.groupby('driver', sort=False):
        best = g.loc[g['r2'].idxmax()]
        zero = g.loc[g['tau_hours'].abs().idxmin()]
        rows.append({
            'driver': d,
            'best_tau_hours': best['tau_hours'],
            'slope_at_best': best['slope'],
            'r2_at_best': best['r2'],
            'r2_instantaneous': zero['r2'],
            'r2_gain': round(float(best['r2'] - zero['r2']), 4),
        })
    return pd.DataFrame(rows).set_index('driver')


def add_lagged(df, spec, suffix='lag'):
    """
    Materialise lagged copies of drivers as new columns.

    Adding the shifted series to the frame, rather than teaching the model
    builders about lags, keeps every downstream experiment — ridge and
    NeuralProphet alike — working with plain column names.

    Parameters
    ----------
    df : pd.DataFrame
        Frame to extend, modified in place and returned.
    spec : dict
        Mapping of column name to lag in steps. A positive lag means the driver
        leads: the value from ``k`` steps ago is aligned with the present.
    suffix : str, optional
        Name fragment for the created columns. Default ``'lag'``.

    Returns
    -------
    names : list of str
        Names of the created columns, in the order of ``spec``.
    """
    names = []
    for col, k in spec.items():
        name = f'{col}_{suffix}{int(k)}'
        df[name] = df[col].shift(int(k))
        names.append(name)
    return names


def add_inertia(df, spec, dt_hours=None, suffix='tau'):
    """
    Materialise thermal-lag-filtered copies of drivers as new columns.

    Parameters
    ----------
    df : pd.DataFrame
        Frame to extend, modified in place and returned.
    spec : dict
        Mapping of column name to time constant in hours.
    dt_hours : float or None, optional
        Sampling interval. Inferred from the index when ``None``.
    suffix : str, optional
        Name fragment for the created columns. Default ``'tau'``.

    Returns
    -------
    names : list of str
        Names of the created columns, in the order of ``spec``.
    """
    if dt_hours is None:
        dt_hours = sampling_hours(df)
    names = []
    for col, tau in spec.items():
        name = f'{col}_{suffix}{int(round(tau))}'
        df[name] = thermal_lag_filter(df[col], tau, dt_hours)
        names.append(name)
    return names


def fit_slope(df, target, driver, differenced=False, hac_lags=24):
    """
    Regress a target on one driver and return the slope with robust errors.

    Standard errors use a heteroskedasticity- and autocorrelation-consistent
    estimator, because an hourly structural series is strongly autocorrelated
    and ordinary standard errors would be far too small.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    target : str
        Response column.
    driver : str
        Explanatory column.
    differenced : bool, optional
        Fit on first differences. Default ``False``.
    hac_lags : int, optional
        Newey-West lag truncation. Default ``24``, one day on the hourly
        analysis grid.

    Returns
    -------
    dict
        Slope, its confidence interval and standard error, R², and sample size.
    """
    data = (df[[target, driver]].diff() if differenced
            else df[[target, driver]]).dropna()
    if len(data) < 10:
        return {'driver': driver, 'slope': np.nan, 'ci_low': np.nan,
                'ci_high': np.nan, 'se': np.nan, 'r2': np.nan, 'n': len(data)}
    X = sm.add_constant(data[driver].values)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = sm.OLS(data[target].values, X).fit(
            cov_type='HAC', cov_kwds={'maxlags': hac_lags})
    ci = res.conf_int()[1]
    return {
        'driver': driver,
        'slope': round(res.params[1], 4),
        'ci_low': round(ci[0], 4),
        'ci_high': round(ci[1], 4),
        'se': round(res.bse[1], 4),
        'r2': round(res.rsquared, 4),
        'n': len(data),
    }


def slope_table(df, target, drivers, differenced=False, hac_lags=24):
    """
    Run :func:`fit_slope` for several drivers and collect the results.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    target : str
        Response column.
    drivers : list of str
        Explanatory columns, fitted one at a time.
    differenced : bool, optional
        Fit on first differences. Default ``False``.
    hac_lags : int, optional
        Newey-West lag truncation. Default ``24``.

    Returns
    -------
    pd.DataFrame
        One row per driver.
    """
    return pd.DataFrame(
        [fit_slope(df, target, d, differenced, hac_lags) for d in drivers]
    ).set_index('driver')


def multivariate_fit(df, target, drivers, hac_lags=24):
    """
    Fit the target on several drivers jointly.

    Answers whether air temperature is the right single explanatory variable
    once wall temperature and solar radiation are allowed to compete for the
    same variance.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    target : str
        Response column.
    drivers : list of str
        Explanatory columns, entered together.
    hac_lags : int, optional
        Newey-West lag truncation. Default ``24``.

    Returns
    -------
    statsmodels results object
    """
    data = df[[target] + drivers].dropna()
    X = sm.add_constant(data[drivers])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return sm.OLS(data[target], X).fit(
            cov_type='HAC', cov_kwds={'maxlags': hac_lags})


# ──────────────────────────────────────────────────────────────────────
# Acceptance tests for the compensation
# ──────────────────────────────────────────────────────────────────────

def diurnal_amplitude(series):
    """
    Mean daily peak-to-peak range of a series.

    A thermally driven signal has a large daily swing. If compensation is
    working, the swing shrinks.

    Parameters
    ----------
    series : pd.Series
        Signal on a datetime index.

    Returns
    -------
    float
        Mean of the daily maximum minus daily minimum, over days with data.
    """
    daily = series.groupby(series.index.date)
    amp = daily.max() - daily.min()
    return float(amp.mean())


def compensation_tests(df, comp_series, temp_cols=('tair', 'twall'),
                       extra_drivers=('sr',)):
    """
    Evaluate whether the compensation achieves what it is meant to achieve.

    The stated purpose of the correction is to remove the thermal component
    from the inclinometer signal. That gives three falsifiable predictions,
    each checked here on the raw and compensated series:

    - the variance of the signal should fall;
    - the correlation with temperature should move towards zero;
    - the daily peak-to-peak amplitude should fall.

    A correlation that changes sign is reported explicitly, because it means
    the correction has overshot: it removed more than the thermal component and
    has injected an inverted copy of the temperature signal.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``inc`` and the temperature columns.
    comp_series : pd.Series
        Compensated signal, aligned to ``df``.
    temp_cols : tuple of str, optional
        Temperature channels to test against. Default ``('tair', 'twall')``.
    extra_drivers : tuple of str, optional
        Further channels to report. Default ``('sr',)``.

    Returns
    -------
    pd.DataFrame
        One row per test with raw and compensated values and a verdict.
    """
    raw = df['inc']
    comp = comp_series.reindex(df.index)
    rows = []

    rows.append({
        'test': 'variance [mdeg^2]',
        'raw': round(float(raw.var()), 3),
        'compensated': round(float(comp.var()), 3),
        'target': 'lower',
    })
    rows.append({
        'test': 'std [mdeg]',
        'raw': round(float(raw.std()), 3),
        'compensated': round(float(comp.std()), 3),
        'target': 'lower',
    })
    rows.append({
        'test': 'mean diurnal amplitude [mdeg]',
        'raw': round(diurnal_amplitude(raw), 3),
        'compensated': round(diurnal_amplitude(comp), 3),
        'target': 'lower',
    })

    for col in list(temp_cols) + list(extra_drivers):
        if col not in df:
            continue
        rows.append({
            'test': f'|corr| with {col} (levels)',
            'raw': round(abs(raw.corr(df[col])), 4),
            'compensated': round(abs(comp.corr(df[col])), 4),
            'target': 'lower',
        })
        rows.append({
            'test': f'|corr| with {col} (differenced)',
            'raw': round(abs(raw.diff().corr(df[col].diff())), 4),
            'compensated': round(abs(comp.diff().corr(df[col].diff())), 4),
            'target': 'lower',
        })

    out = pd.DataFrame(rows)
    out['change_%'] = (100 * (out['compensated'] - out['raw'])
                       / out['raw'].abs()).round(1)
    out['verdict'] = np.where(out['compensated'] < out['raw'],
                              'improved', 'WORSE')
    return out.set_index('test')


def signed_correlation_check(df, comp_series, cols=('tair', 'twall', 'sr')):
    """
    Report signed correlations before and after compensation.

    A sign flip is the signature of overcorrection and is invisible in a table
    of absolute values.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``inc`` and the listed columns.
    comp_series : pd.Series
        Compensated signal.
    cols : tuple of str, optional
        Channels to check.

    Returns
    -------
    pd.DataFrame
        Signed correlations and a flag for sign reversal.
    """
    comp = comp_series.reindex(df.index)
    rows = []
    for c in cols:
        if c not in df:
            continue
        r_raw = df['inc'].corr(df[c])
        r_cmp = comp.corr(df[c])
        rows.append({
            'driver': c,
            'corr_raw': round(r_raw, 4),
            'corr_compensated': round(r_cmp, 4),
            'sign_flipped': bool(np.sign(r_raw) != np.sign(r_cmp)),
            'overcorrected': bool(np.sign(r_raw) != np.sign(r_cmp)
                                  and abs(r_cmp) > 0.05),
        })
    return pd.DataFrame(rows).set_index('driver')


def coefficient_sweep(df, temp_col='tair', coeffs=None):
    """
    Residual variance and thermal correlation as a function of the coefficient.

    Locates the coefficient that actually minimises residual variance and shows
    where the documented value sits relative to it.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``inc`` and ``temp_col``.
    temp_col : str, optional
        Driving temperature. Default ``'tair'``.
    coeffs : array-like or None, optional
        Coefficients in mdeg · °C⁻¹ · 10⁻³. Defaults to a sweep from -0.002
        to 0.020.

    Returns
    -------
    pd.DataFrame
        One row per coefficient.
    """
    if coeffs is None:
        coeffs = np.arange(-0.002, 0.0201, 0.0002)
    rows = []
    for c in coeffs:
        comp = compensate(df, temp_col=temp_col, coeff=c, normalise=False)
        rows.append({
            'coeff': round(float(c), 5),
            'mdeg_per_degC': round(float(c) * 1000, 3),
            'residual_var': float(comp.var()),
            'abs_corr_with_temp': abs(float(comp.corr(df[temp_col]))),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Prediction experiment
# ──────────────────────────────────────────────────────────────────────

def build_supervised(df, target, regressors, n_lags, horizon):
    """
    Assemble a supervised matrix for direct multi-step forecasting.

    Features are ``n_lags`` past values of the target plus the contemporaneous
    regressors. The label is the target ``horizon`` steps ahead, so a single
    fitted model produces a direct forecast at that horizon rather than an
    iterated one.

    Parameters
    ----------
    df : pd.DataFrame
        Source data, regular index.
    target : str
        Column to predict.
    regressors : list of str
        Exogenous columns, used at the forecast time.
    n_lags : int
        Number of autoregressive lags.
    horizon : int
        Forecast horizon in steps.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    frame = pd.DataFrame(index=df.index)
    for k in range(1, n_lags + 1):
        frame[f'{target}_lag{k}'] = df[target].shift(horizon + k - 1)
    for r in regressors:
        frame[r] = df[r]
    y = df[target]
    both = frame.join(y.rename('__y__')).dropna()
    return both.drop(columns='__y__'), both['__y__']


def run_ridge_experiment(df, configs, n_lags=12, horizon=1, train_frac=0.7,
                         alpha=1.0):
    """
    Compare preprocessing choices under a common, fast forecasting model.

    Every configuration is scored **on the raw signal scale**. Configurations
    whose target is the compensated series have the compensation term added
    back to their predictions before the error is computed, so that all
    configurations are predicting the same physical quantity and their errors
    are directly comparable. Without this step a comparison between a
    compensated and an uncompensated target is meaningless, because the two
    targets have different variances.

    The split is chronological: the model never sees data from after the point
    at which it is evaluated.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``inc``, ``inc_comp``, ``comp_term`` and the regressors.
    configs : dict
        Mapping of configuration name to ``{'target': str,
        'regressors': list, 'add_back': bool}``.
    n_lags : int, optional
        Autoregressive lags. Default ``12`` (12 hours on the hourly grid).
    horizon : int, optional
        Forecast horizon in steps. Default ``1``.
    train_frac : float, optional
        Fraction of the record used for training. Default ``0.7``.
    alpha : float, optional
        Ridge penalty. Default ``1.0``.

    Returns
    -------
    results : pd.DataFrame
        One row per configuration with test MAE and RMSE on the raw scale.
    predictions : dict
        Configuration name to a Series of raw-scale predictions on the test
        set, for plotting.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    rows, preds = [], {}
    for name, cfg in configs.items():
        X, y = build_supervised(df, cfg['target'], cfg['regressors'],
                                n_lags, horizon)
        if len(X) < 200:
            continue
        cut = int(len(X) * train_frac)
        Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
        ytr, yte = y.iloc[:cut], y.iloc[cut:]

        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(Xtr, ytr)
        yhat = pd.Series(model.predict(Xte), index=Xte.index)

        # Return both prediction and truth to the raw signal scale.
        if cfg['add_back']:
            term = df['comp_term'].reindex(Xte.index)
            shift = df['comp_shift'].iloc[0]
            yhat_raw = yhat + term + shift
            ytrue_raw = df['inc'].reindex(Xte.index)
        else:
            yhat_raw = yhat
            ytrue_raw = yte

        ok = yhat_raw.notna() & ytrue_raw.notna()
        err = (yhat_raw[ok] - ytrue_raw[ok])
        rows.append({
            'config': name,
            'target': cfg['target'],
            'regressors': ', '.join(cfg['regressors']) or 'none',
            'n_train': len(Xtr),
            'n_test': int(ok.sum()),
            'MAE_mdeg': round(float(err.abs().mean()), 4),
            'RMSE_mdeg': round(float(np.sqrt((err ** 2).mean())), 4),
        })
        preds[name] = yhat_raw[ok]

    if not rows:
        raise ValueError(
            'No configuration produced enough usable rows. Check that the '
            'window has at least a few hundred slots in which the target and '
            'every regressor are simultaneously present.')
    res = pd.DataFrame(rows).set_index('config')
    best = res['MAE_mdeg'].min()
    res['MAE_vs_best_%'] = (100 * (res['MAE_mdeg'] - best) / best).round(2)
    return res, preds


def run_neuralprophet_experiment(df, configs, freq=ANALYSIS_FREQ, train_frac=0.7,
                                 n_lags=24, epochs=30, horizon=1,
                                 yearly_seasonality=False, verbose=False,
                                 seed=0):
    """
    Repeat the comparison with the model the project actually deploys.

    The ridge experiment establishes the ordering cheaply; this confirms it
    under NeuralProphet, whose autoregressive and seasonal terms interact with
    the preprocessing choice in ways a linear model cannot show. Scoring is on
    the raw scale, exactly as in :func:`run_ridge_experiment`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``inc``, ``inc_comp``, ``comp_term``, ``comp_shift`` and
        the regressors.
    configs : dict
        As in :func:`run_ridge_experiment`.
    freq : str, optional
        Sampling frequency string. Default :data:`ANALYSIS_FREQ`.
    train_frac : float, optional
        Chronological training fraction. Default ``0.7``.
    n_lags : int, optional
        Autoregressive depth. Default ``24`` (one day on the hourly grid).
    epochs : int, optional
        Training epochs. Default ``30``.
    horizon : int, optional
        Forecast horizon in steps. Default ``1``.
    yearly_seasonality : bool, optional
        Include an annual term. Only meaningful when the record spans a full
        year; ``False`` by default because the current era does not.
    verbose : bool, optional
        Let NeuralProphet print progress. Default ``False``.
    seed : int or None, optional
        Random seed, re-applied before every configuration. Default ``0``.
        Training this model is stochastic, and without a fixed seed the
        ordering of configurations whose errors differ by a few per cent is not
        reproducible between runs. Seeding does not make such differences
        meaningful — it only makes them repeatable — so small gaps should still
        be read as ties.

    Returns
    -------
    pd.DataFrame
        One row per configuration, MAE and RMSE on the raw scale.
    """
    import logging
    from neuralprophet import NeuralProphet, set_log_level, set_random_seed

    if not verbose:
        set_log_level('ERROR')
        logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

    rows = []
    for name, cfg in configs.items():
        if seed is not None:
            set_random_seed(seed)
        # NeuralProphet must be handed a *regular* index: dropping incomplete
        # rows first would compact the series and silently change what a lag
        # means.
        #
        # Gaps are bridged by interpolation so that the autoregressive window
        # has continuous history to consume, and the positions that were
        # originally missing are recorded. Scoring then happens **only at
        # originally observed timestamps**. The model may therefore read a
        # filled value, but it is never rewarded for reproducing one — which
        # is the distinction that matters. Letting the model drop the affected
        # samples instead is not an option: on a record with many short gaps
        # it discards every prediction and returns an empty table.
        data = df[[cfg['target']] + cfg['regressors']]
        grid = pd.date_range(data.index.min(), data.index.max(), freq=freq)
        data = data.reindex(grid)
        if data[cfg['target']].notna().sum() < 500:
            continue

        observed = data[cfg['target']].notna()
        filled = data.interpolate(method='time', limit_direction='both')

        frame = pd.DataFrame({'ds': grid, 'y': filled[cfg['target']].values})
        for r in cfg['regressors']:
            frame[r] = filled[r].values

        cut = int(len(frame) * train_frac)
        train, test = frame.iloc[:cut], frame.iloc[cut:]

        model = NeuralProphet(
            n_lags=n_lags,
            n_forecasts=horizon,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=False,
            daily_seasonality=True,
            epochs=epochs,
            learning_rate=0.01,
            trend_reg=0.0,
            # Short gaps are filled internally; anything longer than that
            # window would otherwise raise, so the affected samples are
            # dropped instead. The target is never imputed in these studies —
            # filling it and then scoring a forecast against the fill would
            # measure the imputation rather than the forecast.
            impute_missing=True,
            drop_missing=False,
        )
        for r in cfg['regressors']:
            model.add_lagged_regressor(r)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(train, freq=freq, progress=None)
            forecast = model.predict(pd.concat([train.tail(n_lags), test]))

        col = f'yhat{horizon}'
        out = forecast[['ds', 'y', col]].set_index('ds')

        if cfg['add_back']:
            term = df['comp_term'].reindex(out.index)
            shift = df['comp_shift'].iloc[0]
            yhat_raw = out[col] + term + shift
        else:
            yhat_raw = out[col]
        # Truth is always the originally observed raw signal, never a fill.
        ytrue_raw = df['inc'].reindex(out.index)

        was_observed = observed.reindex(out.index).fillna(False)
        ok = yhat_raw.notna() & ytrue_raw.notna() & was_observed
        # An empty test set is a failure, not a result. Left unchecked it would
        # emit a row of blanks that reads like a finding.
        if int(ok.sum()) == 0:
            raise ValueError(
                f'Configuration "{name}" produced no scorable test samples. '
                f'The record is probably too fragmented for n_lags={n_lags}; '
                f'reduce it, or widen the window.')
        err = yhat_raw[ok] - ytrue_raw[ok]
        rows.append({
            'config': name,
            'target': cfg['target'],
            'regressors': ', '.join(cfg['regressors']) or 'none',
            'n_test': int(ok.sum()),
            'MAE_mdeg': round(float(err.abs().mean()), 4),
            'RMSE_mdeg': round(float(np.sqrt((err ** 2).mean())), 4),
        })

    if not rows:
        raise ValueError('No configuration had enough data to evaluate.')
    res = pd.DataFrame(rows).set_index('config')
    best = res['MAE_mdeg'].min()
    res['MAE_vs_best_%'] = (100 * (res['MAE_mdeg'] - best) / best).round(2)
    return res


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────
#
# Every figure is drawn through seaborn, and every figure size is expressed in
# units that scale with the active seaborn context. Switching between a
# notebook layout and a paper layout is therefore a single call to
# :func:`set_context`: seaborn rescales the typography and the scale factor
# below rescales the canvas, so a figure exported for the manuscript is the
# same figure at column width rather than a different one.

#: Canvas scale factor per seaborn context. ``'paper'`` lands a full-width
#: figure at about seven inches, the usual two-column text width, while
#: seaborn's own context handling takes care of fonts, line widths and markers.
CONTEXT_SCALE = {'paper': 0.55, 'notebook': 1.0, 'talk': 1.15, 'poster': 1.35}

#: Active context, updated by :func:`set_context`.
_ACTIVE = {'context': 'notebook', 'scale': 1.0}


def set_context(context='notebook', style='ticks', palette='colorblind',
                rc=None):
    """
    Set the seaborn theme and the canvas scale for every figure below.

    Parameters
    ----------
    context : str, optional
        Seaborn context: ``'paper'``, ``'notebook'``, ``'talk'`` or
        ``'poster'``. Default ``'notebook'``.
    style : str, optional
        Seaborn style. Default ``'ticks'``.
    palette : str, optional
        Seaborn palette. Default ``'colorblind'``.
    rc : dict or None, optional
        Extra matplotlib rcParams.

    Returns
    -------
    str
        The context that was applied.
    """
    base_rc = {'axes.axisbelow': True, 'figure.autolayout': False}
    if rc:
        base_rc.update(rc)
    sns.set_theme(context=context, style=style, palette=palette, rc=base_rc)
    _ACTIVE['context'] = context
    _ACTIVE['scale'] = CONTEXT_SCALE.get(context, 1.0)
    return context


def figsize(width, height):
    """
    Scale a nominal figure size by the active context.

    Parameters
    ----------
    width, height : float
        Size in inches at the ``'notebook'`` context.

    Returns
    -------
    tuple of float
    """
    s = _ACTIVE['scale']
    return (width * s, height * s)


def _finish(fig, save_path=None, filename=None):
    """Save a figure to PNG and SVG when a destination is given."""
    fig.tight_layout()
    if save_path and filename:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, filename + '.png'),
                    dpi=200, bbox_inches='tight')
        fig.savefig(os.path.join(save_path, filename + '.svg'),
                    bbox_inches='tight')
    return fig


def _ts(ax, series, label=None, lw=0.8, **kw):
    """Draw one time series on an axis through seaborn."""
    sns.lineplot(x=series.index, y=series.values, ax=ax, label=label,
                 linewidth=lw, **kw)
    return ax


def _steps_per_day(index):
    """Number of samples per day implied by a regular index."""
    if len(index) < 2:
        return 24
    dt = (index[1] - index[0]).total_seconds() / 3600.0
    return max(int(round(24.0 / dt)), 1)


CHANNEL_LABELS = {
    'inc': 'Inclination\n[mdeg]', 'tair': 'Air temp\n[°C]',
    'twall': 'Wall temp\n[°C]', 'sr': 'Solar rad\n[W/m²]',
    'rh': 'Rel. humidity\n[%]', 'batt': 'Battery\n[V]',
}


def plot_overview(df, cols=('inc', 'tair', 'twall', 'sr', 'rh'),
                  title='', save_path=None, filename=None):
    """
    Stacked time series of the signal and every candidate driver.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    cols : tuple of str, optional
        Channels to draw, one panel each.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cols = [c for c in cols if c in df]
    fig, axes = plt.subplots(len(cols), 1,
                             figsize=figsize(13, 2.0 * len(cols)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    palette = sns.color_palette(n_colors=max(len(cols), 3))
    for ax, c, colour in zip(axes, cols, palette):
        _ts(ax, df[c], lw=0.5, color=colour)
        ax.set_ylabel(CHANNEL_LABELS.get(c, c))
        ax.set_xlabel('')
    sns.despine(fig=fig)
    if title:
        fig.suptitle(title, y=1.002)
    return _finish(fig, save_path, filename)


def plot_diurnal(df, cols=('inc', 'tair', 'twall', 'sr'), title='',
                 save_path=None, filename=None):
    """
    Mean daily profile of each channel, standardised for shape comparison.

    Overlaying the standardised profiles shows at a glance whether the signal
    follows air temperature, wall temperature or solar radiation, and how far
    it lags behind them. The lag read here is a *diurnal* one; the seasonal
    behaviour of the same pair can be entirely different, which is why
    :func:`inertia_scan` exists.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    cols : tuple of str, optional
        Channels to overlay.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cols = [c for c in cols if c in df]
    hour = df.index.hour + df.index.minute / 60
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize(13, 4.2))
    for c in cols:
        prof = df[c].groupby(hour).mean()
        sns.lineplot(x=prof.index, y=prof.values, ax=ax1, label=c)
        z = (prof - prof.mean()) / prof.std()
        sns.lineplot(x=z.index, y=z.values, ax=ax2, label=c)
    ax1.set_title('Mean diurnal profile (native units)')
    ax2.set_title('Standardised — shape and phase comparison')
    for ax in (ax1, ax2):
        ax.set_xlabel('Hour of day')
        ax.set_xticks(range(0, 25, 3))
        ax.legend()
    ax2.set_ylabel('z-score')
    sns.despine(fig=fig)
    if title:
        fig.suptitle(title, y=1.02)
    return _finish(fig, save_path, filename)


def plot_correlation_heatmaps(df, cols, title='', save_path=None,
                              filename=None):
    """
    Side-by-side correlation heatmaps on levels and on first differences.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    cols : list of str
        Channels to include.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize(13, 4.8))
    for ax, diff, name in zip(axes, [False, True],
                              ['Levels', 'First differences']):
        m = correlation_matrix(df, cols, differenced=diff)
        sns.heatmap(m, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    vmin=-1, vmax=1, ax=ax, cbar=diff,
                    annot_kws={'size': 8 * _ACTIVE['scale'] ** 0.5})
        ax.set_title(f'{name} — Pearson')
    if title:
        fig.suptitle(title, y=1.02)
    return _finish(fig, save_path, filename)


def plot_cross_correlation(ccf, sampling_hours=1.0, title='',
                           save_path=None, filename=None,
                           ylabel='Correlation of first differences'):
    """
    Cross-correlation functions with the peak of each driver marked.

    Parameters
    ----------
    ccf : pd.DataFrame
        Output of :func:`cross_correlation`.
    sampling_hours : float, optional
        Hours per step. Default ``1.0``.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.
    ylabel : str, optional
        Vertical axis label.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize(12, 4.6))
    hours = ccf.index * sampling_hours
    palette = sns.color_palette(n_colors=max(len(ccf.columns), 3))
    for c, colour in zip(ccf.columns, palette):
        sns.lineplot(x=hours, y=ccf[c].values, ax=ax, label=c, color=colour)
        k = ccf[c].abs().idxmax()
        ax.plot(k * sampling_hours, ccf[c].loc[k], 'o', ms=6, color=colour)
    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Driver lead [hours]   (positive = driver leads signal)')
    ax.set_ylabel(ylabel)
    ax.legend()
    sns.despine(fig=fig)
    if title:
        ax.set_title(title)
    return _finish(fig, save_path, filename)


def plot_lag_scan(ccf_levels, ccf_diff, sampling_hours=1.0, title='',
                  save_path=None, filename=None):
    """
    Lag structure on levels and on first differences, side by side.

    The two panels answer different questions and routinely disagree.
    Differencing is a high-pass filter: it discards the slow component of both
    series, so the differenced panel describes how the signal tracks a driver
    from one interval to the next and says nothing about the response of the
    structure as a whole. The levels panel retains the slow component, which is
    where the thermal mass of a wall expresses itself. A study that reports only
    the differenced panel will conclude that the response is instantaneous
    whether it is or not.

    Parameters
    ----------
    ccf_levels, ccf_diff : pd.DataFrame
        Outputs of :func:`cross_correlation` with ``differenced=False`` and
        ``differenced=True`` respectively.
    sampling_hours : float, optional
        Hours per step. Default ``1.0``.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize(13, 4.6), sharey=False)
    panels = [(axes[0], ccf_levels, 'Levels'),
              (axes[1], ccf_diff, 'First differences')]
    for ax, ccf, name in panels:
        hours = ccf.index * sampling_hours
        palette = sns.color_palette(n_colors=max(len(ccf.columns), 3))
        for c, colour in zip(ccf.columns, palette):
            sns.lineplot(x=hours, y=ccf[c].values, ax=ax, label=c,
                         color=colour)
            k = ccf[c].abs().idxmax()
            ax.plot(k * sampling_hours, ccf[c].loc[k], 'o', ms=6, color=colour)
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xlabel('Driver lead [hours]')
        ax.set_ylabel(f'Correlation — {name.lower()}')
        ax.set_title(name)
        ax.legend()
    sns.despine(fig=fig)
    if title:
        fig.suptitle(title, y=1.02)
    return _finish(fig, save_path, filename)


def plot_inertia_scan(scan, title='', save_path=None, filename=None):
    """
    Explained variance against thermal time constant, one curve per driver.

    Parameters
    ----------
    scan : pd.DataFrame
        Output of :func:`inertia_scan`.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize(13, 4.4))
    sns.lineplot(data=scan, x='tau_hours', y='r2', hue='driver', marker='o',
                 ax=ax1)
    for d, g in scan.groupby('driver', sort=False):
        best = g.loc[g['r2'].idxmax()]
        ax1.axvline(best['tau_hours'], ls=':', lw=1.0, alpha=0.6)
    ax1.set_xlabel('Thermal time constant τ [hours]')
    ax1.set_ylabel('$R^2$ of the fit on levels')
    ax1.set_title('How much a lagged driver explains')

    sns.lineplot(data=scan, x='tau_hours', y='slope', hue='driver',
                 marker='o', ax=ax2, legend=False)
    ax2.set_xlabel('Thermal time constant τ [hours]')
    ax2.set_ylabel('Fitted slope [mdeg per driver unit]')
    ax2.set_title('The slope the correction would need')
    sns.despine(fig=fig)
    if title:
        fig.suptitle(title, y=1.02)
    return _finish(fig, save_path, filename)


def plot_thermal_scatter(df, temp_cols=('tair', 'twall'),
                         documented_slope=DOCUMENTED_COEFF * 1000,
                         title='', save_path=None, filename=None):
    """
    Signal against each temperature, with the fitted and documented slopes.

    Drawing the documented 5 mdeg/°C line on the same axes as the fitted line
    makes the size of the mismatch immediately visible.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    temp_cols : tuple of str, optional
        Temperature channels to plot.
    documented_slope : float, optional
        Slope implied by the documented coefficient, in mdeg/°C.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    temp_cols = [c for c in temp_cols if c in df]
    fig, axes = plt.subplots(1, len(temp_cols),
                             figsize=figsize(6.2 * len(temp_cols), 4.6))
    for ax, c in zip(np.atleast_1d(axes), temp_cols):
        d = df[['inc', c]].dropna()
        sns.scatterplot(x=d[c], y=d['inc'], ax=ax, s=4, alpha=0.15,
                        linewidth=0, rasterized=True, legend=False)
        fit = fit_slope(df, 'inc', c)
        xs = np.linspace(d[c].min(), d[c].max(), 50)
        centre_x, centre_y = d[c].mean(), d['inc'].mean()
        ax.plot(xs, centre_y + fit['slope'] * (xs - centre_x), '-', lw=2,
                color=sns.color_palette()[3],
                label=f"fitted {fit['slope']:.2f} mdeg/°C")
        ax.plot(xs, centre_y + documented_slope * (xs - centre_x), 'k--', lw=2,
                label=f'documented {documented_slope:.1f} mdeg/°C')
        ax.set_xlabel(f'{c} [°C]')
        ax.set_ylabel('Inclination [mdeg]')
        ax.legend()
    sns.despine(fig=fig)
    if title:
        fig.suptitle(title, y=1.02)
    return _finish(fig, save_path, filename)


def plot_compensation_effect(df, comp_series, temp_col='tair', title='',
                             save_path=None, filename=None):
    """
    What the compensation does to the series and to its thermal dependence.

    Parameters
    ----------
    df : pd.DataFrame
        Source data with ``inc``.
    comp_series : pd.Series
        Compensated signal.
    temp_col : str, optional
        Temperature used by the compensation. Default ``'tair'``.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    comp = comp_series.reindex(df.index)
    raw_c = df['inc'] - df['inc'].dropna().iloc[0]
    per_day = _steps_per_day(df.index)
    fig, axes = plt.subplots(2, 2, figsize=figsize(13, 7.5))

    ax = axes[0, 0]
    _ts(ax, raw_c, label='raw (shifted to 0)', lw=0.5)
    _ts(ax, comp, label='compensated', lw=0.5)
    ax.set_ylabel('Inclination [mdeg]')
    ax.set_title('Signal before and after compensation')
    ax.legend()

    ax = axes[0, 1]
    week = df.index[:per_day * 7]
    _ts(ax, raw_c.loc[week], label='raw', lw=1.0)
    _ts(ax, comp.loc[week], label='compensated', lw=1.0)
    ax.set_title('First week — detail')
    ax.legend()

    ax = axes[1, 0]
    d = pd.concat([df[temp_col], raw_c.rename('raw'), comp.rename('comp')],
                  axis=1).dropna()
    sns.scatterplot(x=d[temp_col], y=d['raw'], ax=ax, s=4, alpha=0.15,
                    linewidth=0, label='raw', rasterized=True)
    sns.scatterplot(x=d[temp_col], y=d['comp'], ax=ax, s=4, alpha=0.15,
                    linewidth=0, label='compensated', rasterized=True)
    ax.set_xlabel(f'{temp_col} [°C]')
    ax.set_ylabel('Inclination [mdeg]')
    ax.set_title('Thermal dependence — the target of the correction')
    ax.legend(markerscale=4)

    ax = axes[1, 1]
    hour = df.index.hour + df.index.minute / 60
    raw_prof = raw_c.groupby(hour).mean()
    comp_prof = comp.groupby(hour).mean()
    sns.lineplot(x=raw_prof.index, y=raw_prof.values, ax=ax, label='raw')
    sns.lineplot(x=comp_prof.index, y=comp_prof.values, ax=ax,
                 label='compensated')
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Mean inclination [mdeg]')
    ax.set_title('Mean diurnal cycle')
    ax.set_xticks(range(0, 25, 3))
    ax.legend()

    sns.despine(fig=fig)
    if title:
        fig.suptitle(title, y=1.005)
    return _finish(fig, save_path, filename)


def plot_coefficient_sweep(sweep, documented=DOCUMENTED_COEFF, title='',
                           save_path=None, filename=None):
    """
    Residual variance against the compensation coefficient.

    Parameters
    ----------
    sweep : pd.DataFrame
        Output of :func:`coefficient_sweep`.
    documented : float, optional
        Coefficient used by Notebook 00.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    best = sweep.loc[sweep['residual_var'].idxmin()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize(13, 4.4))

    sns.lineplot(data=sweep, x='mdeg_per_degC', y='residual_var', ax=ax1)
    ax1.axvline(documented * 1000, color='k', ls='--',
                label=f'documented {documented * 1000:.1f}')
    ax1.axvline(best['mdeg_per_degC'], color=sns.color_palette()[3], ls=':',
                label=f"variance-minimising {best['mdeg_per_degC']:.2f}")
    ax1.axvline(0, color='grey', lw=0.8)
    ax1.set_xlabel('Coefficient [mdeg per °C]')
    ax1.set_ylabel('Residual variance [mdeg²]')
    ax1.set_title('Residual variance vs coefficient')
    ax1.legend()

    sns.lineplot(data=sweep, x='mdeg_per_degC', y='abs_corr_with_temp', ax=ax2)
    ax2.axvline(documented * 1000, color='k', ls='--')
    ax2.axvline(best['mdeg_per_degC'], color=sns.color_palette()[3], ls=':')
    ax2.axhline(0, color='grey', lw=0.8)
    ax2.set_xlabel('Coefficient [mdeg per °C]')
    ax2.set_ylabel('|correlation| with temperature')
    ax2.set_title('Residual thermal dependence vs coefficient')

    sns.despine(fig=fig)
    if title:
        fig.suptitle(title, y=1.02)
    return _finish(fig, save_path, filename)


def plot_experiment(results, predictions=None, truth=None, title='',
                    save_path=None, filename=None):
    """
    Forecast accuracy by configuration, with an example of the test period.

    Parameters
    ----------
    results : pd.DataFrame
        Output of :func:`run_ridge_experiment` or the NeuralProphet variant.
    predictions : dict or None, optional
        Configuration name to raw-scale prediction series.
    truth : pd.Series or None, optional
        Observed raw signal, for the overlay panel.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    have_overlay = predictions is not None and truth is not None
    ncols = 2 if have_overlay else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize(7.4 * ncols, 4.8))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    order = results.sort_values('MAE_mdeg')
    sns.barplot(x=order['MAE_mdeg'].values, y=list(order.index), orient='h',
                ax=ax, color=sns.color_palette()[0])
    for i, v in enumerate(order['MAE_mdeg'].values):
        ax.text(v, i, f'  {v:.3f}', va='center')
    ax.set_xlabel('Test MAE on the raw signal scale [mdeg]')
    ax.set_ylabel('')
    ax.set_title('Forecast accuracy by preprocessing choice')

    if have_overlay:
        ax = axes[1]
        first = next(iter(predictions.values()))
        idx = first.index[:_steps_per_day(first.index) * 5]
        sns.lineplot(x=idx, y=truth.reindex(idx).values, ax=ax, color='k',
                     linewidth=1.4, label='observed')
        for name, series in predictions.items():
            sns.lineplot(x=idx, y=series.reindex(idx).values, ax=ax,
                         linewidth=1.0, alpha=0.85, label=name)
        ax.set_ylabel('Inclination [mdeg]')
        ax.set_title('Test period — first five days')
        ax.legend(fontsize='xx-small')

    sns.despine(fig=fig)
    if title:
        fig.suptitle(title, y=1.02)
    return _finish(fig, save_path, filename)
