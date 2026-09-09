"""
Module: shmlib.temporal_alignment

Runs the temporal-alignment side quest for Study 05.

Inputs are Study 01's exported archive, the ground-station CSV, ERA5 CSV,
explicit periods and candidate timestamp shifts. Outputs are scan, choice,
validation and manifest tables written only to the output directory passed by
the caller.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import proxies, site


ENV_PAIRS = (
    ('tair_gs', 'tair_str', 'tair_gs', 'gs', 'tair'),
    ('rh_gs', 'rh_str', 'rh_gs', 'gs', 'rh'),
    ('tair_era5', 'tair_str', 'tair_era5', 'era5', 'tair'),
    ('rh_era5', 'rh_str', 'rh_era5', 'era5', 'rh'),
)

INC_PAIRS = (
    ('inc_tair_gs', 'inc_comp_cleaned', 'tair_gs', 'gs', 'tair'),
    ('inc_rh_gs', 'inc_comp_cleaned', 'rh_gs', 'gs', 'rh'),
    ('inc_sr_gs', 'inc_comp_cleaned', 'sr_gs', 'gs', 'sr'),
    ('inc_tair_era5', 'inc_comp_cleaned', 'tair_era5', 'era5', 'tair'),
    ('inc_rh_era5', 'inc_comp_cleaned', 'rh_era5', 'era5', 'rh'),
    ('inc_sr_era5', 'inc_comp_cleaned', 'sr_era5', 'era5', 'sr'),
)

SCAN_COLUMNS = (
    'variant', 'period', 'era', 'role', 'dst_state', 'pair_id', 'pair_type',
    'sensor_column', 'reference_column', 'source', 'quantity', 'shift_minutes',
    'n_pairs', 'n_days', 'r_levels', 'r_daily', 'status',
)

VALIDATION_COLUMNS = (
    'variant', 'period', 'era', 'candidate_label', 'candidate_shift_minutes',
    'selected_shift_minutes', 'pair_id', 'pair_type', 'source', 'quantity',
    'r_daily_baseline', 'r_daily_candidate', 'delta', 'ci_low', 'ci_high',
    'n_pairs', 'n_days', 'n_blocks', 'n_boot_valid', 'status',
)


@dataclass(frozen=True)
class _Pair:
    pair_id: str
    sensor_column: str
    reference_column: str
    source: str
    quantity: str
    pair_type: str


def _as_series(values, name):
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f'{name} must be a Series or one-column DataFrame')
        values = values.iloc[:, 0]
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    if not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError(f'{name} must have a DatetimeIndex')
    return pd.to_numeric(values, errors='coerce')


def _require_regular_naive_index(obj, freq, name):
    index = pd.DatetimeIndex(obj.index)
    if index.tz is not None:
        raise ValueError(f'{name} index must be naive UTC')
    if index.has_duplicates:
        raise ValueError(f'{name} index must be unique')
    if not index.is_monotonic_increasing:
        raise ValueError(f'{name} index must be sorted')
    if len(index) == 0:
        raise ValueError(f'{name} index must not be empty')
    expected = pd.date_range(index.min(), index.max(), freq=freq)
    if len(expected) != len(index) or not expected.equals(index):
        raise ValueError(f'{name} index must be a complete regular {freq} grid')
    return index


def _shift_delta(shift_minutes, freq):
    delta = pd.Timedelta(minutes=int(shift_minutes))
    step = pd.Timedelta(freq)
    if delta.value % step.value != 0:
        raise ValueError('shifts_minutes must be multiples of freq')
    return delta


def _period_bound(value, freq, is_end):
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if is_end and isinstance(value, str) and len(value) == 10:
        stamp = stamp + pd.Timedelta(days=1) - pd.Timedelta(freq)
    return stamp


def _period_mask(index, start, end, freq):
    start = _period_bound(start, freq, False)
    end = _period_bound(end, freq, True)
    mask = np.ones(len(index), dtype=bool)
    if start is not None:
        mask &= index >= start
    if end is not None:
        mask &= index <= end
    return mask


def _local_dates(index, tz=site.SITE_TZ):
    utc = pd.DatetimeIndex(index).tz_localize('UTC')
    return utc.tz_convert(tz).date


def _dst_state(index, tz=site.SITE_TZ):
    local = pd.DatetimeIndex(index).tz_localize('UTC').tz_convert(tz)
    dst = pd.Series(local.map(lambda value: value.dst()), index=index)
    return np.where(dst == pd.Timedelta(0), 'standard', 'daylight')


def _transition_dates(index, tz=site.SITE_TZ):
    index = pd.DatetimeIndex(index)
    if len(index) == 0:
        return set()
    dates = pd.date_range(index.min().date(), index.max().date(), freq='D')
    zone = ZoneInfo(tz)
    out = set()
    for day in dates:
        start = day.to_pydatetime().replace(tzinfo=zone)
        stop = (day + pd.Timedelta(days=1)).to_pydatetime().replace(tzinfo=zone)
        if start.utcoffset() != stop.utcoffset():
            out.add(day.date())
    return out


def _not_transition_mask(index, tz=site.SITE_TZ):
    transitions = _transition_dates(index, tz=tz)
    if not transitions:
        return np.ones(len(index), dtype=bool)
    dates = _local_dates(index, tz=tz)
    return np.array([date not in transitions for date in dates], dtype=bool)


def _pearson_from_sums(n, sx, sy, sx2, sy2, sxy):
    if n < 2:
        return np.nan
    cov = sxy - sx * sy / n
    vx = sx2 - sx * sx / n
    vy = sy2 - sy * sy / n
    if vx <= 0 or vy <= 0:
        return np.nan
    return float(cov / np.sqrt(vx * vy))


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return np.nan
    x = x[finite]
    y = y[finite]
    return _pearson_from_sums(
        len(x), x.sum(), y.sum(), np.square(x).sum(), np.square(y).sum(),
        np.multiply(x, y).sum())


def _daily_demean(x, y, index):
    frame = pd.DataFrame({'x': x, 'y': y}, index=pd.DatetimeIndex(index))
    days = frame.index.normalize()
    means = frame.groupby(days, sort=False).transform('mean')
    demeaned = frame - means
    return demeaned['x'].to_numpy(), demeaned['y'].to_numpy()


def _corrs(x, y, index):
    r_levels = _pearson(x, y)
    xd, yd = _daily_demean(x, y, index)
    r_daily = _pearson(xd, yd)
    return r_levels, r_daily


def _support_index(sensor, reference, shifts, freq, start, end,
                   dst_state=None):
    sensor_index = _require_regular_naive_index(sensor, freq, 'sensor')
    reference_index = _require_regular_naive_index(reference, freq, 'reference')
    sensor_valid = sensor.notna().to_numpy()
    reference_valid = reference.notna()
    period = _period_mask(sensor_index, start, end, freq)
    source_not_transition = _not_transition_mask(sensor_index)
    support = sensor_valid & period & source_not_transition

    for shift in shifts:
        shifted = sensor_index + shift
        shifted_series = pd.Series(shifted, index=sensor_index)
        shifted_in_period = _period_mask(shifted, start, end, freq)
        shifted_not_transition = _not_transition_mask(shifted)
        in_reference = reference_valid.reindex(shifted).fillna(False).to_numpy()
        shift_support = shifted_in_period & shifted_not_transition & in_reference
        if dst_state is not None:
            shift_support &= _dst_state(sensor_index) == dst_state
            shift_support &= _dst_state(shifted_series.to_numpy()) == dst_state
        support &= shift_support
    return sensor_index[support]


def _shifted_reference(reference, source_index, shift):
    lookup = reference.reindex(source_index + shift)
    lookup.index = source_index
    return lookup


def _scan(sensor, reference, shifts_minutes, freq='20min', start=None,
          end=None, min_pairs=200, min_days=14, dst_state=None):
    sensor = _as_series(sensor, 'sensor').sort_index()
    reference = _as_series(reference, 'reference').sort_index()
    shifts = [_shift_delta(shift, freq) for shift in shifts_minutes]
    source_index = _support_index(sensor, reference, shifts, freq, start, end,
                                  dst_state=dst_state)

    rows = []
    x = sensor.reindex(source_index).to_numpy(dtype=float)
    for minutes, shift in zip(shifts_minutes, shifts):
        paired_index = source_index + shift
        y = _shifted_reference(reference, source_index, shift).to_numpy(
            dtype=float)
        r_levels, r_daily = _corrs(x, y, paired_index)
        n_days = int(pd.DatetimeIndex(paired_index).normalize().nunique())
        enough = len(source_index) >= min_pairs and n_days >= min_days
        rows.append({
            'shift_minutes': int(minutes),
            'n_pairs': int(len(source_index)),
            'n_days': n_days,
            'r_levels': r_levels if enough else np.nan,
            'r_daily': r_daily if enough else np.nan,
            'status': 'ok' if enough else 'insufficient_support',
        })
    return pd.DataFrame(rows)


def paired_shift_scan(sensor, reference, shifts_minutes, freq='20min',
                      start=None, end=None, min_pairs=200, min_days=14):
    """
    Scan timestamp shifts for one sensor-reference pair.

    Positive shifts assign the sensor observation to a later timestamp. For a
    source observation at 10:00 and a +60 minute candidate, the comparison uses
    the reference value at 11:00. All candidate shifts use the same original
    sensor timestamps: a timestamp is retained only when the unshifted sensor
    value and every shifted reference value are valid, in the requested period,
    outside civil DST-transition dates, and on the regular naive-UTC grid.

    The same arithmetic read the other way round — as a correction to the
    reference channel's own clock rather than to the sensor's — is
    :func:`reference_shift_scan`, which returns identical numbers for
    identical arguments and exists so a study need not negate this function's
    convention by hand.

    Parameters
    ----------
    sensor : pandas.Series
        Sensor-side series indexed by a complete regular naive-UTC grid.
    reference : pandas.Series
        Reference-side series indexed by a complete regular naive-UTC grid.
    shifts_minutes : sequence of int
        Candidate residual timestamp shifts, in minutes.
    freq : str, optional
        Grid spacing. Default ``'20min'``.
    start : str or pandas.Timestamp or None, optional
        Inclusive period start. Default ``None`` keeps the first timestamp.
    end : str or pandas.Timestamp or None, optional
        Inclusive period end. A date string includes the whole civil date.
        Default ``None`` keeps the last timestamp.
    min_pairs : int, optional
        Minimum paired samples required for an ``ok`` status. Default ``200``.
    min_days : int, optional
        Minimum UTC days required for an ``ok`` status. Default ``14``.

    Returns
    -------
    pandas.DataFrame
        Columns ``shift_minutes``, ``n_pairs``, ``n_days``, ``r_levels`` and
        ``r_daily``. A ``status`` column records insufficient support.
    """
    return _scan(sensor, reference, shifts_minutes, freq=freq, start=start,
                 end=end, min_pairs=min_pairs, min_days=min_days)[
        ['shift_minutes', 'n_pairs', 'n_days', 'r_levels', 'r_daily']]


def reference_shift_scan(sensor, reference, shifts_minutes, freq='20min',
                         start=None, end=None, min_pairs=200, min_days=14):
    """
    Scan timestamp shifts for one sensor-reference pair, displacing the reference.

    The same comparison :func:`paired_shift_scan` computes, stated in the
    opposite direction. `paired_shift_scan` reads a positive shift as the
    sensor observation assigned a later timestamp; this function reads the
    identical number as the reference channel's own stamp sitting later than
    the true observation time by that many minutes. The two readings are two
    descriptions of the same arithmetic — comparing the sensor at its own
    timestamp `t` against the reference at `t + shift` — and a study asking
    "how far is this source's clock offset from the truth" reads the answer
    directly off this function rather than negating the other's convention.

    A positive `shift_minutes` here means: the reference value carrying the
    label `t + shift` was truly observed at `t`, so aligning it with a sensor
    observation at `t` requires reading the reference at `t + shift`. For a
    sensor observation at 10:00 and a candidate of +15, the comparison uses
    the reference value labelled 11:00 — because a reference stamped 15
    minutes late labels its 10:00 observation with 10:15, and continuing that
    reasoning to a station reporting a 30-minute mean stamped at the
    interval's end, the true centre of the interval labelled 10:30 is 10:15,
    a 15-minute displacement.

    Parameters
    ----------
    sensor : pandas.Series
        Sensor-side series indexed by a complete regular naive-UTC grid, held
        at its own timestamps.
    reference : pandas.Series
        Reference-side series indexed by a complete regular naive-UTC grid,
        the one this function displaces.
    shifts_minutes : sequence of int
        Candidate reference stamp displacements, in minutes. Positive means
        the reference is stamped later than the truth it reports.
    freq : str, optional
        Grid spacing. Default ``'20min'``.
    start : str or pandas.Timestamp or None, optional
        Inclusive period start. Default ``None`` keeps the first timestamp.
    end : str or pandas.Timestamp or None, optional
        Inclusive period end. A date string includes the whole civil date.
        Default ``None`` keeps the last timestamp.
    min_pairs : int, optional
        Minimum paired samples required for an ``ok`` status. Default ``200``.
    min_days : int, optional
        Minimum UTC days required for an ``ok`` status. Default ``14``.

    Returns
    -------
    pandas.DataFrame
        Columns ``shift_minutes``, ``n_pairs``, ``n_days``, ``r_levels`` and
        ``r_daily``, identical in value to what
        ``paired_shift_scan(sensor, reference, shifts_minutes, ...)`` returns
        for the same arguments — only the sign's stated meaning differs.
    """
    return _scan(sensor, reference, shifts_minutes, freq=freq, start=start,
                 end=end, min_pairs=min_pairs, min_days=min_days)[
        ['shift_minutes', 'n_pairs', 'n_days', 'r_levels', 'r_daily']]


def shift_summary(scan, group_cols=('pair', 'period'), shift_col='shift_minutes',
                  r_col='r_daily'):
    """
    Collapse a shift scan to one row per group: the gain over no displacement.

    Built for the long-format table a study assembles by tagging and
    concatenating several :func:`paired_shift_scan` or
    :func:`reference_shift_scan` calls — one pair, one period, one scan each —
    the shape :func:`run_experiment`'s own ``TA_scan.csv`` and a study's
    per-pair scan both share. Reports, per group, the correlation at zero
    displacement, the displacement that maximises the *magnitude* of the
    correlation, the correlation there, and how much was gained. Magnitude
    rather than the signed value, because this module scans pairs whose
    expected sign differs — the two external forcings are expected to
    correlate positively across sources, the inclination against a heating
    driver is expected to correlate negatively — and a signed maximum would
    silently pick the weakest alignment for one of the two.

    Parameters
    ----------
    scan : pandas.DataFrame
        Long-format scan carrying at least `group_cols`, `shift_col` and
        `r_col`.
    group_cols : sequence of str, optional
        Columns identifying one scan curve — typically a pair label and a
        period name. Default ``('pair', 'period')``.
    shift_col : str, optional
        Column carrying the candidate displacement, in minutes. Default
        ``'shift_minutes'``.
    r_col : str, optional
        Column carrying the daily-anomaly correlation scanned. Default
        ``'r_daily'``.

    Returns
    -------
    pandas.DataFrame
        One row per distinct value of `group_cols`, plus:

        ``r_shift0``
            The value of `r_col` at ``shift_col == 0``. ``NaN`` if that
            displacement is absent from the group.
        ``shift_argmax``
            The displacement maximising ``abs(r_col)`` within the group. Ties
            are broken by the smallest absolute displacement, then the
            smallest signed one, matching this module's other tie-breaking
            rules. A group with no finite `r_col` value reports ``NaN``.
        ``r_argmax``
            The value of `r_col` at ``shift_argmax``.
        ``gain``
            ``abs(r_argmax) - abs(r_shift0)``. ``NaN`` if either input is
            missing.
        ``sign``
            ``'positive'``, ``'negative'`` or ``'zero'``, the sign of
            ``shift_argmax``; ``'undefined'`` where no finite `r_col` value
            was found.
    """
    group_cols = list(group_cols)
    rows = []
    for key, group in scan.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        zero = group[group[shift_col] == 0]
        r_zero = float(zero.iloc[0][r_col]) if len(zero) and pd.notna(
            zero.iloc[0][r_col]) else np.nan

        valid = group[group[r_col].notna()]
        if valid.empty:
            shift_argmax, r_argmax, sign = np.nan, np.nan, 'undefined'
        else:
            ranked = valid.assign(_abs_r=valid[r_col].abs(),
                                  _abs_shift=valid[shift_col].abs())
            ranked = ranked.sort_values(
                ['_abs_r', '_abs_shift', shift_col],
                ascending=[False, True, True])
            best = ranked.iloc[0]
            shift_argmax = int(best[shift_col])
            r_argmax = float(best[r_col])
            sign = ('zero' if shift_argmax == 0
                   else 'positive' if shift_argmax > 0 else 'negative')

        gain = (abs(r_argmax) - abs(r_zero)
               if np.isfinite(r_argmax) and np.isfinite(r_zero) else np.nan)

        row = dict(zip(group_cols, key))
        row.update({'r_shift0': r_zero, 'shift_argmax': shift_argmax,
                    'r_argmax': r_argmax, 'gain': gain, 'sign': sign})
        rows.append(row)
    return pd.DataFrame(rows, columns=group_cols + [
        'r_shift0', 'shift_argmax', 'r_argmax', 'gain', 'sign'])


def scan_reference_pairs(records, pairs, periods, shifts_minutes, freq='20min',
                         min_pairs=200, min_days=14):
    """
    Tag and concatenate :func:`reference_shift_scan` over pairs, periods and
    record variants.

    The nested loop a study would otherwise write as a notebook cell — one
    :func:`reference_shift_scan` call per combination of a record variant (for
    instance, a proxy loaded with and without a stamp correction), a period
    and a sensor-reference pair, each tagged so the calls can be told apart
    once concatenated. Kept here, documented once, rather than written inline
    wherever a study needs the same loop.

    Parameters
    ----------
    records : dict of str to pandas.DataFrame
        Record label to the harmonised frame it is scanned in. A study
        comparing a proxy's vendor stamp against a corrected one passes both
        frames here, keyed by a label such as ``'vendor'`` and ``'corrected'``.
    pairs : sequence of tuple
        ``(pair_label, variable, sensor_column, reference_column)`` per scan.
        `sensor_column` and `reference_column` must name columns present in
        every frame of `records`.
    periods : sequence of dict
        Period dictionaries with ``name``, ``start`` and ``end``, as accepted
        by `start`/`end` of :func:`reference_shift_scan`.
    shifts_minutes : sequence of int
        Candidate reference stamp displacements, in minutes.
    freq : str, optional
        Grid spacing. Default ``'20min'``.
    min_pairs : int, optional
        Minimum paired samples required for an ``ok``-worthy row. Default
        ``200``.
    min_days : int, optional
        Minimum UTC days required for an ``ok``-worthy row. Default ``14``.

    Returns
    -------
    pandas.DataFrame
        The concatenation of every ``(record, period, pair)`` scan, each row
        carrying the added columns ``record``, ``period``, ``pair``,
        ``variable``, ``sensor_column`` and ``reference_column`` alongside
        :func:`reference_shift_scan`'s own ``shift_minutes``, ``n_pairs``,
        ``n_days``, ``r_levels`` and ``r_daily``.
    """
    tables = []
    for record_label, record in records.items():
        for period in periods:
            for pair_label, variable, sensor_column, reference_column in pairs:
                scan = reference_shift_scan(
                    record[sensor_column], record[reference_column],
                    shifts_minutes, freq=freq, start=period['start'],
                    end=period['end'], min_pairs=min_pairs, min_days=min_days)
                scan = scan.assign(
                    record=record_label, period=period['name'],
                    pair=pair_label, variable=variable,
                    sensor_column=sensor_column,
                    reference_column=reference_column)
                tables.append(scan)
    return pd.concat(tables, ignore_index=True)


def _aligned_delta_series(reference, baseline, candidate):
    reference = _as_series(reference, 'reference').sort_index()
    baseline = _as_series(baseline, 'baseline').sort_index()
    candidate = _as_series(candidate, 'candidate').sort_index()
    common = reference.index.intersection(baseline.index).intersection(
        candidate.index)
    frame = pd.DataFrame({
        'reference': reference.reindex(common),
        'baseline': baseline.reindex(common),
        'candidate': candidate.reindex(common),
    }).dropna()
    if frame.empty:
        return frame
    for column in frame.columns:
        days = frame.index.normalize()
        frame[column] = frame[column] - frame[column].groupby(
            days, sort=False).transform('mean')
    return frame


def _block_stats(frame, x_col, y_col, block_days):
    blocks = frame.index.normalize()
    origin = blocks.min()
    block_id = ((blocks - origin) // pd.Timedelta(days=block_days)).astype(int)
    tmp = frame[[x_col, y_col]].copy()
    tmp['_block'] = block_id
    rows = []
    for block, group in tmp.groupby('_block', sort=True):
        x = group[x_col].to_numpy(dtype=float)
        y = group[y_col].to_numpy(dtype=float)
        rows.append({
            'block': int(block),
            'n': int(len(group)),
            'sx': float(x.sum()),
            'sy': float(y.sum()),
            'sx2': float(np.square(x).sum()),
            'sy2': float(np.square(y).sum()),
            'sxy': float(np.multiply(x, y).sum()),
        })
    return pd.DataFrame(rows).set_index('block') if rows else pd.DataFrame()


def _delta_from_stats(stats_base, stats_candidate, blocks, absolute):
    base = stats_base.loc[blocks].sum()
    cand = stats_candidate.loc[blocks].sum()
    rb = _pearson_from_sums(
        base['n'], base['sx'], base['sy'], base['sx2'], base['sy2'],
        base['sxy'])
    rc = _pearson_from_sums(
        cand['n'], cand['sx'], cand['sy'], cand['sx2'], cand['sy2'],
        cand['sxy'])
    if not np.isfinite(rb) or not np.isfinite(rc):
        return np.nan
    return float(abs(rc) - abs(rb) if absolute else rc - rb)


def bootstrap_correlation_delta(reference, baseline, candidate, block_days=7,
                                n_boot=1000, seed=20260906, min_blocks=8,
                                absolute=False):
    """
    Bootstrap the paired daily-anomaly correlation change.

    Parameters
    ----------
    reference : pandas.Series
        Reference series on a DatetimeIndex.
    baseline : pandas.Series
        Baseline candidate series on timestamps comparable to ``reference``.
    candidate : pandas.Series
        Alternative candidate series on timestamps comparable to ``reference``.
    block_days : int, optional
        Calendar days per resampled block. Default ``7``.
    n_boot : int, optional
        Number of bootstrap replicates. Default ``1000``.
    seed : int, optional
        Random seed. Default ``20260906``.
    min_blocks : int, optional
        Minimum nonempty blocks needed to report an interval. Default ``8``.
    absolute : bool, optional
        If ``True``, report change in absolute correlation. Default ``False``.

    Returns
    -------
    dict
        ``delta``, ``ci_low``, ``ci_high``, ``n_blocks`` and
        ``n_boot_valid``. Intervals are ``NaN`` when support is inadequate.
    """
    frame = _aligned_delta_series(reference, baseline, candidate)
    if frame.empty:
        return {'delta': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                'n_blocks': 0, 'n_boot_valid': 0}

    base_stats = _block_stats(frame, 'reference', 'baseline', block_days)
    cand_stats = _block_stats(frame, 'reference', 'candidate', block_days)
    blocks = base_stats.index.intersection(cand_stats.index).to_numpy()
    delta = _delta_from_stats(base_stats, cand_stats, blocks, absolute)
    if len(blocks) < min_blocks:
        return {'delta': delta, 'ci_low': np.nan, 'ci_high': np.nan,
                'n_blocks': int(len(blocks)), 'n_boot_valid': 0}

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(n_boot)):
        sample = rng.choice(blocks, size=len(blocks), replace=True)
        value = _delta_from_stats(base_stats, cand_stats, sample, absolute)
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return {'delta': delta, 'ci_low': np.nan, 'ci_high': np.nan,
                'n_blocks': int(len(blocks)), 'n_boot_valid': 0}
    return {
        'delta': delta,
        'ci_low': float(np.percentile(draws, 2.5)),
        'ci_high': float(np.percentile(draws, 97.5)),
        'n_blocks': int(len(blocks)),
        'n_boot_valid': int(len(draws)),
    }


def load_sensor_package(archive_csv, freq):
    """
    Load the response and its on-structure forcings from study 1's archive.

    Public so a study can build the sensor-side record this module's own
    :func:`run_experiment` builds, without having to reimplement the column
    maps and the joining rule across the instrument change. Both the inputs
    and the load itself are exactly study 1's decisions, consumed rather than
    revisited: the compensated, spike-honoured inclination
    (``inc_comp_cleaned``), and air temperature and humidity taken from the
    current package where it was recording and from the legacy blocks
    elsewhere, joined by :func:`shmlib.proxies.join_eras`.

    Parameters
    ----------
    archive_csv : str or pathlib.Path
        Study 1's archive CSV, carrying ``inc_comp_cleaned``, ``inc_spike``
        and the era-specific temperature and humidity columns.
    freq : str
        Working grid, as a pandas offset alias.

    Returns
    -------
    pd.DataFrame
        Columns ``inc_comp_cleaned``, ``tair_str`` and ``rh_str``, on `freq`,
        in UTC. Carries the response's and both eras' load provenance in
        ``attrs['provenance']``, and the compensation immutability statement
        in ``attrs['compensation_immutable']``.
    """
    target, target_provenance = proxies.load_response(
        archive_csv, column='inc_comp_cleaned', spike_column='inc_spike',
        honour_spike=True, freq=freq, tz=site.SITE_TZ, min_count=1)
    current_map = {
        'tair': 'tair',
        'rh': 'n_rh_ok',
    }
    legacy_map = {
        'tair': 'tair',
        'rh': f'{site.TARGET_STATION}_rh_ok',
    }
    sensor_current, current_provenance = proxies.load_sensor_forcings(
        archive_csv, column_map=current_map, freq=freq, tz=site.SITE_TZ,
        honour_suspect=True, min_count=1)
    sensor_legacy, legacy_provenance = proxies.load_sensor_forcings(
        archive_csv, column_map=legacy_map, freq=freq, tz=site.SITE_TZ,
        honour_suspect=True, min_count=1)
    sensor = proxies.join_eras([sensor_current, sensor_legacy])
    sensor['inc_comp_cleaned'] = target.reindex(sensor.index)
    sensor.attrs['source'] = 'sensor_package'
    sensor.attrs['provenance'] = {
        'response': target_provenance,
        'current_forcings': current_provenance,
        'legacy_forcings': legacy_provenance,
    }
    sensor.attrs['compensation_immutable'] = {
        'source_column': 'inc_comp_cleaned',
        'spike_column': 'inc_spike',
        'honour_spike': True,
        'operation': 'timestamp_shift_only',
    }
    return sensor


def load_proxy_variants(ground_csv, era5_csv, freq):
    """
    Load the ground-station and ERA5 proxies onto `freq`, two ways.

    Public so a study can build the same two candidate proxy frames this
    module's own :func:`run_experiment` scans, without reimplementing which
    native grid each variant is resampled from before it is interpolated onto
    the working frequency. Neither variant applies a ground-station stamp
    correction; a study investigating one loads the ground station itself,
    with :func:`shmlib.proxies.load_ground_station`'s own ``stamp_offset``.

    Parameters
    ----------
    ground_csv : str or pathlib.Path
        Ground-station export CSV.
    era5_csv : str or pathlib.Path
        ERA5 export CSV.
    freq : str
        Working grid the two variants are interpolated onto, as a pandas
        offset alias.

    Returns
    -------
    dict
        ``{'pipeline': [ground, era5], 'native_ground': [ground, era5]}``.
        ``'pipeline'`` resamples the ground station from its documented
        hourly-equivalent grid first, matching how earlier studies read it;
        ``'native_ground'`` resamples it from its native half-hourly grid
        instead. ERA5 is identical in both: read hourly, then brought onto
        `freq` with :func:`shmlib.proxies.to_native_grid`, treating radiation
        as an interval accumulation.
    """
    ground_hourly = proxies.load_ground_station(ground_csv, freq='1h')
    ground_native = proxies.load_ground_station(ground_csv, freq='30min')
    era5_hourly = proxies.load_era5(era5_csv, freq='1h')
    era5 = proxies.to_native_grid(era5_hourly, freq=freq,
                                  accumulations=('sr',))
    return {
        'pipeline': [
            proxies.to_native_grid(ground_hourly, freq=freq,
                                   accumulations=()),
            era5,
        ],
        'native_ground': [
            proxies.to_native_grid(ground_native, freq=freq,
                                   accumulations=()),
            era5,
        ],
    }


def _scan_rows(record, period, variant, pair, shifts_minutes, freq, min_pairs,
               min_days, dst_state):
    table = _scan(
        record[pair.sensor_column], record[pair.reference_column],
        shifts_minutes, freq=freq, start=period['start'], end=period['end'],
        min_pairs=min_pairs, min_days=min_days,
        dst_state=None if dst_state == 'all' else dst_state)
    for column, value in (
        ('variant', variant),
        ('period', period['name']),
        ('era', period['era']),
        ('role', period['role']),
        ('dst_state', dst_state),
        ('pair_id', pair.pair_id),
        ('pair_type', pair.pair_type),
        ('sensor_column', pair.sensor_column),
        ('reference_column', pair.reference_column),
        ('source', pair.source),
        ('quantity', pair.quantity),
    ):
        table[column] = value
    return table[list(SCAN_COLUMNS)]


def _fisher_z(value):
    if not np.isfinite(value):
        return np.nan
    return float(np.arctanh(np.clip(value, -0.999999, 0.999999)))


def _choice_rows(scan):
    rows = []
    train = scan[(scan['role'] == 'train') & (scan['pair_type'] == 'env')
                 & (scan['dst_state'].isin(['standard', 'daylight']))]
    for (variant, era), group in train.groupby(['variant', 'era'], sort=False):
        objective_rows = []
        for shift, block in group.groupby('shift_minutes', sort=True):
            ok = block[(block['status'] == 'ok') & block['r_daily'].notna()]
            cells = ok[['pair_id', 'dst_state']].drop_duplicates()
            if len(cells) != 8:
                continue
            objective_rows.append({
                'shift_minutes': int(shift),
                'objective': float(ok['r_daily'].map(_fisher_z).mean()),
                'n_cells': 8,
            })
        if objective_rows:
            ranked = sorted(objective_rows,
                            key=lambda row: (-row['objective'],
                                             abs(row['shift_minutes']),
                                             row['shift_minutes']))
            selected = ranked[0]
            rows.append({
                'variant': variant,
                'era': era,
                'choice_type': 'common_shift',
                'pair_id': 'environment_mean',
                'dst_state': 'standard+daylight',
                'selected_shift_minutes': selected['shift_minutes'],
                'objective': selected['objective'],
                'n_cells': selected['n_cells'],
                'status': 'ok',
            })
        else:
            rows.append({
                'variant': variant,
                'era': era,
                'choice_type': 'common_shift',
                'pair_id': 'environment_mean',
                'dst_state': 'standard+daylight',
                'selected_shift_minutes': np.nan,
                'objective': np.nan,
                'n_cells': 0,
                'status': 'insufficient_training_cells',
            })

        for (pair_id, dst), block in group.groupby(['pair_id', 'dst_state'],
                                                  sort=False):
            ok = block[(block['status'] == 'ok') & block['r_daily'].notna()]
            if ok.empty:
                rows.append({
                    'variant': variant,
                    'era': era,
                    'choice_type': 'pair_preference',
                    'pair_id': pair_id,
                    'dst_state': dst,
                    'selected_shift_minutes': np.nan,
                    'objective': np.nan,
                    'n_cells': 0,
                    'status': 'insufficient_training_support',
                })
                continue
            ranked = ok.assign(objective=ok['r_daily'].map(_fisher_z)).sort_values(
                ['objective', 'shift_minutes'], ascending=[False, True])
            best_objective = ranked.iloc[0]['objective']
            ties = ranked[np.isclose(ranked['objective'], best_objective)]
            best = ties.assign(abs_shift=ties['shift_minutes'].abs()).sort_values(
                ['abs_shift', 'shift_minutes']).iloc[0]
            rows.append({
                'variant': variant,
                'era': era,
                'choice_type': 'pair_preference',
                'pair_id': pair_id,
                'dst_state': dst,
                'selected_shift_minutes': int(best['shift_minutes']),
                'objective': float(best_objective),
                'n_cells': 1,
                'status': 'ok',
            })
    return pd.DataFrame(rows)


def _candidate_label(shift, selected):
    if shift == selected:
        return 'selected'
    if shift == 0:
        return 'zero'
    if shift == -60:
        return 'minus_60'
    if shift == 60:
        return 'plus_60'
    return f'shift_{shift:+d}'


def _shifted_pair_series(record, pair, shift_minutes, period, freq):
    sensor = _as_series(record[pair.sensor_column], pair.sensor_column)
    reference = _as_series(record[pair.reference_column], pair.reference_column)
    shift = _shift_delta(shift_minutes, freq)
    source_index = _support_index(sensor, reference, [pd.Timedelta(0), shift],
                                  freq, period['start'], period['end'])
    paired_index = source_index + shift
    ref = sensor.reindex(source_index)
    base = _shifted_reference(reference, source_index, pd.Timedelta(0))
    cand = _shifted_reference(reference, source_index, shift)
    ref.index = paired_index
    base.index = paired_index
    cand.index = paired_index
    return ref, base, cand


def _validation_pair_row(record, period, variant, pair, shift, selected, freq,
                         n_boot, seed, min_pairs, min_days, block_days,
                         min_blocks):
    ref, base, cand = _shifted_pair_series(record, pair, shift, period, freq)
    frame = pd.DataFrame({'ref': ref, 'base': base, 'cand': cand}).dropna()
    n_pairs = int(len(frame))
    n_days = int(frame.index.normalize().nunique()) if n_pairs else 0
    absolute = pair.pair_type == 'inc'
    r_base = np.nan
    r_cand = np.nan
    if n_pairs:
        _, r_base = _corrs(frame['ref'], frame['base'], frame.index)
        _, r_cand = _corrs(frame['ref'], frame['cand'], frame.index)
    if n_pairs < min_pairs or n_days < min_days:
        stats = {'delta': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                 'n_blocks': 0, 'n_boot_valid': 0}
        status = 'insufficient_support'
    else:
        stats = bootstrap_correlation_delta(
            frame['ref'], frame['base'], frame['cand'],
            block_days=block_days, n_boot=n_boot, seed=seed,
            min_blocks=min_blocks, absolute=absolute)
        status = 'ok' if stats['n_boot_valid'] else 'insufficient_blocks'
    return {
        'variant': variant,
        'period': period['name'],
        'era': period['era'],
        'candidate_label': _candidate_label(shift, selected),
        'candidate_shift_minutes': int(shift),
        'selected_shift_minutes': int(selected) if pd.notna(selected) else np.nan,
        'pair_id': pair.pair_id,
        'pair_type': pair.pair_type,
        'source': pair.source,
        'quantity': pair.quantity,
        'r_daily_baseline': r_base,
        'r_daily_candidate': r_cand,
        'delta': stats['delta'],
        'ci_low': stats['ci_low'],
        'ci_high': stats['ci_high'],
        'n_pairs': n_pairs,
        'n_days': n_days,
        'n_blocks': stats['n_blocks'],
        'n_boot_valid': stats['n_boot_valid'],
        'status': status,
    }


def _joint_env_delta(record, period, pairs, shift, freq, block_days, absolute):
    pair_stats = []
    point = []
    counts = []
    for pair in pairs:
        ref, base, cand = _shifted_pair_series(record, pair, shift, period,
                                               freq)
        frame = _aligned_delta_series(ref, base, cand)
        if frame.empty:
            return None
        base_stats = _block_stats(frame, 'reference', 'baseline', block_days)
        cand_stats = _block_stats(frame, 'reference', 'candidate', block_days)
        blocks = base_stats.index.intersection(cand_stats.index)
        pair_stats.append((base_stats, cand_stats, blocks))
        point.append(_delta_from_stats(base_stats, cand_stats, blocks,
                                       absolute))
        counts.append((len(frame), frame.index.normalize().nunique()))
    common_blocks = pair_stats[0][2]
    for _, _, blocks in pair_stats[1:]:
        common_blocks = common_blocks.intersection(blocks)
    return pair_stats, common_blocks.to_numpy(), float(np.nanmean(point)), counts


def _validation_aggregate_row(record, period, variant, pairs, shift, selected,
                              freq, n_boot, seed, min_pairs, min_days,
                              block_days, min_blocks):
    prepared = _joint_env_delta(record, period, pairs, shift, freq, block_days,
                                absolute=False)
    base = {
        'variant': variant,
        'period': period['name'],
        'era': period['era'],
        'candidate_label': _candidate_label(shift, selected),
        'candidate_shift_minutes': int(shift),
        'selected_shift_minutes': int(selected) if pd.notna(selected) else np.nan,
        'pair_id': 'environment_mean_delta_r',
        'pair_type': 'env_aggregate',
        'source': 'gs+era5',
        'quantity': 'tair+rh',
        'r_daily_baseline': np.nan,
        'r_daily_candidate': np.nan,
    }
    if prepared is None:
        return {**base, 'delta': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                'n_pairs': 0, 'n_days': 0, 'n_blocks': 0,
                'n_boot_valid': 0, 'status': 'insufficient_support'}
    pair_stats, blocks, delta, counts = prepared
    n_pairs = min(count[0] for count in counts)
    n_days = min(count[1] for count in counts)
    if n_pairs < min_pairs or n_days < min_days or len(blocks) < min_blocks:
        return {**base, 'delta': delta, 'ci_low': np.nan, 'ci_high': np.nan,
                'n_pairs': int(n_pairs), 'n_days': int(n_days),
                'n_blocks': int(len(blocks)), 'n_boot_valid': 0,
                'status': 'insufficient_blocks'}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(n_boot)):
        sample = rng.choice(blocks, size=len(blocks), replace=True)
        values = []
        for base_stats, cand_stats, _ in pair_stats:
            values.append(_delta_from_stats(base_stats, cand_stats, sample,
                                            absolute=False))
        if np.all(np.isfinite(values)):
            draws.append(float(np.mean(values)))
    if not draws:
        status = 'insufficient_blocks'
        ci_low = np.nan
        ci_high = np.nan
    else:
        status = 'ok'
        ci_low = float(np.percentile(draws, 2.5))
        ci_high = float(np.percentile(draws, 97.5))
    return {**base, 'delta': delta, 'ci_low': ci_low, 'ci_high': ci_high,
            'n_pairs': int(n_pairs), 'n_days': int(n_days),
            'n_blocks': int(len(blocks)), 'n_boot_valid': int(len(draws)),
            'status': status}


def _validation_rows(records, choices, periods, freq, n_boot, seed, min_pairs,
                     min_days, block_days, min_blocks):
    rows = []
    env_pairs = [_Pair(*item, 'env') for item in ENV_PAIRS]
    inc_pairs = [_Pair(*item, 'inc') for item in INC_PAIRS]
    for variant, record in records.items():
        for period in periods:
            if period['role'] != 'validation':
                continue
            selected_rows = choices[
                (choices['variant'] == variant)
                & (choices['era'] == period['era'])
                & (choices['choice_type'] == 'common_shift')
            ]
            if selected_rows.empty or pd.isna(
                    selected_rows.iloc[0]['selected_shift_minutes']):
                selected = 0
            else:
                selected = int(selected_rows.iloc[0]['selected_shift_minutes'])
            candidates = []
            for shift in (0, -60, 60, selected):
                if shift not in candidates:
                    candidates.append(shift)
            for shift in candidates:
                for pair in env_pairs + inc_pairs:
                    rows.append(_validation_pair_row(
                        record, period, variant, pair, shift, selected, freq,
                        n_boot, seed, min_pairs, min_days, block_days,
                        min_blocks))
                rows.append(_validation_aggregate_row(
                    record, period, variant, env_pairs, shift, selected, freq,
                    n_boot, seed, min_pairs, min_days, block_days,
                    min_blocks))
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def _sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path):
    path = Path(path)
    return {
        'path': str(path),
        'size_bytes': int(path.stat().st_size),
        'sha256': _sha256(path),
    }


def _source_record(module):
    path = Path(inspect.getsourcefile(module))
    return _file_record(path)


def _version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta,)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _manifest(archive_csv, ground_csv, era5_csv, output_paths, periods,
              shifts_minutes, freq, n_boot, seed, min_pairs, min_days,
              block_days, min_blocks, sensor_attrs):
    return {
        'parameters': {
            'freq': freq,
            'shifts_minutes': [int(shift) for shift in shifts_minutes],
            'n_boot': int(n_boot),
            'seed': int(seed),
            'min_pairs': int(min_pairs),
            'min_days': int(min_days),
            'block_days': int(block_days),
            'min_blocks': int(min_blocks),
            'timezone': site.SITE_TZ,
        },
        'periods': periods,
        'shift_sign': {
            'positive_minutes': 'sensor observation assigned later timestamp',
            'example': '10:00 shifted +60 minutes is compared at 11:00',
            'baseline_conversion': 'site.to_utc removes the full Europe/Rome offset',
            'plus_60_interpretation': 'residual +60 approximates removing DST only',
        },
        'pair_schema': {
            'environment_pairs': [dict(zip(
                ('pair_id', 'sensor_column', 'reference_column', 'source',
                 'quantity'), item)) for item in ENV_PAIRS],
            'inclination_diagnostics': [dict(zip(
                ('pair_id', 'sensor_column', 'reference_column', 'source',
                 'quantity'), item)) for item in INC_PAIRS],
        },
        'objective_schema': {
            'selector': 'training only',
            'cells': '4 environmental pairs times 2 DST states',
            'score': 'equal-weight mean Fisher z of r_daily',
            'tie_break': 'smallest absolute shift, then smaller signed shift',
            'inclination_role': 'diagnostic only; never selects clock',
        },
        'support_and_exclusions': {
            'grid': 'complete regular naive-UTC index',
            'support': 'same original sensor timestamps across all scanned shifts per pair',
            'period': 'original and shifted timestamps must stay inside period',
            'dst': 'civil transition dates excluded after UTC conversion',
            'dst_strata': 'source and shifted reference timestamps must share stratum',
            'spikes': 'inc_spike honored; compensated values not recomputed',
        },
        'input_files': {
            'archive_csv': _file_record(archive_csv),
            'ground_csv': _file_record(ground_csv),
            'era5_csv': _file_record(era5_csv),
        },
        'source_files': {
            'temporal_alignment': _source_record(inspect.getmodule(_manifest)),
            'site': _source_record(site),
            'proxies': _source_record(proxies),
        },
        'package_versions': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'shmlib': None,
            'neuralprophet': _version('neuralprophet'),
        },
        'sensor_provenance': sensor_attrs,
        'outputs': {key: str(path) for key, path in output_paths.items()},
        'secrets': 'none recorded',
    }


def run_experiment(archive_csv, ground_csv, era5_csv, output_dir,
                   shifts_minutes, periods, freq='20min', n_boot=1000,
                   seed=20260906, min_pairs=200, min_days=14, block_days=7,
                   min_blocks=8):
    """
    Run the complete temporal-alignment protocol.

    Parameters
    ----------
    archive_csv : str or pathlib.Path
        Study 01 archive CSV holding ``inc_comp_cleaned`` and verdict columns.
    ground_csv : str or pathlib.Path
        Ground-station export CSV.
    era5_csv : str or pathlib.Path
        ERA5 export CSV.
    output_dir : str or pathlib.Path
        Directory where ``TA_scan.csv``, ``TA_choices.csv``,
        ``TA_validation.csv`` and ``TA_manifest.json`` are written.
    shifts_minutes : sequence of int
        Candidate residual timestamp shifts, in minutes.
    periods : sequence of dict
        Period dictionaries with ``name``, ``era``, ``role``, ``start`` and
        ``end``. Roles are ``'train'`` and ``'validation'``.
    freq : str, optional
        Working grid. Default ``'20min'``.
    n_boot : int, optional
        Bootstrap replicates. Default ``1000``.
    seed : int, optional
        Random seed. Default ``20260906``.
    min_pairs : int, optional
        Minimum paired samples for a reported scan/validation. Default ``200``.
    min_days : int, optional
        Minimum observed UTC days for a reported scan/validation. Default ``14``.
    block_days : int, optional
        Calendar days per bootstrap block. Default ``7``.
    min_blocks : int, optional
        Minimum nonempty blocks for intervals. Default ``8``.

    Returns
    -------
    dict
        Paths of the four output files written by this function.

    Side Effects
    ------------
    Creates ``output_dir`` if needed and writes the four ``TA_*`` artifacts.
    """
    periods = [dict(period) for period in periods]
    for period in periods:
        if period.get('role') not in {'train', 'validation'}:
            raise ValueError('period role must be train or validation')
        for key in ('name', 'era', 'start', 'end'):
            if key not in period:
                raise ValueError(f'period missing {key!r}')

    shifts_minutes = [int(shift) for shift in shifts_minutes]
    sensor = load_sensor_package(archive_csv, freq)
    proxy_variants = load_proxy_variants(ground_csv, era5_csv, freq)
    records = {}
    for variant, frames in proxy_variants.items():
        records[variant] = proxies.harmonise([sensor, *frames], freq=freq)

    pairs = [_Pair(*item, 'env') for item in ENV_PAIRS]
    pairs += [_Pair(*item, 'inc') for item in INC_PAIRS]
    scan_rows = []
    for variant, record in records.items():
        for period in periods:
            states = ['all', 'standard', 'daylight']
            for dst in states:
                for pair in pairs:
                    scan_rows.append(_scan_rows(
                        record, period, variant, pair, shifts_minutes, freq,
                        min_pairs, min_days, dst))
    scan = pd.concat(scan_rows, ignore_index=True)
    choices = _choice_rows(scan)
    validation = _validation_rows(
        records, choices, periods, freq, n_boot, seed, min_pairs, min_days,
        block_days, min_blocks)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        'scan': output_dir / 'TA_scan.csv',
        'choices': output_dir / 'TA_choices.csv',
        'validation': output_dir / 'TA_validation.csv',
        'manifest': output_dir / 'TA_manifest.json',
    }
    scan.to_csv(output_paths['scan'], index=False)
    choices.to_csv(output_paths['choices'], index=False)
    validation.to_csv(output_paths['validation'], index=False)
    manifest = _manifest(
        archive_csv, ground_csv, era5_csv, output_paths, periods,
        shifts_minutes, freq, n_boot, seed, min_pairs, min_days, block_days,
        min_blocks, sensor.attrs)
    with output_paths['manifest'].open('w', encoding='utf-8') as handle:
        json.dump(_json_safe(manifest), handle, indent=2, sort_keys=True)
        handle.write('\n')
    return output_paths
