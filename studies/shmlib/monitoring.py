"""
Module: monitoring.py
Residual-based monitoring for the studies: reference statistics, EWMA and CUSUM
control charts, joint alarms, alarm episodes and run lengths.

Inputs are residual series - observed minus expected - indexed by timestamp.
Outputs are chart tables and episode tables. Nothing here fits a model or reads
a file; the caller supplies the residual and the settings, and every threshold
is an argument so that the study which chose it states its value.
"""

import numpy as np
import pandas as pd

# A normal distribution's median absolute deviation is this fraction of its
# standard deviation; dividing by it turns a MAD into a comparable sigma.
MAD_TO_SIGMA = 0.6744897501960817


def reference_stats(residuals, start=None, end=None, robust=True):
    """
    Centre and scale of the residual over an in-control reference window.

    Every limit drawn later is a multiple of these two numbers, so the window
    they are estimated on is the single most consequential choice in the whole
    monitoring step: a window containing the event to be detected calibrates the
    detector against the very thing it is meant to find.

    Parameters
    ----------
    residuals : pd.Series
        Observed minus expected, indexed by timestamp.
    start, end : pd.Timestamp or str or None, optional
        Inclusive bounds of the reference window. ``None`` extends to the
        respective end of the series. Default ``None``.
    robust : bool, optional
        When true, centre by the median and scale by the median absolute
        deviation rescaled to a standard deviation, so that a contaminated tail
        cannot inflate the limits it should be judged against. When false, use
        the mean and the sample standard deviation. Default ``True``.

    Returns
    -------
    dict
        ``mu``, ``sigma``, ``n``, ``start``, ``end``.
    """
    values = pd.to_numeric(residuals, errors='coerce').dropna()
    if start is not None:
        values = values.loc[pd.Timestamp(start):]
    if end is not None:
        values = values.loc[:pd.Timestamp(end)]

    if values.empty:
        return {'mu': np.nan, 'sigma': np.nan, 'n': 0,
                'start': None, 'end': None}

    if robust:
        mu = float(values.median())
        sigma = float((values - mu).abs().median() / MAD_TO_SIGMA)
    else:
        mu = float(values.mean())
        sigma = float(values.std(ddof=1))

    return {'mu': mu, 'sigma': sigma, 'n': int(values.size),
            'start': values.index.min(), 'end': values.index.max()}


def _standardise(residuals, mu, sigma):
    values = pd.to_numeric(residuals, errors='coerce')
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(
            'sigma must be finite and positive; a degenerate reference window '
            'cannot standardise a residual. Widen the reference window, or '
            'check that it holds more than one distinct value.')
    return (values - mu) / sigma


def _regular_step_hours(index, freq=None):
    """
    The grid spacing in hours, refusing an index that is not a regular grid.

    Every duration this module reports is a slot count multiplied by one
    spacing, which is only meaningful when the spacing is constant. An alarm
    series whose rows were dropped across an outage, rather than carried as
    missing values on a complete grid, would otherwise report an episode
    spanning time the record does not cover.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Index to measure.
    freq : str or None, optional
        Spacing the caller declared. When given it must agree with the index,
        and it is what is returned for an index too short to measure. Default
        ``None``.

    Returns
    -------
    float
        Spacing in hours, or ``NaN`` when the index is too short and no
        ``freq`` was declared.

    Raises
    ------
    ValueError
        When the index carries more than one distinct spacing, or when ``freq``
        contradicts the spacing the index actually has.
    """
    index = pd.DatetimeIndex(index)
    declared_hours = (float(pd.Timedelta(freq) / pd.Timedelta(hours=1))
                      if freq is not None else None)

    if len(index) < 2:
        return declared_hours if declared_hours is not None else np.nan

    diffs = np.diff(index.to_numpy())
    distinct = np.unique(diffs)
    if distinct.size > 1:
        raise ValueError(
            f'index is not a regular grid: {distinct.size} distinct spacings '
            f'found; first offending pair is '
            f'{pd.Timestamp(index[0])} to {pd.Timestamp(index[1])}.')

    step_hours = float(distinct[0] / np.timedelta64(1, 'h'))
    if declared_hours is not None and not np.isclose(step_hours, declared_hours):
        raise ValueError(
            f'freq={freq!r} ({declared_hours} h) disagrees with the index '
            f'spacing ({step_hours} h).')
    return step_hours


def ewma_chart(residuals, mu, sigma, lam=0.2, L=3.0):
    """
    Exponentially weighted moving average chart on the standardised residual.

    EWMA answers "has the level moved and stayed moved", which is the shape a
    structural departure takes. Missing residuals do not update the statistic;
    the chart carries its previous value across them rather than treating an
    absent reading as a zero.

    Parameters
    ----------
    residuals : pd.Series
        Observed minus expected, indexed by timestamp.
    mu, sigma : float
        Reference centre and scale, normally from ``reference_stats``.
    lam : float, optional
        Smoothing weight in ``(0, 1]``. Smaller reacts more slowly and detects
        smaller sustained shifts. Default ``0.2``.
    L : float, optional
        Control limits in standard deviations of the EWMA statistic. Larger
        means fewer false alarms and slower detection. Default ``3.0``.

    Returns
    -------
    pd.DataFrame
        Columns ``z``, ``ewma``, ``ucl``, ``lcl`` and ``alarm``, indexed as the
        input.

    Raises
    ------
    ValueError
        When ``sigma`` is not finite or not positive: a degenerate reference
        window cannot standardise a residual, and reporting an all-``False``
        alarm column in that case would read as a quiet structure rather than
        as the broken reference window it actually is.
    """
    z = _standardise(residuals, mu, sigma)
    statistic = np.full(len(z), np.nan)
    limit = np.full(len(z), np.nan)

    current = 0.0
    step = 0
    for position, value in enumerate(z.to_numpy()):
        if np.isfinite(value):
            current = lam * value + (1.0 - lam) * current
            step += 1
        statistic[position] = current if step else np.nan
        if step:
            spread = np.sqrt((lam / (2.0 - lam))
                             * (1.0 - (1.0 - lam) ** (2 * step)))
            limit[position] = L * spread

    chart = pd.DataFrame({'z': z, 'ewma': statistic,
                          'ucl': limit, 'lcl': -limit}, index=z.index)
    chart['alarm'] = ((chart['ewma'] > chart['ucl'])
                      | (chart['ewma'] < chart['lcl'])).fillna(False)
    return chart


def cusum_chart(residuals, mu, sigma, k=0.5, h=5.0):
    """
    Two-sided tabular CUSUM on the standardised residual.

    CUSUM accumulates evidence, so it finds a shift too small to breach an EWMA
    limit on any single sample provided it persists. The two charts are run
    together and their alarms combined, because they fail in different ways.

    Parameters
    ----------
    residuals : pd.Series
        Observed minus expected, indexed by timestamp.
    mu, sigma : float
        Reference centre and scale, normally from ``reference_stats``.
    k : float, optional
        Slack in standard deviations; conventionally half the shift to be
        detected quickly. Default ``0.5``.
    h : float, optional
        Decision interval in standard deviations. Default ``5.0``.

    Returns
    -------
    pd.DataFrame
        Columns ``z``, ``cusum_high``, ``cusum_low``, ``limit`` and ``alarm``.
        ``cusum_low`` is reported as a positive magnitude.

    Raises
    ------
    ValueError
        When ``sigma`` is not finite or not positive: a degenerate reference
        window cannot standardise a residual, and reporting an all-``False``
        alarm column in that case would read as a quiet structure rather than
        as the broken reference window it actually is.
    """
    z = _standardise(residuals, mu, sigma)
    high = np.zeros(len(z))
    low = np.zeros(len(z))

    running_high = 0.0
    running_low = 0.0
    for position, value in enumerate(z.to_numpy()):
        if np.isfinite(value):
            running_high = max(0.0, running_high + value - k)
            running_low = max(0.0, running_low - value - k)
        high[position] = running_high
        low[position] = running_low

    chart = pd.DataFrame({'z': z, 'cusum_high': high, 'cusum_low': low,
                          'limit': float(h)}, index=z.index)
    chart['alarm'] = (chart['cusum_high'] > h) | (chart['cusum_low'] > h)
    return chart


def joint_alarm(ewma_alarm, cusum_alarm, window='24h'):
    """
    Alarms that both charts raise within a coincidence window.

    Requiring coincidence trades a little sensitivity for a large reduction in
    isolated false alarms, which is the right trade for a monitoring system a
    person has to trust.

    Parameters
    ----------
    ewma_alarm, cusum_alarm : pd.Series
        Boolean alarm series on a shared index.
    window : str, optional
        Coincidence window, as a pandas offset string. Default ``'24h'``.

    Returns
    -------
    pd.Series
        Boolean, true where both charts alarmed within ``window`` of each other.
    """
    left = ewma_alarm.fillna(False).astype(bool)
    right = cusum_alarm.reindex(left.index).fillna(False).astype(bool)

    span = pd.Timedelta(window)
    nearby_left = left.rolling(span, center=True, min_periods=1).max().astype(bool)
    nearby_right = right.rolling(span, center=True, min_periods=1).max().astype(bool)
    return ((left & nearby_right) | (right & nearby_left)).rename('alarm')


def alarm_episodes(alarm, residuals=None):
    """
    Consecutive alarming samples collapsed into one row each.

    Parameters
    ----------
    alarm : pd.Series
        Boolean alarm series indexed by timestamp on a regular grid.
    residuals : pd.Series or None, optional
        Standardised residual, used to describe each episode's size. Default
        ``None``.

    Returns
    -------
    pd.DataFrame
        Columns ``start``, ``end``, ``duration_h``, ``n_slots``, ``mean_z`` and
        ``peak_abs_z``. Empty with those columns when nothing alarmed.

    Raises
    ------
    ValueError
        When ``alarm``'s index is not a regular grid. ``duration_h`` is a slot
        count times one spacing, which only means what it says when the
        spacing is constant throughout.
    """
    columns = ['start', 'end', 'duration_h', 'n_slots', 'mean_z', 'peak_abs_z']
    flags = alarm.fillna(False).astype(bool)
    index = pd.DatetimeIndex(flags.index)
    positions = np.flatnonzero(flags.to_numpy())
    if positions.size == 0:
        return pd.DataFrame(columns=columns)

    step_hours = _regular_step_hours(index)
    breaks = np.flatnonzero(np.diff(positions) != 1)
    starts = positions[np.r_[0, breaks + 1]]
    ends = positions[np.r_[breaks, positions.size - 1]]

    rows = []
    for start, end in zip(starts, ends):
        window = (residuals.iloc[start:end + 1]
                  if residuals is not None else None)
        rows.append({
            'start': index[start],
            'end': index[end],
            'duration_h': (end - start + 1) * step_hours,
            'n_slots': int(end - start + 1),
            'mean_z': float(window.mean()) if window is not None else np.nan,
            'peak_abs_z': (float(window.abs().max())
                           if window is not None else np.nan),
        })
    return pd.DataFrame(rows, columns=columns)


def average_run_length(alarm, freq='20min'):
    """
    Watched time per alarm episode - the false-alarm budget, in plain units.

    Reported on an in-control stretch, this is the number a deployment is
    actually tuned to: "one false alarm every N days". Every other detection
    figure in a study is only comparable at a stated run length.

    Parameters
    ----------
    alarm : pd.Series
        Boolean alarm series indexed by timestamp.
    freq : str, optional
        Spacing of the series. Default ``'20min'``.

    Returns
    -------
    dict
        ``n_episodes``, ``hours``, ``arl_hours``, ``arl_days``. The run lengths
        are infinite when nothing alarmed.

    Raises
    ------
    ValueError
        When ``alarm``'s index is not a regular grid, or when ``freq``
        disagrees with the spacing the index actually has. ``hours`` is a slot
        count times one spacing, which only means what it says when the two
        agree.
    """
    episodes = alarm_episodes(alarm)
    step_hours = _regular_step_hours(alarm.index, freq)
    hours = float(len(alarm) * step_hours)
    count = int(len(episodes))
    arl_hours = hours / count if count else np.inf
    return {'n_episodes': count, 'hours': hours,
            'arl_hours': arl_hours, 'arl_days': arl_hours / 24.0}
