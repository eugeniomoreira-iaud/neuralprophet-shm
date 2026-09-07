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

from . import coupling

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
        values = values.loc[start:]
    if end is not None:
        values = values.loc[:end]

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


def inject_anomaly(series, kind, magnitude, start, duration=None,
                   freq='20min', period='24h'):
    """
    Add a synthetic departure of known size and shape to a series.

    A detector's sensitivity cannot be read off a record that contains one real
    event. Injecting departures of known size is how the question "what is the
    smallest movement this would find" gets a number rather than an opinion.

    Parameters
    ----------
    series : pd.Series
        Signal to contaminate, indexed by timestamp. Not modified in place.
    kind : {'step', 'ramp', 'pulse', 'amplitude', 'phase', 'drift'}
        ``'step'`` shifts every sample from ``start`` onward; ``'ramp'`` rises
        linearly to ``magnitude`` over ``duration`` and holds it; ``'pulse'``
        shifts only the samples inside ``duration``. ``'amplitude'`` and
        ``'phase'`` establish themselves linearly over ``duration`` and then
        hold, modulating the cycle named by ``period`` rather than shifting the
        level: ``'amplitude'`` grows the swing of that cycle, standing for a
        wall that bends further under the same forcing once the leaves stop
        acting together, and ``'phase'`` is its quadrature partner, standing
        for a changed thermal path - such as water in the core - that answers
        the same forcing earlier or later without answering it more strongly.
        ``'drift'`` takes ``magnitude`` as a rate per year, not a size, and
        needs no ``duration``: it accumulates to the end of the record, the
        shape of mortar creep, thermal ratcheting or settlement.
    magnitude : float
        Size of the departure, in the series' own units, except for
        ``'drift'``, where it is a rate in units per year.
    start : pd.Timestamp or str
        When the departure begins.
    duration : str or pd.Timedelta or None, optional
        Length of the ramp, pulse, or amplitude/phase build-up. Required for
        those kinds, ignored for a step or a drift. Default ``None``.
    freq : str, optional
        Not read: the injection is placed by timestamp arithmetic, not by grid
        position. Accepted so a caller can pass the study's grid uniformly with
        the rest of this module. Default ``'20min'``.
    period : str or pd.Timedelta, optional
        The cycle that the ``'amplitude'`` and ``'phase'`` kinds modulate.
        Ignored by every other kind. Default ``'24h'``, the daily cycle a
        thermally driven inclination record follows.

    Returns
    -------
    pd.Series
        A copy of ``series`` with the departure added.
    """
    kinds = {'step', 'ramp', 'pulse', 'amplitude', 'phase', 'drift'}
    if kind not in kinds:
        raise ValueError(
            "kind must be one of 'step', 'ramp', 'pulse', 'amplitude', "
            "'phase' or 'drift'")
    if kind in {'ramp', 'pulse', 'amplitude', 'phase'} and duration is None:
        raise ValueError(f"kind '{kind}' requires a duration")

    out = series.copy()
    index = pd.DatetimeIndex(out.index)
    begin = pd.Timestamp(start)
    after = index >= begin

    if kind == 'step':
        out.loc[after] = out.loc[after] + magnitude
        return out

    if kind == 'drift':
        # A rate, not a size: the departure keeps accumulating to the end of the
        # record, which is what creep and settlement do.
        years = ((index[after] - begin)
                 / pd.Timedelta(days=365.25)).to_numpy()
        out.loc[after] = out.loc[after] + magnitude * years
        return out

    span = pd.Timedelta(duration)
    inside = after & (index < begin + span)

    if kind == 'pulse':
        out.loc[inside] = out.loc[inside] + magnitude
        return out

    if kind in {'amplitude', 'phase'}:
        # Both modulate the same cycle and differ only by quadrature: a growing
        # swing is in phase with the response, a timing change is a quarter
        # cycle away from it. The envelope rises linearly over `duration` and
        # then holds, so `magnitude` is the size the departure settles at.
        cycles = ((index[after] - begin) / pd.Timedelta(period)).to_numpy()
        envelope = np.clip(
            ((index[after] - begin) / span).to_numpy(), 0.0, 1.0)
        angle = 2.0 * np.pi * cycles
        wave = np.sin(angle) if kind == 'amplitude' else np.cos(angle)
        out.loc[after] = out.loc[after] + magnitude * envelope * wave
        return out

    elapsed = (index[after] - begin) / span
    profile = np.clip(elapsed, 0.0, 1.0) * magnitude
    out.loc[after] = out.loc[after] + profile
    return out


def phase_shift_amplitude(daily_amplitude, shift_hours, period_hours=24.0):
    """
    The residual amplitude implied by a timing shift of a periodic response.

    A wall whose thermal path has changed answers the same forcing later or
    earlier without necessarily answering it more strongly. Subtracting the
    unshifted cycle from the shifted one leaves a harmonic in quadrature whose
    amplitude is the chord of the shift, ``2 A sin(pi dt / P)``. This converts
    the quantity an engineer states — a lag change in hours — into the
    millidegree amplitude a detector actually sees.

    Parameters
    ----------
    daily_amplitude : float
        Amplitude of the fitted periodic component, in the series' units. Half
        its peak-to-peak range.
    shift_hours : float
        Timing shift, in hours. Sign is irrelevant: a lead and a lag of the same
        size leave the same amplitude.
    period_hours : float, optional
        Period of the component. Default ``24.0``.

    Returns
    -------
    float
        Amplitude of the residual harmonic, in the series' units.
    """
    return float(2.0 * abs(daily_amplitude)
                 * abs(np.sin(np.pi * float(shift_hours) / float(period_hours))))


def _monitor_statistic(series, statistic, phi=None, freq='20min', min_slots=60):
    """Reduce a residual to the series a chart is built on, and say its grid."""
    if statistic == 'residual':
        return series, freq
    if statistic == 'innovation':
        return prewhiten(series, phi=phi, freq=freq)[0], freq
    if statistic in ('daily_amplitude', 'daily_phase'):
        column = 'amplitude' if statistic == 'daily_amplitude' else 'phase_h'
        return daily_harmonic(series, min_slots=min_slots)[column], '1D'
    if statistic == 'daily_mean':
        return series.resample('1D').mean(), '1D'
    raise ValueError(f'unknown statistic {statistic!r}')


def detectability_curve(residuals, mu, sigma, magnitudes, durations,
                        freq='20min', lam=0.2, L=3.0, k=0.5, h=5.0, seed=0,
                        kind='pulse', period='24h', response_window='24h',
                        statistic='residual', phi=None, injection_starts=None,
                        min_slots=60):
    """
    Whether a departure of each size and length is found, and how late.

    For every pair, a departure of that magnitude and length is injected at
    one or more points of the residual, the statistic each chart is actually
    built on is charted, and its joint alarm is compared against the alarm the
    uncontaminated record raises on its own within the same span. Only an
    alarm the clean run does not also raise counts as a detection; a chart
    that would have fired in that slot regardless of the injection has not
    found the injection, and counting it would report a sensitivity the
    detector does not have. The search is bounded to the injection window plus
    ``response_window``, so an unrelated alarm far down the record cannot be
    attributed to the injection either. The reference statistics are the
    caller's, estimated once on the uncontaminated record, so that the
    detector is never re-tuned to the anomaly it is being asked to find.

    Parameters
    ----------
    residuals : pd.Series
        Uncontaminated residual, indexed by timestamp, on its native grid.
        The raw residual is always what is contaminated; ``statistic`` says
        what is charted afterwards.
    mu, sigma : float
        Reference centre and scale from ``reference_stats``, estimated on the
        same statistic named by ``statistic``.
    magnitudes : sequence of float
        Departure sizes in the charted statistic's units.
    durations : sequence of str or pd.Timedelta
        How long each departure persists.
    freq : str, optional
        Spacing of ``residuals``. Default ``'20min'``.
    lam, L : float, optional
        EWMA settings, as in ``ewma_chart``. Defaults ``0.2`` and ``3.0``.
    k, h : float, optional
        CUSUM settings, as in ``cusum_chart``. Defaults ``0.5`` and ``5.0``.
    seed : int, optional
        Reserved for future randomised placement; the injection points are
        currently deterministic. Default ``0``.
    kind : str, optional
        Shape passed to ``inject_anomaly`` for every point of the sweep.
        Default ``'pulse'``, which reproduces the sweep's original behaviour: a
        magnitude held for the full duration. ``'drift'`` reads ``magnitude``
        as a rate per year rather than a size, so for that kind ``durations``
        sets how long the drift is watched, not how long it lasts.
    period : str or pd.Timedelta, optional
        Cycle modulated by the ``'amplitude'`` and ``'phase'`` kinds, passed
        through to ``inject_anomaly``. Ignored by every other kind. Default
        ``'24h'``.
    response_window : str, optional
        How long after the departure ends an alarm still counts as having
        found it, as a pandas offset string. Default ``'24h'``.
    statistic : {'residual', 'innovation', 'daily_amplitude', 'daily_phase', 'daily_mean'}, optional
        What is charted. ``'residual'`` (default) reproduces the sweep's
        original behaviour, charting the residual itself at ``freq``.
        ``'innovation'`` prewhitens both the contaminated and the
        uncontaminated residual with ``phi`` before charting, at ``freq``.
        ``'daily_amplitude'`` and ``'daily_phase'`` chart the ``amplitude`` or
        ``phase_h`` column of ``daily_harmonic``, and ``'daily_mean'`` charts
        the daily mean; all three run on a daily grid, so for them the joint
        window is one day and ``delay_h`` is counted in days times 24.
    phi : float or None, optional
        AR(1) coefficient passed to ``prewhiten`` when ``statistic`` is
        ``'innovation'``. Ignored otherwise. Default ``None``.
    injection_starts : sequence of str or pd.Timestamp or None, optional
        Where to inject the departure. ``None`` (default) keeps the original
        single injection at the middle of the record. A sequence sweeps the
        injection once per start, and ``detected`` becomes the fraction of
        starts detected and ``delay_h`` their mean delay.
    min_slots : int, optional
        Passed to ``daily_harmonic`` for the ``'daily_amplitude'`` and
        ``'daily_phase'`` statistics. Default ``60``.

    Returns
    -------
    pd.DataFrame
        Columns ``magnitude``, ``duration_h``, ``detected`` and ``delay_h``,
        one row per pair. ``detected`` is the fraction of injection points
        detected — ``1.0`` or ``0.0`` for the single default point, which
        keeps ``bool(...)`` working for a caller written against the
        original behaviour. ``delay_h`` is the mean delay over the points
        that detected, missing where none did. When ``statistic`` is not
        ``'residual'`` or several ``injection_starts`` are swept, two more
        columns are carried: ``statistic`` and ``n_starts``, the count of
        injection points the row was swept over.
    """
    values = pd.to_numeric(residuals, errors='coerce')
    index = pd.DatetimeIndex(values.index)
    if injection_starts is None:
        points = pd.DatetimeIndex([index[len(index) // 2]])
    else:
        points = pd.DatetimeIndex(injection_starts)
        # A caller names injection dates as plain, zone-free strings; the
        # residual's own index carries whatever zone it was built on. The
        # two must agree before either is compared with the other.
        if index.tz is not None and points.tz is None:
            points = points.tz_localize(index.tz)
        elif index.tz is None and points.tz is not None:
            points = points.tz_convert(None)

    base_values, chart_freq = _monitor_statistic(values, statistic, phi, freq, min_slots)
    baseline = joint_alarm(
        ewma_chart(base_values, mu, sigma, lam=lam, L=L)['alarm'],
        cusum_chart(base_values, mu, sigma, k=k, h=h)['alarm'], window=chart_freq)

    # The original four columns are always present; the two describing the
    # sweep itself are added only when the sweep departs from the original
    # single-point, residual-statistic behaviour, so that a caller written
    # against that behaviour sees exactly the frame it always has.
    extended = statistic != 'residual' or injection_starts is not None
    columns = (['magnitude', 'duration_h', 'detected', 'delay_h', 'statistic', 'n_starts']
              if extended else ['magnitude', 'duration_h', 'detected', 'delay_h'])

    rows = []
    for magnitude in magnitudes:
        for duration in durations:
            span = pd.Timedelta(duration)
            hits = 0
            delays = []
            for point in points:
                contaminated = inject_anomaly(
                    values, kind, float(magnitude), start=point,
                    duration=span, freq=freq, period=period)
                chart_values, _ = _monitor_statistic(
                    contaminated, statistic, phi, freq, min_slots)

                ewma = ewma_chart(chart_values, mu, sigma, lam=lam, L=L)
                cusum = cusum_chart(chart_values, mu, sigma, k=k, h=h)
                alarm = joint_alarm(ewma['alarm'], cusum['alarm'], window=chart_freq)

                horizon = point + span + pd.Timedelta(response_window)
                fired = alarm.loc[point:horizon]
                # An alarm counts only where the uncontaminated run is silent.
                # A chart that would have raised this slot anyway has not
                # detected the injection, and counting it would report a
                # sensitivity the detector does not have.
                attributable = fired.astype(bool) & ~baseline.loc[
                    point:horizon].astype(bool)
                hit = attributable[attributable].index
                if len(hit) > 0:
                    hits += 1
                    delays.append(float((hit[0] - point) / pd.Timedelta(hours=1)))

            row = {
                'magnitude': float(magnitude),
                'duration_h': float(span / pd.Timedelta(hours=1)),
                'detected': hits / len(points),
                'delay_h': float(np.mean(delays)) if delays else np.nan,
            }
            if extended:
                row['statistic'] = statistic
                row['n_starts'] = len(points)
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def daily_harmonic(series, min_slots=60, period_hours=24.0):
    """
    Amplitude and phase of the daily cycle, one row per calendar day.

    A 24-hour harmonic and its 12-hour companion are fitted to each day by
    least squares. The amplitude of the 24-hour term is the size of the
    daily swing; its phase is the hour at which that term peaks. On a
    residual these two series are the daily chart's statistics (spec D10);
    on the diurnal band of the target they are the measured daily cycle the
    harmonic diagnostic fits against day of year (spec D6).

    Parameters
    ----------
    series : pd.Series
        Signal indexed by UTC timestamp on a regular sub-daily grid.
    min_slots : int, optional
        Fewest finite samples a day needs to be fitted. Default ``60``.
    period_hours : float, optional
        Period of the fundamental. Default ``24.0``.

    Returns
    -------
    pd.DataFrame
        Indexed by day (UTC midnight): ``amplitude``, ``phase_h`` in
        ``[0, period_hours)``, ``amplitude_12h``, ``n``, ``r2``.
    """
    values = pd.to_numeric(series, errors='coerce')
    index = pd.DatetimeIndex(values.index)
    omega = 2.0 * np.pi / float(period_hours)
    rows = []
    for day, chunk in values.groupby(index.floor('D')):
        chunk = chunk.dropna()
        if len(chunk) < int(min_slots):
            continue
        t = (chunk.index.hour + chunk.index.minute / 60.0
             + chunk.index.second / 3600.0).to_numpy(dtype=float)
        design = np.column_stack([
            np.ones_like(t), np.cos(omega * t), np.sin(omega * t),
            np.cos(2 * omega * t), np.sin(2 * omega * t)])
        y = chunk.to_numpy(dtype=float)
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ coef
        total = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float(((y - fitted) ** 2).sum()) / total if total > 0 else np.nan
        a1, b1, a2, b2 = coef[1], coef[2], coef[3], coef[4]
        rows.append({
            'day': day,
            'amplitude': float(np.hypot(a1, b1)),
            'phase_h': float((np.arctan2(b1, a1) / omega) % period_hours),
            'amplitude_12h': float(np.hypot(a2, b2)),
            'n': int(len(chunk)),
            'r2': r2,
        })
    out = pd.DataFrame(rows, columns=['day', 'amplitude', 'phase_h',
                                      'amplitude_12h', 'n', 'r2'])
    return out.set_index('day')


def prewhiten(residuals, phi=None, start=None, end=None, freq='20min'):
    """
    The part of the residual its own previous value could not predict.

    Control-chart limits are derived for independent samples, and this
    project's residual is nothing of the kind (lag-1 autocorrelation 0.995
    at 20 minutes in Study 04). Charting ``e(t) = r(t) - phi r(t - 1)``
    restores the assumption for sudden departures; slow ones need the daily
    and slow charts instead (spec D10).

    Parameters
    ----------
    residuals : pd.Series
        Residual on a regular grid.
    phi : float or None, optional
        AR(1) coefficient. ``None`` estimates it as the lag-1 autocorrelation
        over ``[start, end]``. Default ``None``.
    start, end : str or pd.Timestamp or None, optional
        Reference window for the estimate.
    freq : str, optional
        Grid spacing; a previous sample farther than this is a gap. Default
        ``'20min'``.

    Returns
    -------
    (pd.Series, float)
        Innovations aligned to ``residuals``, and the ``phi`` used.
    """
    values = pd.to_numeric(residuals, errors='coerce')
    if phi is None:
        window = values.loc[start:end].dropna()
        phi = float(window.autocorr(1))
    previous = values.shift(1, freq=freq).reindex(values.index)
    innovations = values - float(phi) * previous
    return innovations, float(phi)


def channel_coincidence(alarm, channels, scale_start=None, scale_end=None,
                        window='24h', threshold=5.0, instrument=('batt',),
                        environment=('tair', 'rh')):
    """
    What else moved in the slot of each alarm: the instrument, the weather, or nothing.

    Study 01 settled the summer 2026 excursions by asking whether the
    environmental and supply channels carried them too. This applies that
    test to every fast-chart alarm, so the episode table can say which alarms
    are the wall's to answer for (spec D10).

    Parameters
    ----------
    alarm : pd.Series
        Boolean alarm series indexed by timestamp.
    channels : pd.DataFrame
        One column per candidate channel, on the same grid as ``alarm``.
        Every column named in ``instrument`` or ``environment`` is read from
        here; a name absent from ``channels`` is silently skipped.
    scale_start, scale_end : pd.Timestamp or str or None, optional
        Reference window the excursion scale is estimated on. ``None``
        extends to the respective end of the series. Default ``None``.
    window : str, optional
        Width of each channel's centred rolling median, as a pandas offset
        string. Default ``'24h'``.
    threshold : float, optional
        Number of scaled departures a slot must exceed to count as an
        excursion. Default ``5.0``.
    instrument : sequence of str, optional
        Channel names whose excursion attributes an alarm to the instrument.
        Default ``('batt',)``.
    environment : sequence of str, optional
        Channel names whose excursion attributes an alarm to the
        environment, checked after ``instrument``. Default ``('tair',
        'rh')``.

    Returns
    -------
    pd.Series
        ``'instrument'``, ``'environment'`` or ``'unattributed'`` for every
        slot where ``alarm`` is true.
    """
    alarm = alarm.astype(bool)
    excursions = {}
    for column in channels.columns:
        series = pd.to_numeric(channels[column], errors='coerce')
        departure = series - series.rolling(window, center=True, min_periods=1).median()
        scale = 1.4826 * departure.loc[scale_start:scale_end].abs().median()
        excursions[column] = departure.abs() > float(threshold) * float(scale)
    excursions = pd.DataFrame(excursions).reindex(alarm.index).fillna(False)
    labels = pd.Series('unattributed', index=alarm.index[alarm])
    env = excursions[[c for c in environment if c in excursions]].any(axis=1)
    ins = excursions[[c for c in instrument if c in excursions]].any(axis=1)
    labels[env.reindex(labels.index).fillna(False).to_numpy()] = 'environment'
    labels[ins.reindex(labels.index).fillna(False).to_numpy()] = 'instrument'
    return labels


def chart_series(residual, freq, min_slots, start, end):
    """
    The four series Study 05's monitor charts are actually built on.

    One rolling residual feeds three time scales: the fast chart's
    prewhitened innovations, the daily chart's amplitude and phase of the
    daily cycle, and the slow chart's daily mean. Building all four here,
    from one call, keeps the notebook's tuning loop free of the choice of
    which transform belongs to which chart (spec D10).

    Parameters
    ----------
    residual : pd.Series
        The rolling residual, on its native grid.
    freq : str
        Spacing of ``residual``, passed to ``prewhiten``.
    min_slots : int
        Fewest samples a day needs, passed to ``daily_harmonic``.
    start, end : str or pd.Timestamp
        Reference window ``prewhiten`` estimates its AR(1) coefficient over.

    Returns
    -------
    (dict, float)
        ``{'fast', 'daily_amplitude', 'daily_phase', 'slow'}`` mapped to
        their series, and the ``phi`` the fast chart's innovations were
        computed with. The daily and slow series are on a complete,
        regular one-day grid spanning ``residual``'s own extent: a day
        ``daily_harmonic`` could not fit is carried as a missing value at
        its place, never as an absent row.

    Notes
    -----
    ``daily_harmonic`` omits a calendar day outright when it holds fewer
    than ``min_slots`` finite samples, which leaves its own output on an
    irregular grid the moment the window contains one such day. Every
    downstream chart function tolerates a missing value in the middle of
    a regular grid, but ``alarm_episodes`` and ``average_run_length`` —
    which ``tune_limit_to_budget`` and ``run_chart`` both call — refuse an
    irregular one outright, so the two daily series are reindexed onto a
    full calendar range here before being handed on.
    """
    innovations, phi = prewhiten(residual, start=start, end=end, freq=freq)
    daily_index = pd.date_range(residual.index.min().floor('D'),
                                residual.index.max().floor('D'), freq='1D')
    daily = daily_harmonic(residual, min_slots=min_slots).reindex(daily_index)
    series = {
        'fast': innovations,
        'daily_amplitude': daily['amplitude'],
        'daily_phase': daily['phase_h'],
        'slow': residual.resample('1D').mean(),
    }
    return series, phi


def tune_limit_to_budget(series, start, end, budget_days, candidates,
                         lam, k, h, joint_window, freq):
    """
    The smallest control limit whose joint alarm meets a false-alarm budget.

    Every candidate ``L`` is charted on the in-control reference stretch
    alone; the smallest one whose average run length reaches the budget is
    the limit a chart is deployed at, so that every chart's sensitivity is
    stated at the same false-alarm cost rather than chosen by eye (spec D10).

    Parameters
    ----------
    series : pd.Series
        The statistic one chart is built on (e.g. one of
        ``chart_series``'s outputs).
    start, end : str or pd.Timestamp
        Bounds of the in-control reference window.
    budget_days : float
        Watched days per false alarm the chosen limit must reach.
    candidates : sequence of float
        Control limits to sweep, in standard deviations, smallest first.
    lam, k, h : float
        EWMA smoothing constant and CUSUM slack and decision interval, as in
        ``ewma_chart`` and ``cusum_chart``.
    joint_window : str
        Coincidence window passed to ``joint_alarm``.
    freq : str
        Spacing of ``series``, passed to ``average_run_length``.

    Returns
    -------
    (float, pd.DataFrame)
        The smallest ``L`` meeting the budget, and the sweep table: one row
        per candidate, its ``L`` and everything ``average_run_length``
        returns.

    Raises
    ------
    RuntimeError
        When no candidate's average run length reaches ``budget_days`` — the
        reference window is not as quiet as the budget assumes, and widening
        the candidate grid would not fix that.
    """
    reference = reference_stats(series, start=start, end=end)
    in_control = series.loc[start:end]
    rows = []
    for candidate in candidates:
        ewma = ewma_chart(in_control, reference['mu'], reference['sigma'],
                          lam=lam, L=candidate)
        cusum = cusum_chart(in_control, reference['mu'], reference['sigma'],
                            k=k, h=h)
        joint = joint_alarm(ewma['alarm'], cusum['alarm'], window=joint_window)
        rows.append({'L': float(candidate), **average_run_length(joint, freq=freq)})
    sweep = pd.DataFrame(rows)

    meeting = sweep.loc[sweep['arl_days'] >= budget_days, 'L']
    if meeting.empty:
        raise RuntimeError(
            f'no candidate limit reaches the {budget_days:.1f}-day budget on '
            f'this reference window; widest achieved is '
            f'{sweep["arl_days"].max():.1f} days')
    return float(meeting.min()), sweep


def run_chart(series, reference, L, lam, k, h, joint_window, monitored_start, freq):
    """
    The tuned EWMA and CUSUM charts, their joint alarm and its episodes.

    Parameters
    ----------
    series : pd.Series
        The statistic one chart is built on.
    reference : dict
        Output of ``reference_stats``: ``mu`` and ``sigma`` the charts are
        standardised against.
    L : float
        Control limit, in standard deviations, normally from
        ``tune_limit_to_budget``.
    lam, k, h : float
        EWMA smoothing constant and CUSUM slack and decision interval.
    joint_window : str
        Coincidence window passed to ``joint_alarm``.
    monitored_start : str or pd.Timestamp
        First instant scored; ``series`` before it is the reference window
        and is not charted here.
    freq : str
        Spacing of ``series``. Accepted for interface symmetry with the
        other monitor functions; not read directly, since neither
        ``ewma_chart``, ``cusum_chart`` nor ``joint_alarm`` needs it.

    Returns
    -------
    dict
        ``ewma``, ``cusum``, ``joint`` and ``episodes`` — the last from
        ``alarm_episodes(joint, standardised)``, so an episode's ``mean_z``
        and ``peak_abs_z`` are genuinely standard deviations of the
        reference, not the chart's own native units under a name that
        implies otherwise.
    """
    watched = series.loc[monitored_start:]
    ewma = ewma_chart(watched, reference['mu'], reference['sigma'], lam=lam, L=L)
    cusum = cusum_chart(watched, reference['mu'], reference['sigma'], k=k, h=h)
    joint = joint_alarm(ewma['alarm'], cusum['alarm'], window=joint_window)
    standardised = _standardise(watched, reference['mu'], reference['sigma'])
    episodes = alarm_episodes(joint, standardised)
    return {'ewma': ewma, 'cusum': cusum, 'joint': joint, 'episodes': episodes}


def attribute_episodes(episodes, labels, default='unattributed'):
    """
    Every alarm episode's attribution: the mode of ``labels`` inside its span.

    Parameters
    ----------
    episodes : pd.DataFrame
        Output of ``alarm_episodes``: at least ``start`` and ``end`` columns.
    labels : pd.Series
        Per-slot attribution, normally ``channel_coincidence``'s output —
        indexed only where an alarm fired, so an episode with no matching
        index entry is exactly an episode ``channel_coincidence`` was never
        asked about.
    default : str, optional
        Attribution used where no label falls inside an episode's span.
        Default ``'unattributed'``.

    Returns
    -------
    pd.DataFrame
        ``episodes`` with an added ``attribution`` column.
    """
    out = episodes.copy()
    attributions = []
    for start, end in zip(out['start'], out['end']):
        window = labels.loc[start:end]
        attributions.append(window.mode().iloc[0] if not window.empty else default)
    out['attribution'] = attributions
    return out


def daily_response_amplitude(components, dates, window=72, min_slots=60,
                             driver='future_regressor_tair'):
    """
    The size of the wall's own daily response, averaged over a set of dates.

    A phase-shift injection must be sized against something the wall
    actually does, not against an arbitrary millidegree figure: the driver's
    fitted component plus every conditional or plain daily-seasonal
    component it decomposes into is the model's own account of the daily
    response, and its diurnal band's daily harmonic amplitude on the named
    dates is what a timing change of that response would look like (spec
    D11).

    Parameters
    ----------
    components : pd.DataFrame
        A decomposition frame such as ``components_a['str']``, carrying
        ``driver`` and every column starting with ``'season_daily'``.
    dates : sequence of str or pd.Timestamp
        Calendar dates the amplitude is averaged over.
    window : int, optional
        Rolling-mean width, in samples, passed to
        ``shmlib.coupling.diurnal_band``. Default ``72``, one day at twenty
        minutes.
    min_slots : int, optional
        Passed to ``daily_harmonic``. Default ``60``.
    driver : str, optional
        Column carrying the driver's fitted component. Default
        ``'future_regressor_tair'``.

    Returns
    -------
    float
        The daily response's harmonic amplitude, averaged over the days
        named by ``dates``.
    """
    seasonal_columns = [c for c in components.columns if c.startswith('season_daily')]
    response = components[driver] + components[seasonal_columns].sum(axis=1)
    band = coupling.diurnal_band(response, window=window)
    amplitude = daily_harmonic(band, min_slots=min_slots)['amplitude']
    days = pd.DatetimeIndex([pd.Timestamp(d).floor('D') for d in dates])
    return float(amplitude.reindex(days).mean())


def detectability_by_mechanism(reference_residual, tuned, specs, mechanisms,
                               magnitudes, durations, freq, k, h, phi,
                               response_window, injection_starts, min_slots,
                               seed=0, all_charts=False):
    """
    Detectability swept once per damage mechanism, on its own chart.

    Each mechanism has one chart and one statistic it is scored on — an
    amplitude growth and a phase shift on the daily chart, a drift on the
    slow chart, a step on the fast chart (spec D11) — and this calls
    ``detectability_curve`` once per mechanism with that chart's tuned limit
    and smoothing constant, so the notebook's detectability cell is a single
    call rather than a hand-written loop over the four mechanisms.

    Parameters
    ----------
    reference_residual : pd.Series
        The uncontaminated residual, restricted to the in-control reference
        window: what every mechanism's sweep injects into.
    tuned : dict of dict
        Keyed by chart name; each entry carries at least ``reference``
        (``reference_stats``' output) and ``L``, normally from
        ``tune_limit_to_budget``.
    specs : dict of dict
        Keyed by chart name; each entry carries at least ``lam``.
    mechanisms : dict
        Mechanism name to ``(chart_name, statistic)``, e.g.
        ``{'step': ('fast', 'innovation')}``. The chart named for a
        mechanism is that mechanism's own, primary chart; the statistic
        named alongside a chart is also read as that chart's own statistic
        when ``all_charts`` sweeps a mechanism onto a chart that is not its
        own.
    magnitudes : dict
        Mechanism name to the sequence of magnitudes swept for it.
    durations : sequence of str or pd.Timedelta, or dict
        Durations swept. A plain sequence is shared by every mechanism, as
        before; a dict maps a mechanism name to its own sequence, so a
        mechanism whose damage accumulates on a different timescale — a
        drift, watched for weeks rather than hours — is not forced through
        the same duration grid as the others.
    freq : str
        Spacing of ``reference_residual``.
    k, h : float
        CUSUM slack and decision interval, shared by every mechanism.
    phi : float
        AR(1) coefficient, passed through for the ``'innovation'`` statistic.
    response_window : str
        How long after a departure ends an alarm still counts as having
        found it.
    injection_starts : sequence of str or pd.Timestamp or None
        Injection dates, shared by every mechanism. ``None`` keeps the
        single mid-record injection.
    min_slots : int
        Passed through for the daily statistics.
    seed : int, optional
        Passed through to ``detectability_curve``. Default ``0``.
    all_charts : bool, optional
        When ``True``, each mechanism is swept not only on its own chart
        but on every other chart named in ``mechanisms``' values too, each
        with that chart's own statistic, tuned limit and smoothing
        constant — the report's own promise that every mechanism is
        "scored on the chart built for it, with the other two reported as
        well". Default ``False``, which sweeps the same rows as the
        original one-chart-per-mechanism behaviour; the ``primary`` column
        is present either way, and is ``True`` on every row at the default.

    Returns
    -------
    pd.DataFrame
        The concatenation of every mechanism's ``detectability_curve``
        calls, each with its own ``mechanism`` and ``chart`` columns, plus
        a boolean ``primary`` column that is ``True`` on the rows scored on
        the mechanism's own chart.
    """
    chart_statistic = {chart_name: statistic
                       for chart_name, statistic in mechanisms.values()}
    rows = []
    for mechanism, (own_chart, _) in mechanisms.items():
        mechanism_durations = (durations[mechanism] if isinstance(durations, dict)
                               else durations)
        charts_to_sweep = ([own_chart] + [name for name in chart_statistic
                                          if name != own_chart]
                           if all_charts else [own_chart])
        for chart_name in charts_to_sweep:
            fit = tuned[chart_name]
            curve = detectability_curve(
                reference_residual, fit['reference']['mu'], fit['reference']['sigma'],
                magnitudes=magnitudes[mechanism], durations=mechanism_durations,
                freq=freq, lam=specs[chart_name]['lam'], L=fit['L'], k=k, h=h,
                seed=seed, kind=mechanism, statistic=chart_statistic[chart_name],
                phi=phi, response_window=response_window,
                injection_starts=injection_starts, min_slots=min_slots)
            rows.append(curve.assign(mechanism=mechanism, chart=chart_name,
                                     primary=(chart_name == own_chart)))
    return pd.concat(rows, ignore_index=True)


def detection_thresholds(detectability):
    """
    The smallest departure each mechanism-and-chart pair actually catches.

    ``GM_13`` answers "was this specific magnitude and duration found"; a
    reader wants the single number a magnitude sweep exists to produce —
    the smallest departure size the chart catches at all, and the smallest
    it catches every time — read off the longest duration swept, which is
    the mechanism's best chance to be found.

    Parameters
    ----------
    detectability : pd.DataFrame
        Output of :func:`detectability_by_mechanism`: at least
        ``mechanism``, ``chart``, ``magnitude``, ``duration_h``,
        ``detected`` and ``delay_h``. A ``primary`` column, when present,
        is carried through unchanged rather than recomputed.

    Returns
    -------
    pd.DataFrame
        One row per ``(mechanism, chart)`` pair present in `detectability`:
        ``mechanism``, ``chart``, ``primary``, ``horizon_h`` (the longest
        ``duration_h`` swept for that mechanism), ``smallest_any`` (the
        smallest magnitude with ``detected > 0`` at that horizon, ``NaN``
        when none), ``smallest_all`` (the smallest magnitude with
        ``detected == 1`` at that horizon, ``NaN`` when none), and
        ``delay_h_at_smallest_all`` (that magnitude's own ``delay_h``,
        ``NaN`` when ``smallest_all`` is ``NaN``).
    """
    columns = ['mechanism', 'chart', 'primary', 'horizon_h', 'smallest_any',
              'smallest_all', 'delay_h_at_smallest_all']
    rows = []
    for (mechanism, chart_name), group in detectability.groupby(
            ['mechanism', 'chart'], sort=False):
        horizon_h = float(group['duration_h'].max())
        at_horizon = group.loc[group['duration_h'] == horizon_h].sort_values('magnitude')
        any_hit = at_horizon.loc[at_horizon['detected'] > 0]
        all_hit = at_horizon.loc[at_horizon['detected'] >= 1.0]
        primary = bool(group['primary'].iloc[0]) if 'primary' in group else True
        rows.append({
            'mechanism': mechanism,
            'chart': chart_name,
            'primary': primary,
            'horizon_h': horizon_h,
            'smallest_any': (float(any_hit['magnitude'].iloc[0])
                            if not any_hit.empty else np.nan),
            'smallest_all': (float(all_hit['magnitude'].iloc[0])
                             if not all_hit.empty else np.nan),
            'delay_h_at_smallest_all': (float(all_hit['delay_h'].iloc[0])
                                        if not all_hit.empty else np.nan),
        })
    return pd.DataFrame(rows, columns=columns)
