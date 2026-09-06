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


def detectability_curve(residuals, mu, sigma, magnitudes, durations,
                        freq='20min', lam=0.2, L=3.0, k=0.5, h=5.0, seed=0,
                        kind='pulse', period='24h', response_window='24h'):
    """
    Whether a departure of each size and length is found, and how late.

    For every pair, a step of that magnitude lasting that long is injected into
    the middle of the residual, both charts are run, and the joint alarm is
    compared against the alarm the uncontaminated record raises on its own
    within the same span. Only an alarm the clean run does not also raise
    counts as a detection; a chart that would have fired in that slot
    regardless of the injection has not found the injection, and counting it
    would report a sensitivity the detector does not have. The search is
    bounded to the injection window plus ``response_window``, so an unrelated
    alarm far down the record cannot be attributed to the injection either.
    The reference statistics are the caller's, estimated once on the
    uncontaminated record, so that the detector is never re-tuned to the
    anomaly it is being asked to find.

    Parameters
    ----------
    residuals : pd.Series
        Uncontaminated residual, indexed by timestamp.
    mu, sigma : float
        Reference centre and scale from ``reference_stats``.
    magnitudes : sequence of float
        Departure sizes in the residual's units.
    durations : sequence of str or pd.Timedelta
        How long each departure persists.
    freq : str, optional
        Spacing of the residual. Default ``'20min'``.
    lam, L : float, optional
        EWMA settings, as in ``ewma_chart``. Defaults ``0.2`` and ``3.0``.
    k, h : float, optional
        CUSUM settings, as in ``cusum_chart``. Defaults ``0.5`` and ``5.0``.
    seed : int, optional
        Reserved for future randomised placement; the injection point is
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

    Returns
    -------
    pd.DataFrame
        Columns ``magnitude``, ``duration_h``, ``detected`` and ``delay_h``,
        one row per pair. ``delay_h`` is missing where nothing alarmed.
    """
    values = pd.to_numeric(residuals, errors='coerce')
    index = pd.DatetimeIndex(values.index)
    injection = index[len(index) // 2]

    baseline = joint_alarm(
        ewma_chart(values, mu, sigma, lam=lam, L=L)['alarm'],
        cusum_chart(values, mu, sigma, k=k, h=h)['alarm'], window=freq)

    rows = []
    for magnitude in magnitudes:
        for duration in durations:
            span = pd.Timedelta(duration)
            contaminated = inject_anomaly(
                values, kind, float(magnitude), start=injection,
                duration=span, freq=freq, period=period)

            ewma = ewma_chart(contaminated, mu, sigma, lam=lam, L=L)
            cusum = cusum_chart(contaminated, mu, sigma, k=k, h=h)
            alarm = joint_alarm(ewma['alarm'], cusum['alarm'], window=freq)

            horizon = injection + span + pd.Timedelta(response_window)
            fired = alarm.loc[injection:horizon]
            # An alarm counts only where the uncontaminated run is silent. A
            # chart that would have raised this slot anyway has not detected
            # the injection, and counting it would report a sensitivity the
            # detector does not have.
            attributable = fired.astype(bool) & ~baseline.loc[
                injection:horizon].astype(bool)
            hit = attributable[attributable].index
            detected = len(hit) > 0
            rows.append({
                'magnitude': float(magnitude),
                'duration_h': float(span / pd.Timedelta(hours=1)),
                'detected': bool(detected),
                'delay_h': (float((hit[0] - injection) / pd.Timedelta(hours=1))
                            if detected else np.nan),
            })
    return pd.DataFrame(
        rows, columns=['magnitude', 'duration_h', 'detected', 'delay_h'])


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
