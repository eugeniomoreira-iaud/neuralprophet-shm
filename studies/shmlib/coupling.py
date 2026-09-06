"""
Module: shmlib.coupling

How one series moves with another, and after how long.

Every function here answers a version of the same question: given a response
and a candidate driver on a common grid, at what delay do they line up, how
strongly, and how much of the response does a unit of the driver buy. The
question recurs wherever two records of the same site are compared — a clock
offset between two sources is the same cross-correlation as a thermal lag
between a forcing and a deformation, differing only in what is being maximised
and over which range — so the search lives here once and its callers state the
range and the objective they want.

Three conventions run through the module and are not negotiable inside it.

**A lag is measured on the driver, and it is positive when the response
follows.** The pairing at lag ``L`` is ``driver.shift(L)`` against the
response, so a positive ``L`` says the response reproduces what the driver did
``L`` steps earlier. A negative ``L`` says the response moves first; whether
that is admissible is a property of the driver, not of the search, and the
caller expresses it by the range of lags it passes.

**Nothing is interpolated to manufacture an overlap.** Every statistic is
computed on the hours both series actually carry at that lag, and the count of
those hours is reported beside every number that rests on it.

**A correlation on an hourly geophysical series has autocorrelated residuals,**
so the ordinary least-squares standard error of a gain estimated from one is
too small — often by an order of magnitude. :func:`gain_at_lag` therefore
reports a heteroskedasticity- and autocorrelation-consistent (Newey-West)
standard error, with the truncation lag as an argument the study states rather
than a default it inherits.

What this module does not hold is any judgement about which lags are
admissible, which drivers are candidates, or what counts as a coupling worth
reporting. Those are the study's decisions and arrive as arguments: the lag
range per driver, the objective the scan maximises, the minimum overlap a
number needs, and the control channel a candidate is read against.
"""

import numpy as np
import pandas as pd

from . import site


#: Objective the lag search maximises.
#:
#: ``'absolute'`` maximises ``|r|`` and is what a physical coupling of unknown
#: sign wants: at the Gubbio site a heating driver is expected to move the
#: inclination *negatively*, so a search that maximised the raw correlation
#: would walk away from the very optimum it was sent to find.
#: ``'signed'`` maximises ``r`` itself and is what an alignment problem wants,
#: where the two series are the same quantity and the answer is which shift
#: makes them agree.
OBJECTIVES = ('absolute', 'signed')

#: Name of the band holding the series as recorded.
BAND_LEVEL = 'level'

#: Name of the band holding the series with its low-frequency component
#: removed, which for a 24-hour window leaves the daily cycle and the noise.
BAND_DIURNAL = 'diurnal'


def thermal_lag_filter(series, tau_hours, dt_hours=1.0):
    """
    A driver as the mass behind it would feel it: a first-order thermal lag.

    Masonry does not follow the air around it, it integrates it. A body of
    thermal time constant ``tau`` exposed to a forcing reaches a temperature
    that is a smoothed and attenuated version of that forcing, not a shifted
    copy of it, and the discrete form of that lumped-capacitance response is a
    one-pole exponential filter with ``alpha = dt / (tau + dt)``. A time
    constant of zero returns the driver untouched, which makes the
    instantaneous assumption the zero-inertia member of this family rather than
    a separate model.

    Parameters
    ----------
    series : pd.Series
        Driver on a regular ``DatetimeIndex``.
    tau_hours : float
        Thermal time constant in hours. ``0`` disables the filter.
    dt_hours : float, optional
        Sampling interval in hours. Default 1.0.

    Returns
    -------
    pd.Series
        The filtered driver, ``NaN`` wherever the input was ``NaN``.

    Notes
    -----
    Gaps are bridged before the filter runs and masked out again afterwards, so
    that a missing hour neither resets the filter state nor leaves a fill in the
    output. What the bridge cannot do is know what happened during a long
    outage: after a gap of many time constants the state carries an
    interpolation rather than a measurement, and it decays back towards the
    truth over a few multiples of ``tau``. On this record that matters for the
    hours immediately following a multi-day outage, and the study reports the
    paired hours every result rests on so that such a stretch is visible.
    """
    if tau_hours <= 0:
        return series.copy()
    alpha = dt_hours / (tau_hours + dt_hours)
    bridged = series.interpolate(method='time', limit_direction='both')
    return bridged.ewm(alpha=alpha, adjust=False).mean().where(series.notna())


def thermal_operator(series, delay=0, tau=0.0, dt_hours=1.0):
    """
    A driver put through the two things that can delay a response, in order.

    They are not interchangeable and this module scans them jointly for that
    reason. A **transport delay** shifts the driver in time without changing
    its shape: it is the time a thermal front takes to travel from the exposed
    face to the depth that governs the movement. A **thermal inertia**
    low-passes it: the mass integrates the forcing with a time constant. A
    filter of the right time constant and no delay, and a delay with no filter,
    can produce very similar peak correlations while implying entirely
    different physics — which is exactly why a scan over delay alone cannot
    tell the study which of them it is looking at.

    The inertia is applied first and the delay second, following the physics:
    the mass integrates the forcing, and the resulting thermal state then takes
    time to reach the depth that moves the instrument. That order is for the
    reader rather than for the arithmetic — a shift and a one-pole filter are
    both linear and time-invariant, so they commute, and only the treatment of
    the series' two ends distinguishes the orders. What matters is that both
    are applied and that the pair is reported together, since it is the pair,
    not either member, that the record identifies.

    Parameters
    ----------
    series : pd.Series
        Driver on a regular index.
    delay : int, optional
        Transport delay in samples. Default 0.
    tau : float, optional
        Thermal time constant in hours. Default 0.0, no inertia.
    dt_hours : float, optional
        Sampling interval in hours. Default 1.0.

    Returns
    -------
    pd.Series
        The transformed driver, on the index it arrived with.
    """
    out = thermal_lag_filter(series, tau, dt_hours=dt_hours)
    return out.shift(int(delay)) if delay else out


def operator_scan(response, driver, delays, taus=(0.0,), min_paired=24,
                  dt_hours=1.0):
    """
    Correlation over a grid of transport delays and thermal time constants.

    The two-dimensional counterpart of :func:`lag_scan`, and the form this
    project's couplings are actually measured in. The whole grid is returned
    rather than its maximum, because the maximum of such a grid is usually not
    a point but a ridge running diagonally from short delay with no inertia
    towards longer delay with more of it: the two parameters trade off against
    each other, and a single winning cell reported alone claims a resolution
    the record does not have.

    Parameters
    ----------
    response : pd.Series
        The responding series.
    driver : pd.Series
        The candidate driver, on the same index.
    delays : sequence of int
        Transport delays to evaluate, in samples.
    taus : sequence of float, optional
        Thermal time constants to evaluate, in hours. Default ``(0.0,)``, which
        reduces the scan to :func:`lag_scan` exactly. Include ``0`` in any grid
        so that the instantaneous case stays in the comparison.
    min_paired : int, optional
        Paired samples a cell needs before it is scored. Default 24.
    dt_hours : float, optional
        Sampling interval in hours. Default 1.0.

    Returns
    -------
    pd.DataFrame
        One row per cell: ``delay``, ``tau``, ``r``, ``r2`` and ``n``.
    """
    response = pd.to_numeric(response, errors='coerce')
    driver = pd.to_numeric(driver, errors='coerce')

    response_values = response.to_numpy(dtype=float)

    rows = []
    for tau in taus:
        # The filter runs once per time constant and the delays are then shifts
        # of its output, which is what makes a grid of this size affordable.
        filtered = thermal_lag_filter(driver, tau, dt_hours=dt_hours)
        for delay in delays:
            shifted = filtered.shift(int(delay)).to_numpy(dtype=float)
            r, count = _paired_correlation(shifted, response_values)
            if count < min_paired:
                r = np.nan
            rows.append({'delay': int(delay), 'tau': float(tau), 'r': r,
                         'r2': r ** 2 if not pd.isna(r) else np.nan,
                         'n': count})
    return pd.DataFrame(rows, columns=['delay', 'tau', 'r', 'r2', 'n'])


def best_operator(scan, objective='absolute', causal_only=False):
    """
    The winning cell of a grid, and what the instantaneous case would have got.

    Parameters
    ----------
    scan : pd.DataFrame
        A grid as :func:`operator_scan` returns it.
    objective : {'absolute', 'signed'}, optional
        What the search maximises. Default ``'absolute'``. See
        :data:`OBJECTIVES`.
    causal_only : bool, optional
        Restrict the search to non-negative delays. Default ``False``. The
        restriction is what a forecasting system would be held to: a negative
        delay needs the driver's future values at the moment the forecast is
        issued.

    Returns
    -------
    dict
        ``delay``, ``tau``, ``r``, ``r2`` and ``n`` of the winner, plus
        ``r2_instantaneous`` — the grid's own ``delay = 0, tau = 0`` cell — and
        ``gain_from_operator``, the difference between them. That difference is
        the number that says whether the operator earned its place: a driver
        whose best cell barely beats the instantaneous one is coupled now, not
        after a delay, whatever the winning cell claims. Ties go to the
        smallest delay and then the smallest time constant.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f'objective must be one of {OBJECTIVES}, got {objective!r}')

    flat = scan[(scan['delay'] == 0) & (scan['tau'] == 0.0)]
    r2_instantaneous = (float(flat['r2'].iloc[0])
                        if len(flat) and not pd.isna(flat['r2'].iloc[0])
                        else np.nan)

    scored = scan.dropna(subset=['r'])
    if causal_only:
        scored = scored[scored['delay'] >= 0]
    if scored.empty:
        return {'delay': np.nan, 'tau': np.nan, 'r': np.nan, 'r2': np.nan,
                'n': 0, 'r2_instantaneous': r2_instantaneous,
                'gain_from_operator': np.nan}

    scored = scored.sort_values(['delay', 'tau'])
    strength = scored['r'].abs() if objective == 'absolute' else scored['r']
    winner = scored.loc[strength.idxmax()]
    return {'delay': float(winner['delay']), 'tau': float(winner['tau']),
            'r': float(winner['r']), 'r2': float(winner['r2']),
            'n': int(winner['n']), 'r2_instantaneous': r2_instantaneous,
            'gain_from_operator': float(winner['r2']) - r2_instantaneous}


def lag_ranges(drivers, max_forward, signed_drivers=(), max_signed=None):
    """
    The lags each driver may be scanned over, from what kind of thing it is.

    An external forcing is scanned forwards only: it must precede the response
    it causes, so a negative optimum against one is not a measurement of
    anything physical. An internal state variable is scanned both ways, because
    nothing requires it to precede the deformation — at Gubbio a wall
    temperature probe sitting shallow means the inclination lags it, and one
    sitting deep means the inclination leads it, and which holds is unknown.

    Parameters
    ----------
    drivers : sequence of str
        Driver columns.
    max_forward : int
        Largest forward lag admitted for an external forcing, in samples.
    signed_drivers : sequence of str, optional
        Drivers to scan over signed delays. Default empty.
    max_signed : int or None, optional
        Largest delay, either way, for a signed driver. Default ``None``,
        which uses `max_forward`.

    Returns
    -------
    dict of str to range
        Lags per driver, ready for :func:`couple`.
    """
    if max_signed is None:
        max_signed = max_forward
    return {driver: (range(-max_signed, max_signed + 1)
                     if driver in signed_drivers
                     else range(0, max_forward + 1))
            for driver in drivers}


def strata(index, season_months=None, eras=None):
    """
    The slices a screen is run over, as boolean masks on one index.

    Three kinds, and the study says which of them it wants. The whole record,
    always, because a pooled result is what a later model would inherit. The
    instrument eras, because the archive spans two units of hardware and a
    coupling that appears in only one of them is a statement about an
    instrument rather than about a wall. And the seasons, because a driver
    whose range collapses outside one of them can produce a pooled gain that no
    single season supports.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Index the masks are built on.
    season_months : dict of str to sequence of int or None, optional
        Months belonging to each season, as :func:`shmlib.site.season_of`
        takes them. Default ``None``, no seasonal strata.
    eras : dict of str to tuple or None, optional
        Era label to ``(start, end)``, either bound may be ``None`` for open.
        Default ``None``, which uses the two eras of
        :data:`shmlib.site.ARCHIVE_START`, :data:`shmlib.site.LEGACY_END` and
        :data:`shmlib.site.CURRENT_START`.

    Returns
    -------
    dict of str to pd.Series
        ``'all'`` first, then one mask per era, then one per season.
    """
    if eras is None:
        eras = {'legacy': (site.ARCHIVE_START, site.LEGACY_END),
                'current': (site.CURRENT_START, None)}

    masks = {'all': pd.Series(True, index=index)}
    for label, (start, end) in eras.items():
        mask = pd.Series(True, index=index)
        if start is not None:
            mask &= index >= pd.Timestamp(start)
        if end is not None:
            # The era boundaries are dates, and the last day of an era belongs
            # to it in full rather than up to its first instant.
            mask &= index < pd.Timestamp(end) + pd.Timedelta(days=1)
        masks[label] = mask
    if season_months:
        labels = site.season_of(index, season_months)
        for season in season_months:
            masks[season] = labels == season
    return masks


def r2_ceiling(scans, panels, step=0.05):
    """
    A colour range that covers every panel a set of figures will draw.

    Several grids drawn on one scale can be read against each other; drawn on
    their own scales they cannot, because the same shade means a different
    number in each. The scale has to come from the data rather than from a
    guess, though, or it clips the strongest panel and hides exactly the
    comparison it was introduced to make. This returns the largest ``R^2``
    among the panels named, rounded up to a round number.

    Parameters
    ----------
    scans : dict
        The grids :func:`couple` returns, keyed by ``(stratum, band, driver)``.
    panels : sequence of tuple
        The panels to cover, as ``(stratum, band, drivers)``, matching what
        each figure will be asked to draw.
    step : float, optional
        Granularity the ceiling is rounded up to. Default 0.05.

    Returns
    -------
    float
        The ceiling, at most 1.0. Returns `step` when no panel carries a
        scored cell.
    """
    largest = 0.0
    for stratum, band, drivers in panels:
        for driver in drivers:
            scan = scans.get((stratum, band, driver))
            if scan is None or scan['r2'].isna().all():
                continue
            largest = max(largest, float(scan['r2'].max()))
    if largest <= 0.0:
        return float(step)
    return float(min(1.0, np.ceil(largest / step) * step))


def driver_coverage(frame, response, drivers):
    """
    What each driver has to work with, before any statistic is computed.

    A coupling table gives every driver a lag and a gain, including the ones
    whose overlap with the response is a fortnight in one summer. This says how
    much record each result rests on, and it belongs in front of the tables
    rather than in a footnote after them.

    Parameters
    ----------
    frame : pd.DataFrame
        Response and drivers on one grid.
    response : str
        Column carrying the responding series.
    drivers : sequence of str
        Driver columns to describe.

    Returns
    -------
    pd.DataFrame
        One row per driver: ``driver``, ``hours`` it carries, ``paired_hours``
        it shares with the response, and the ``first`` and ``last`` timestamp
        of that overlap.
    """
    observed = frame[response].notna()
    rows = []
    for driver in drivers:
        present = frame[driver].notna()
        paired = frame.index[present & observed]
        rows.append({
            'driver': driver,
            'hours': int(present.sum()),
            'paired_hours': int(len(paired)),
            'first': paired.min() if len(paired) else pd.NaT,
            'last': paired.max() if len(paired) else pd.NaT,
        })
    return pd.DataFrame(rows, columns=['driver', 'hours', 'paired_hours',
                                       'first', 'last'])


def diurnal_band(series, window=24, min_periods=None):
    """
    The series with its low-frequency component removed.

    Computed as the series minus its centred rolling mean, which for a window
    of one day removes everything slower than a day and keeps the daily cycle
    and whatever rides on it. It is a crude high-pass filter and is chosen for
    exactly that reason: its definition is one line of arithmetic that a reader
    can reproduce, where a designed filter would put an unstated phase response
    between the data and every lag this module goes on to measure.

    Separating the two bands matters because they answer different questions.
    A correlation on the levels is dominated by whatever slow drift the two
    series share — season, trend, the arbitrary anchor of an inclinometer —
    while a correlation on the diurnal band is a statement about the daily
    forcing cycle alone. A pair can score very differently on the two, and both
    numbers belong in the report.

    Parameters
    ----------
    series : pd.Series
        Series to filter, indexed by a ``DatetimeIndex`` on a regular grid.
    window : int, optional
        Width of the centred rolling mean, in samples of the grid the series
        is on. Default 24, one day of an hourly grid.
    min_periods : int or None, optional
        Samples the window must hold before it produces a mean. Default
        ``None``, which requires three quarters of `window`: a window mostly
        made of gap would otherwise subtract a mean of the few hours that
        survived, which is not the local level of anything.

    Returns
    -------
    pd.Series
        The high-passed series, ``NaN`` wherever the rolling mean could not be
        formed or the series itself is missing.

    Notes
    -----
    An even window has no centre sample, so its mean sits half a step off the
    point it is subtracted from and a linear drift survives as a *constant*
    offset rather than as zero. The offset is deliberate rather than
    overlooked: it changes no correlation and no regression slope, which are
    what this band is built for, whereas widening the window to an odd number
    of samples to remove it would stop the window spanning exactly one day and
    would leak a few per cent of the daily cycle into the baseline it
    subtracts. Where the offset would matter — a plot of the band's absolute
    level — subtract the band's own mean over the window being drawn.
    """
    if min_periods is None:
        min_periods = int(np.ceil(0.75 * window))
    baseline = series.rolling(window, center=True, min_periods=min_periods).mean()
    return series - baseline


def _paired_correlation(x, y, mask=None):
    """
    Pearson correlation over the samples both arrays carry, with its count.

    Written on arrays rather than on Series because the scans call it tens of
    thousands of times: a grid of delays and time constants over several strata
    is a large number of small correlations, and the per-call cost of building
    a frame and dropping its missing rows dominates everything else.

    Parameters
    ----------
    x, y : np.ndarray
        Aligned arrays, ``NaN`` where a value is missing.
    mask : np.ndarray or None, optional
        Boolean selection applied on top of the two arrays' own validity.
        Default ``None``, everything.

    Returns
    -------
    r : float
        Correlation, ``NaN`` when fewer than two samples survive or either
        side has no variation.
    n : int
        Samples the correlation rests on.
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    if mask is not None:
        valid &= mask
    count = int(valid.sum())
    if count < 2:
        return np.nan, count

    xv = x[valid]
    yv = y[valid]
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    denominator = np.sqrt(float(xv @ xv) * float(yv @ yv))
    if denominator == 0.0:
        return np.nan, count
    return float((xv @ yv) / denominator), count


def lag_scan(response, driver, lags, min_paired=24):
    """
    Correlation between a response and a driver at every lag in a range.

    The whole curve is returned rather than only its peak, because the shape of
    a lag curve says things its maximum does not: a broad flat curve is a
    coupling whose delay the record cannot resolve, a curve with two comparable
    peaks 24 hours apart is the aliasing a bounded search exists to avoid, and
    a curve that rises monotonically to the edge of the range is a warning that
    the range was chosen too narrow.

    The pairing at lag ``L`` is ``driver.shift(L)`` against `response`, so a
    positive lag means the response follows the driver.

    Parameters
    ----------
    response : pd.Series
        The responding series, indexed by a ``DatetimeIndex``.
    driver : pd.Series
        The candidate driver, on the same index.
    lags : sequence of int
        Lags to evaluate, in samples of the shared grid. The caller chooses the
        range, and the choice is physical: a forcing must precede the response
        it causes, so its range starts at zero, while an internal state
        variable may legitimately follow and its range is signed.
    min_paired : int, optional
        Hours both series must carry at a lag before that lag is scored. Below
        it the lag is reported with a ``NaN`` correlation and its true count,
        rather than dropped, so the curve shows where the record ran out.
        Default 24.

    Returns
    -------
    pd.DataFrame
        One row per lag, ascending, with columns ``lag``, ``r`` and ``n``.
    """
    response = pd.to_numeric(response, errors='coerce')
    driver = pd.to_numeric(driver, errors='coerce')

    rows = []
    for lag in lags:
        paired = pd.concat([driver.shift(lag), response], axis=1).dropna()
        count = len(paired)
        if count < min_paired:
            rows.append({'lag': int(lag), 'r': np.nan, 'n': count})
            continue
        rows.append({'lag': int(lag),
                     'r': float(paired.iloc[:, 0].corr(paired.iloc[:, 1])),
                     'n': count})
    return pd.DataFrame(rows, columns=['lag', 'r', 'n'])


def best_lag(curve, objective='absolute'):
    """
    The winning lag of a scan, under the stated objective.

    Parameters
    ----------
    curve : pd.DataFrame
        A scan as :func:`lag_scan` returns it.
    objective : {'absolute', 'signed'}, optional
        What the search maximises. ``'absolute'`` (the default) takes the
        largest ``|r|``, which is what a coupling of expected but unproven sign
        needs; ``'signed'`` takes the largest ``r``, which is what an alignment
        between two records of the same quantity needs. See :data:`OBJECTIVES`.

    Returns
    -------
    dict
        ``lag``, ``r`` and ``n`` of the winning row. All three are ``NaN``
        (``n`` is 0) when no lag in the curve was scored. Ties go to the
        smallest lag, so a curve with a plateau reports the earliest delay
        consistent with it rather than an arbitrary point inside it.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f'objective must be one of {OBJECTIVES}, got {objective!r}')

    scored = curve.dropna(subset=['r'])
    if scored.empty:
        return {'lag': np.nan, 'r': np.nan, 'n': 0}

    scored = scored.sort_values('lag')
    strength = scored['r'].abs() if objective == 'absolute' else scored['r']
    winner = scored.loc[strength.idxmax()]
    return {'lag': float(winner['lag']), 'r': float(winner['r']),
            'n': int(winner['n'])}


def gain_at_lag(response, driver, lag, hac_maxlags=24, tau=0.0,
                dt_hours=1.0):
    """
    Least-squares gain of the response on the driver, at one lag.

    The slope answers the question a correlation cannot: how much of the
    response a unit of the driver buys, in the response's own units. Its
    standard error is Newey-West, computed with the truncation lag the caller
    passes. This is not a refinement — the residuals of an hourly geophysical
    regression are strongly autocorrelated, and the ordinary standard error of
    such a fit understates the true uncertainty badly enough to turn a
    coincidence into a significant coefficient.

    Parameters
    ----------
    response : pd.Series
        The responding series.
    driver : pd.Series
        The candidate driver, on the same index.
    lag : int or float
        Lag to evaluate, in samples of the shared grid, applied to the driver
        as in :func:`lag_scan`. A ``NaN`` lag returns a row of ``NaN``, which
        is what a scan that found no admissible optimum should propagate.
    hac_maxlags : int, optional
        Truncation lag of the Newey-West covariance estimator, in samples.
        Default 24, one day of an hourly grid: long enough to cover the
        autocorrelation a daily cycle imposes on the residuals.
    tau : float, optional
        Thermal time constant applied to the driver before the lag, in hours.
        Default 0.0, no inertia, which makes the gain the slope against the
        driver as recorded. Where a scan chose a time constant, the gain must
        be fitted against the same operator that scan selected, or it is the
        slope of a different variable.
    dt_hours : float, optional
        Sampling interval in hours. Default 1.0.

    Returns
    -------
    dict
        ``slope``, ``slope_se``, ``ci_low``, ``ci_high``, ``intercept``, ``r``
        and ``n``. The interval is the 95 % confidence interval of the slope
        under the HAC covariance. Every field is ``NaN`` when the lag is
        ``NaN`` or fewer than three paired samples survive.
    """
    empty = {'slope': np.nan, 'slope_se': np.nan, 'ci_low': np.nan,
             'ci_high': np.nan, 'intercept': np.nan, 'r': np.nan, 'n': 0}
    if lag is None or (isinstance(lag, float) and np.isnan(lag)):
        return empty

    # Imported here rather than at module level: statsmodels costs about a
    # second to import, `shmlib/__init__` imports every module, and a study
    # that never estimates a gain should not pay for one.
    import statsmodels.api as sm

    operated = thermal_operator(pd.to_numeric(driver, errors='coerce'),
                                delay=int(lag), tau=tau, dt_hours=dt_hours)
    paired = pd.concat([operated, pd.to_numeric(response, errors='coerce')],
                       axis=1).dropna()
    if len(paired) < 3:
        return dict(empty, n=len(paired))

    x = paired.iloc[:, 0]
    y = paired.iloc[:, 1]
    model = sm.OLS(y.to_numpy(dtype=float),
                   sm.add_constant(x.to_numpy(dtype=float))).fit(
        cov_type='HAC', cov_kwds={'maxlags': int(hac_maxlags)})
    low, high = model.conf_int()[1]
    return {'slope': float(model.params[1]),
            'slope_se': float(model.bse[1]),
            'ci_low': float(low),
            'ci_high': float(high),
            'intercept': float(model.params[0]),
            'r': float(x.corr(y)),
            'n': int(len(paired))}


def couple(frame, response, drivers, lags, taus=(0.0,), band_taus=None,
           bands=(BAND_LEVEL, BAND_DIURNAL), strata=None, diurnal_window=24,
           diurnal_min_periods=None, objective='absolute', min_paired=24,
           hac_maxlags=24, dt_hours=1.0):
    """
    Scan and fit every driver, in every band, over every stratum.

    One call produces the study's whole coupling table: for each combination it
    runs :func:`operator_scan` over the delays and time constants the caller
    admits, takes the winner with :func:`best_operator`, and estimates the gain
    at that same operator with :func:`gain_at_lag`. Both the response and the
    drivers are band-passed together, so a diurnal-band result compares like
    with like.

    Two columns of the result exist to keep the scan honest about itself.
    ``gain_from_operator`` is how much of the fit the operator bought over the
    instantaneous case, and a driver whose best cell barely beats that case is
    coupled now rather than after a delay. ``r2_lost_to_causality`` is what the
    fit gives up when the search is restricted to non-negative delays, which is
    the restriction any forecast is under; for a driver whose optimum is
    already causal it is zero, and for one whose optimum is a lead it prices
    the clamp that must be applied before the driver can feed a model.

    The band and the operator are built on the **whole record** and the stratum
    selects only which pairs are then scored. The alternative — cutting the
    stratum out first — cannot be used here: a season is not a contiguous
    stretch of time, so a filter run on one would carry its state across a
    nine-month jump from September to the following June, and a rolling mean
    over one would average across the same jump. Both would be filtering a
    series that never existed. The cost of this order is that a stratum's first
    hours carry filter state accumulated just outside it, which for the time
    constants scanned here decays within a few days of its boundary.

    Parameters
    ----------
    frame : pd.DataFrame
        Response and drivers on one regular grid, in one time zone.
    response : str
        Column carrying the responding series.
    drivers : sequence of str
        Columns to screen. Every one is scanned in every band and stratum,
        including any control channel — a control that were treated specially
        here could not be read on the same terms as the candidates.
    lags : dict of str to sequence of int
        Transport delays to scan per driver. A driver absent from the mapping
        is scanned over the range given for ``None``, if one is present, and
        skipped otherwise. The mapping is how a study states that its external
        forcings may only be scanned forwards while an internal state variable
        may be scanned both ways.
    taus : sequence of float or dict of str to sequence of float, optional
        Thermal time constants to scan, in hours, either one grid for every
        driver or one per driver as ``lags`` is given. Default ``(0.0,)``,
        which scans transport delay alone. A grid should include ``0`` so that
        the instantaneous case stays in the comparison.
    band_taus : dict of str to sequence of float or None, optional
        Time constants for a particular band, overriding `taus` there. Default
        ``None``.

        This exists for one reason, and a study screening a narrow band needs
        it. A one-pole filter applied to a signal of a single frequency is
        exactly an amplitude scaling and a phase shift, and a transport delay
        is also a phase shift, so on a band that contains essentially one
        period the two parameters are **not separately identifiable**: every
        combination lying on one line of constant total phase fits equally
        well, and a search over the pair is free to rotate the phase far enough
        to turn an anti-correlation into a correlation and report the flipped
        sign as the optimum. Pass ``{BAND_DIURNAL: (0.0,)}`` to keep such a
        band a delay-only screen, and scan the pair on the levels, where a
        driver has the broadband content that separates them.
    bands : sequence of str, optional
        Bands to run. Default both :data:`BAND_LEVEL` and
        :data:`BAND_DIURNAL`.
    strata : dict of str to pd.Series or None, optional
        Boolean masks over `frame`'s index, one per stratum, evaluated in the
        order given. Default ``None``, which runs a single stratum named
        ``'all'`` over the whole frame.
    diurnal_window, diurnal_min_periods : optional
        Passed to :func:`diurnal_band` when the diurnal band is built.
    objective : {'absolute', 'signed'}, optional
        Passed to :func:`best_lag`. Default ``'absolute'``.
    min_paired : int, optional
        Passed to :func:`lag_scan`. Default 24.
    hac_maxlags : int, optional
        Passed to :func:`gain_at_lag`. Default 24.
    dt_hours : float, optional
        Sampling interval in hours. Default 1.0.

    Returns
    -------
    table : pd.DataFrame
        One row per stratum, band and driver: ``stratum``, ``band``,
        ``driver``, the winning ``lag`` and ``tau``, ``r``, ``r2``, ``n``,
        ``slope``, ``slope_se``, ``ci_low``, ``ci_high``, ``intercept``,
        ``sign``, and the four columns that judge the operator itself —
        ``r2_instantaneous``, ``gain_from_operator``, ``causal_r2``,
        ``r2_lost_to_causality`` and ``tau_at_edge``. The last is ``True``
        where the winning time constant is the largest one scanned, which means
        the grid stopped before the optimum did: such a row reports where the
        search ran out, not where the physics settled, and must not be quoted
        as a measured time constant. ``sign`` is ``'-'``, ``'+'`` or ``''``
        according to the slope. The intercept is carried for the figures that
        draw the fitted line; on a series whose level rests on an arbitrary
        anchor it means nothing on its own.
    scans : dict
        The full grid behind every row, keyed by ``(stratum, band, driver)``,
        for the figures that draw the scan rather than its maximum.
    """
    if strata is None:
        strata = {'all': pd.Series(True, index=frame.index)}

    masks = {stratum: mask.reindex(frame.index, fill_value=False)
             .to_numpy(dtype=bool)
             for stratum, mask in strata.items()}

    rows, scans = [], {}
    for band in bands:
        if band == BAND_LEVEL:
            banded = frame
        elif band == BAND_DIURNAL:
            banded = frame.apply(diurnal_band, window=diurnal_window,
                                 min_periods=diurnal_min_periods)
        else:
            raise ValueError(f'unknown band {band!r}')

        response_values = pd.to_numeric(banded[response],
                                        errors='coerce').to_numpy(dtype=float)

        for driver in drivers:
            driver_lags = lags.get(driver, lags.get(None))
            if driver_lags is None or driver not in banded.columns:
                continue
            driver_taus = (taus.get(driver, taus.get(None, (0.0,)))
                           if isinstance(taus, dict) else taus)
            if band_taus is not None and band in band_taus:
                driver_taus = band_taus[band]

            cells = {stratum: [] for stratum in strata}
            for tau in driver_taus:
                # One filter run per time constant, shared by every delay and
                # every stratum: the filter is the expensive part of the grid.
                filtered = thermal_lag_filter(
                    pd.to_numeric(banded[driver], errors='coerce'), tau,
                    dt_hours=dt_hours)
                for delay in driver_lags:
                    shifted = filtered.shift(int(delay)).to_numpy(dtype=float)
                    for stratum, mask in masks.items():
                        r, count = _paired_correlation(shifted,
                                                       response_values, mask)
                        if count < min_paired:
                            r = np.nan
                        cells[stratum].append(
                            {'delay': int(delay), 'tau': float(tau), 'r': r,
                             'r2': r ** 2 if not pd.isna(r) else np.nan,
                             'n': count})

            for stratum in strata:
                scan = pd.DataFrame(cells[stratum],
                                    columns=['delay', 'tau', 'r', 'r2', 'n'])
                winner = best_operator(scan, objective=objective)
                causal = best_operator(scan, objective=objective,
                                       causal_only=True)
                block = banded.loc[strata[stratum]
                                   .reindex(banded.index, fill_value=False)]
                gain = gain_at_lag(block[response], block[driver],
                                   winner['delay'], hac_maxlags=hac_maxlags,
                                   tau=winner['tau'] if not pd.isna(winner['tau'])
                                   else 0.0, dt_hours=dt_hours)
                scans[(stratum, band, driver)] = scan
                largest_tau = max(driver_taus) if len(driver_taus) else 0.0
                rows.append({
                    'stratum': stratum, 'band': band, 'driver': driver,
                    'lag': winner['delay'], 'tau': winner['tau'],
                    'r': winner['r'], 'r2': winner['r2'], 'n': winner['n'],
                    'slope': gain['slope'], 'slope_se': gain['slope_se'],
                    'ci_low': gain['ci_low'], 'ci_high': gain['ci_high'],
                    'intercept': gain['intercept'],
                    'r2_instantaneous': winner['r2_instantaneous'],
                    'gain_from_operator': winner['gain_from_operator'],
                    'causal_r2': causal['r2'],
                    'r2_lost_to_causality': winner['r2'] - causal['r2'],
                    'tau_at_edge': bool(winner['tau'] == largest_tau
                                        and largest_tau > 0.0),
                    'sign': _sign_of(gain['slope']),
                })
    table = pd.DataFrame(rows, columns=['stratum', 'band', 'driver', 'lag',
                                        'tau', 'r', 'r2', 'n', 'slope',
                                        'slope_se', 'ci_low', 'ci_high',
                                        'intercept', 'r2_instantaneous',
                                        'gain_from_operator', 'causal_r2',
                                        'r2_lost_to_causality', 'tau_at_edge',
                                        'sign'])
    return table, scans


def _sign_of(value):
    """
    ``'-'``, ``'+'`` or ``''`` for a slope, for a table cell rather than a test.
    """
    if value is None or np.isnan(value):
        return ''
    return '-' if value < 0 else '+'


def gain_stability(frame, response, drivers, lags, taus=None, freq='MS',
                   min_paired=24, hac_maxlags=24, dt_hours=1.0):
    """
    The gain of each driver re-estimated window by window.

    A gain fitted over a whole record is a single number that a single season
    can produce on its own: a driver active only in summer, or a coincidence
    between two slow drifts, both give a respectable pooled slope. Re-fitting
    the same lag over successive windows separates the two — a real coupling
    holds its slope, within its uncertainty, across windows in which the
    driver's own range changes.

    The operator is held fixed at what the pooled scan chose — both the delay
    and the time constant — and is applied once over the whole record before
    the windows are cut, so a pair straddling a window boundary is kept rather
    than lost to the cut, and the filter is not restarted twelve times a year.

    Parameters
    ----------
    frame : pd.DataFrame
        Response and drivers on one regular grid, already in the band the
        caller wants the gains measured in.
    response : str
        Column carrying the responding series.
    drivers : sequence of str
        Columns to re-fit.
    lags : dict of str to int or float
        Transport delay to hold each driver at, as chosen by the pooled scan. A
        driver whose delay is ``NaN`` or absent is skipped: there is no optimum
        to follow over time.
    taus : dict of str to float or None, optional
        Thermal time constant to hold each driver at. Default ``None``, no
        inertia for any of them. Where a scan chose a time constant, passing it
        here is not optional: a gain re-fitted against a different operator
        from the one that selected it is the slope of a different variable.
    freq : str, optional
        Window the gain is tracked over, as a pandas offset alias. Default
        ``'MS'``, calendar months.
    min_paired : int, optional
        Paired hours a window needs before its gain is reported. Default 24.
    hac_maxlags : int, optional
        Truncation lag of the Newey-West covariance, in samples. Default 24.
    dt_hours : float, optional
        Sampling interval in hours. Default 1.0.

    Returns
    -------
    pd.DataFrame
        One row per window and driver: ``window``, ``driver``, ``lag``,
        ``tau``, ``slope``, ``slope_se``, ``ci_low``, ``ci_high``, ``r`` and
        ``n``.
    """
    rows = []
    for driver in drivers:
        lag = lags.get(driver)
        if lag is None or (isinstance(lag, float) and np.isnan(lag)):
            continue
        if driver not in frame.columns:
            continue

        tau = 0.0 if taus is None else float(taus.get(driver, 0.0))
        operated = thermal_operator(pd.to_numeric(frame[driver], errors='coerce'),
                                    delay=int(lag), tau=tau, dt_hours=dt_hours)
        paired = pd.concat([operated.rename('driver'),
                            pd.to_numeric(frame[response], errors='coerce')
                            .rename('response')], axis=1).dropna()

        for window, block in paired.groupby(pd.Grouper(freq=freq)):
            if len(block) < min_paired:
                continue
            gain = gain_at_lag(block['response'], block['driver'], 0,
                               hac_maxlags=hac_maxlags)
            rows.append({'window': window, 'driver': driver, 'lag': float(lag),
                         'tau': tau,
                         'slope': gain['slope'], 'slope_se': gain['slope_se'],
                         'ci_low': gain['ci_low'], 'ci_high': gain['ci_high'],
                         'r': gain['r'], 'n': gain['n']})
    return pd.DataFrame(rows, columns=['window', 'driver', 'lag', 'tau',
                                       'slope', 'slope_se', 'ci_low',
                                       'ci_high', 'r', 'n'])


def _annual_design(index, order, period_days=365.25):
    doy = pd.DatetimeIndex(index).dayofyear.to_numpy(dtype=float)
    columns = [np.ones_like(doy)]
    for k in range(1, int(order) + 1):
        angle = 2.0 * np.pi * k * doy / period_days
        columns += [np.cos(angle), np.sin(angle)]
    return np.column_stack(columns)


def annual_modulation(daily_series, harmonics=(1, 2), holdout='year',
                      min_gain=0.01):
    """
    How a daily statistic moves through the year, as a low-order Fourier fit.

    Fits ``a0 + sum_k (a_k cos + b_k sin)(2 pi k doy / 365.25)`` to a daily
    series for each candidate order and scores each by leave-one-year-out
    mean squared error. The order is chosen by a parsimony rule rather than
    by the bare minimum of that score: candidates are tried in ascending
    order, the first with a defined hold-out score is the current choice,
    and a higher order replaces it only when its hold-out MSE improves on
    the current choice's by at least the fraction ``min_gain``
    (``mse < best_mse * (1 - min_gain)``). On a series whose true structure
    is a single harmonic, the hold-out MSE of a higher order differs from
    the lower order's only by noise, and choosing whichever of the two
    happens to score marginally lower would let that noise decide the
    order; requiring a minimum relative gain keeps the simpler order unless
    the data give a real reason to prefer the richer one. An order whose
    hold-out MSE is ``NaN`` (too few years to hold one out) is never
    chosen. The chosen fit is what the smooth conditional daily term
    consumes as its weight curve (spec D6, D7).

    Parameters
    ----------
    daily_series : pd.Series
        One value per day, indexed by day.
    harmonics : sequence of int, optional
        Candidate Fourier orders, tried in ascending order. Default
        ``(1, 2)``.
    holdout : {'year'}, optional
        Hold-out unit for the order choice. Default ``'year'``.
    min_gain : float, optional
        Minimum relative improvement in hold-out MSE a higher order must
        show over the current choice before it replaces it. Default
        ``0.01``.

    Returns
    -------
    (pd.DataFrame, dict)
        ``table`` with ``order``, ``holdout_mse``, ``chosen``; ``fit`` with
        ``order``, ``coef``, ``period_days`` and ``n`` for the chosen order.
    """
    values = pd.to_numeric(daily_series, errors='coerce').dropna()
    index = pd.DatetimeIndex(values.index)
    years = index.year
    ordered_harmonics = sorted(int(order) for order in harmonics)
    rows, coefs = [], {}
    for order in ordered_harmonics:
        errors = []
        for held in np.unique(years):
            train, test = years != held, years == held
            if train.sum() < 3 * (2 * order + 1) or test.sum() == 0:
                continue
            coef, _, _, _ = np.linalg.lstsq(
                _annual_design(index[train], order), values.to_numpy()[train],
                rcond=None)
            predicted = _annual_design(index[test], order) @ coef
            errors.append(float(((values.to_numpy()[test] - predicted) ** 2).mean()))
        coef, _, _, _ = np.linalg.lstsq(_annual_design(index, order),
                                        values.to_numpy(), rcond=None)
        coefs[order] = coef
        rows.append({'order': order,
                     'holdout_mse': float(np.mean(errors)) if errors else np.nan})

    best_order, best_mse = None, None
    for row in rows:
        order, mse = row['order'], row['holdout_mse']
        if pd.isna(mse):
            continue
        if best_order is None or mse < best_mse * (1.0 - min_gain):
            best_order, best_mse = order, mse
    if best_order is None:
        raise ValueError(
            'annual_modulation: every candidate order has too few years to '
            'hold one out; pass fewer years worth of harmonics or a longer '
            'daily_series.')

    table = pd.DataFrame(rows)
    table['chosen'] = table['order'] == best_order
    fit = {'order': best_order, 'coef': coefs[best_order],
           'period_days': 365.25, 'n': int(values.size)}
    return table, fit


def evaluate_modulation(fit, index):
    """
    The fitted annual modulation at each timestamp's day of year.

    Parameters
    ----------
    fit : dict
        Second return value of :func:`annual_modulation`.
    index : pd.DatetimeIndex

    Returns
    -------
    pd.Series
        Indexed by ``index``.
    """
    design = _annual_design(index, fit['order'], fit['period_days'])
    return pd.Series(design @ fit['coef'], index=pd.DatetimeIndex(index))


def shortlist(table, control, expected_sign='-', margin=0.0):
    """
    Which drivers clear the control, and which of those move the right way.

    A screen over a dozen candidate drivers finds an optimum for every one of
    them, because a bounded search over a near-periodic pair always can. What
    separates a coupling from that background is not the size of ``|r|`` on its
    own but its size *relative to a channel with no mechanism* — a supply
    voltage cannot move a wall, so whatever it scores is the study's own noise
    floor and every candidate is read against it.

    Two further conditions are applied and reported separately, because a
    driver can fail one and pass the other and the report should say which:
    the gain's confidence interval must exclude zero, and its sign must be the
    one the site's geometry predicts.

    Parameters
    ----------
    table : pd.DataFrame
        A coupling table as :func:`couple` returns it.
    control : str
        Driver acting as the noise floor. Compared within each stratum and
        band, never pooled across them, since the floor of a winter month and
        of an eight-year record are different numbers.
    expected_sign : {'-', '+'}, optional
        Sign the site's convention predicts. Default ``'-'``: at Gubbio the
        valley face is the more exposed, so daytime heating tips the wall
        towards the mountain, which is a negative change in inclination.
        Recorded in ``docs/raw-data-format.md`` section 7.5, along with the
        raw channel's contradicting positive slope.
    margin : float, optional
        How far above the control's ``|r|`` a candidate must reach before it
        counts as clearing it. Default 0.0, which asks only that it be larger.

    Returns
    -------
    pd.DataFrame
        `table` with the control's rows removed and four columns added:
        ``control_r`` (the floor it was read against), ``clears_control``,
        ``gain_significant`` (the confidence interval excludes zero) and
        ``expected_sign`` (the slope points the way the site predicts).
    """
    floors = (table[table['driver'] == control]
              .set_index(['stratum', 'band'])['r'].abs())

    out = table[table['driver'] != control].copy()
    keys = pd.MultiIndex.from_frame(out[['stratum', 'band']])
    out['control_r'] = floors.reindex(keys).to_numpy()
    out['clears_control'] = out['r'].abs() > (out['control_r'] + margin)
    out['gain_significant'] = (out['ci_low'] * out['ci_high']) > 0
    out['expected_sign'] = out['sign'] == expected_sign
    return out
