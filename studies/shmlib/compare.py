"""
Module: shmlib.compare

How a source behaves on its own, and how two sources compare.

This module answers two related questions that recur whenever more than one
record of the same physical quantity exists: what does its mean daily cycle
look like over a season (:func:`diurnal_profile`, :func:`diurnal_table`), and
how far does it stand from a reference — in level, in common variation, in
amplitude and in phase (:func:`pairwise_agreement`), whether that distance
holds still over time (:func:`agreement_stability`), and what a documented
linear substitution against a trustworthy reference window would cost
(:func:`calibrate_against_sensor`).

Moved here from study 2's now-deleted private library, where these functions were written to
compare the on-structure record against the two external proxy sources. None of
them encodes a choice specific to that comparison — a season definition, a
reference source, a minimum sample count, all arrive as arguments — so a later
study comparing any two series of its own finds the same measures here rather
than writing them a second time.

Every pair uses its own overlap. Nothing here interpolates one series to
manufacture agreement with another.
"""

import numpy as np
import pandas as pd

from . import meteo, proxies, quality, site, solar


def diurnal_profile(df, columns, season, season_months, freq=site.ANALYSIS_FREQ,
                    circular=proxies.CIRCULAR, min_days=1, complete_day=None):
    """
    Mean daily cycle of several channels over one season.

    Each hour of the profile is reported with the number of days that
    contributed to it, so that a poorly covered hour is not presented as equally
    certain to a well covered one. That count is the whole reason this returns a
    frame rather than a Series.

    Days are not required to be complete by default. Requiring completeness is
    the right choice when the object of interest is the shape of one instrument's
    day, as it is in study 1; here the object is the climatology of a source, and
    a source that loses a scattered handful of hours across eight years would
    lose most of its record to a completeness rule that gains it nothing.
    ``complete_day`` is provided for the cases where the stricter rule is wanted.

    Parameters
    ----------
    df : pd.DataFrame
        Frame on the analysis grid.
    columns : sequence of str
        Channels to profile.
    season : str
        Season to restrict to, as named in ``season_months``.
    season_months : dict of str to sequence of int
        Months belonging to each season.
    freq : str, optional
        Grid of ``df``, used to derive the positions within the day. Default
        :data:`shmlib.site.ANALYSIS_FREQ`.
    circular : sequence of str, optional
        Quantities averaged by the unit-vector method. Default
        :data:`shmlib.proxies.CIRCULAR`.
    min_days : int, optional
        Days an hour must draw on before its mean is reported. Default ``1``.
    complete_day : int or None, optional
        Slots a day must carry to contribute at all. Default ``None``, which
        admits every day.

    Returns
    -------
    pd.DataFrame
        Indexed by position within the day in fractional hours. Two columns per
        channel: the mean, and ``{channel}_n``, the days behind it.
    """
    labels = site.season_of(df.index, season_months)
    data = df.loc[labels.values == season, list(columns)]
    if not len(data):
        return pd.DataFrame()

    position = site.time_of_day(data.index)
    day = pd.DatetimeIndex(data.index).normalize()

    out = pd.DataFrame(index=pd.Index(sorted(set(position)), name='hour'))
    for column in columns:
        values = data[column]
        if complete_day is not None:
            counts = values.groupby(day).count()
            keep = counts[counts >= complete_day].index
            values = values.where(np.isin(day, keep))

        frame = pd.DataFrame({'position': position, 'day': day,
                              'value': values.to_numpy()})
        frame = frame.dropna(subset=['value'])
        if not len(frame):
            out[column] = np.nan
            out[f'{column}_n'] = 0
            continue

        quantity = column.rpartition('_')[0]
        if quantity in circular:
            means = frame.groupby('position')['value'].apply(
                lambda group: meteo.circular_mean(group.to_numpy()))
        else:
            means = frame.groupby('position')['value'].mean()
        days = frame.groupby('position')['day'].nunique()

        means = means.where(days >= min_days)
        out[column] = means.reindex(out.index)
        out[f'{column}_n'] = days.reindex(out.index).fillna(0).astype(int)

    out.attrs['season'] = season
    out.attrs['freq'] = freq
    return out


def diurnal_table(df, columns, seasons, season_months, **kwargs):
    """
    Stack :func:`diurnal_profile` over several seasons into one long table.

    The tidy form the CSV behind the diurnal figures is written from: one row
    per season, per channel, per position in the day.

    Parameters
    ----------
    df : pd.DataFrame
        Frame on the analysis grid.
    columns : sequence of str
        Channels to profile.
    seasons : sequence of str
        Seasons to compute, in the order they should appear.
    season_months : dict of str to sequence of int
        Months belonging to each season.
    **kwargs
        Passed through to :func:`diurnal_profile`.

    Returns
    -------
    pd.DataFrame
        Columns ``season``, ``channel``, ``quantity``, ``hour``, ``mean``,
        ``n_days``.
    """
    rows = []
    for season in seasons:
        profile = diurnal_profile(df, columns, season, season_months, **kwargs)
        if not len(profile):
            continue
        for column in columns:
            if column not in profile.columns:
                continue
            for hour, value in profile[column].items():
                rows.append({
                    'season': season,
                    'channel': column,
                    'quantity': column.rpartition('_')[0],
                    'hour': float(hour),
                    'mean': float(value) if pd.notna(value) else np.nan,
                    'n_days': int(profile.loc[hour, f'{column}_n']),
                })
    return pd.DataFrame(rows)


def _paired(df, quantity, reference, compared):
    """
    The timestamps on which two sources of one quantity are both present.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame.
    quantity : str
        Canonical quantity name.
    reference, compared : str
        Source suffixes.

    Returns
    -------
    pd.DataFrame
        Two columns, ``reference`` and ``compared``, over the shared timestamps.
        Empty when either channel is absent.
    """
    left, right = f'{quantity}_{reference}', f'{quantity}_{compared}'
    if left not in df.columns or right not in df.columns:
        return pd.DataFrame(columns=['reference', 'compared'])
    paired = pd.DataFrame({'reference': pd.to_numeric(df[left],
                                                      errors='coerce'),
                           'compared': pd.to_numeric(df[right],
                                                     errors='coerce')})
    return paired.dropna()


def _agreement_scores(paired, circular=False):
    """
    Bias, error, and correlation between two aligned series.

    Parameters
    ----------
    paired : pd.DataFrame
        Columns ``reference`` and ``compared``, already free of missing values.
    circular : bool, optional
        Take differences on the circle. Default ``False``.

    Returns
    -------
    dict
        ``n``, ``bias``, ``mae``, ``rmse``, ``r``. Every value is ``nan`` when
        fewer than two pairs are present.
    """
    if len(paired) < 2:
        return {'n': int(len(paired)), 'bias': np.nan, 'mae': np.nan,
                'rmse': np.nan, 'r': np.nan}
    if circular:
        difference = meteo.circular_difference(paired['compared'],
                                               paired['reference'])
    else:
        difference = paired['compared'] - paired['reference']
    return {
        'n': int(len(paired)),
        'bias': float(difference.mean()),
        'mae': float(difference.abs().mean()),
        'rmse': float(np.sqrt((difference ** 2).mean())),
        'r': float(paired['compared'].corr(paired['reference'])),
    }


def daily_amplitude_phase(series, min_hours=20, circular=False):
    """
    Median daily amplitude and the median hour of the daily maximum.

    Amplitude and phase are what separate two series that a correlation says are
    the same. A grid-cell average and a sun-exposed housing can track each other
    almost perfectly and still differ by six degrees of daily swing, which is a
    difference that matters to a wall.

    Both are medians over days rather than statistics of the pooled record, so a
    single unusual day cannot set the answer.

    Parameters
    ----------
    series : pd.Series
        Channel on the analysis grid.
    min_hours : int, optional
        Hours a day must carry to contribute. Default ``20``.
    circular : bool, optional
        Skip the amplitude, which is meaningless for a direction. Default
        ``False``.

    Returns
    -------
    dict
        ``n_days``, ``amplitude``, ``peak_hour``.
    """
    values = pd.to_numeric(series, errors='coerce').dropna()
    if not len(values):
        return {'n_days': 0, 'amplitude': np.nan, 'peak_hour': np.nan}

    day = pd.DatetimeIndex(values.index).normalize()
    counts = values.groupby(day).count()
    usable = counts[counts >= min_hours].index
    kept = values[np.isin(day, usable)]
    if not len(kept):
        return {'n_days': 0, 'amplitude': np.nan, 'peak_hour': np.nan}

    kept_day = pd.DatetimeIndex(kept.index).normalize()
    grouped = kept.groupby(kept_day)
    amplitude = np.nan if circular else float((grouped.max()
                                               - grouped.min()).median())
    peak_hours = grouped.apply(
        lambda group: site.time_of_day(pd.DatetimeIndex([group.idxmax()]))[0])
    peak = float(meteo.circular_mean(peak_hours.to_numpy() * 15.0) / 15.0) \
        if len(peak_hours) else np.nan

    return {'n_days': int(len(usable)), 'amplitude': amplitude,
            'peak_hour': peak}


def pairwise_agreement(df, quantity, reference='str', compared=None,
                       season_months=None, by=None, circular=proxies.CIRCULAR,
                       min_hours=20):
    """
    How far each external source stands from the reference, for one quantity.

    Reported per pair rather than as a single matrix, and optionally split by
    season or by time of day, because the sources being compared are not
    necessarily replicates and the structure of their disagreement is the
    result. A single correlation coefficient over eight years would hide
    exactly what a study built on this function is for: a grid-cell average
    that agrees in winter and departs by five degrees on summer afternoons is
    not a source with a good correlation, it is a source with a known,
    seasonal, time-of-day-dependent bias.

    Every pair uses its own overlap. Nothing is interpolated to manufacture one.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame.
    quantity : str
        Canonical quantity name.
    reference : str, optional
        Source the others are measured against. Default ``'str'``, the
        on-structure measurement.
    compared : sequence of str or None, optional
        Sources to score. Default ``None``, which is every source in
        :data:`shmlib.proxies.SOURCES` other than ``reference``.
    season_months : dict or None, optional
        Months belonging to each season. Required when ``by='season'``.
    by : {None, 'season', 'hour'}, optional
        Split the comparison. Default ``None``, one row per pair over the whole
        overlap.
    circular : sequence of str, optional
        Quantities differenced on the circle. Default
        :data:`shmlib.proxies.CIRCULAR`.
    min_hours : int, optional
        Hours a day must carry to contribute to the amplitude and phase columns.
        Default ``20``.

    Returns
    -------
    pd.DataFrame
        One row per pair per split: overlap, bias, MAE, RMSE, correlation, the
        median daily amplitude of each side and their ratio, and the median peak
        hour of each side and the lag between them.
    """
    compared = ([source for source in proxies.SOURCES if source != reference]
                if compared is None else compared)
    is_circular = quantity in circular

    def _split_labels():
        if by is None:
            return pd.Series('all', index=df.index)
        if by == 'season':
            if season_months is None:
                raise ValueError("by='season' needs season_months")
            return site.season_of(df.index, season_months)
        if by == 'hour':
            return pd.Series(pd.DatetimeIndex(df.index).hour, index=df.index)
        raise ValueError(f'unknown split: {by!r}')

    labels = _split_labels()
    rows = []
    for source in compared:
        for label in [value for value in pd.unique(labels.dropna())]:
            window = df[labels.values == label]
            paired = _paired(window, quantity, reference, source)
            scores = _agreement_scores(paired, circular=is_circular)

            left = daily_amplitude_phase(window.get(f'{quantity}_{reference}',
                                                    pd.Series(dtype=float)),
                                         min_hours=min_hours,
                                         circular=is_circular)
            right = daily_amplitude_phase(window.get(f'{quantity}_{source}',
                                                     pd.Series(dtype=float)),
                                          min_hours=min_hours,
                                          circular=is_circular)
            phase_lag = (np.nan if (np.isnan(left['peak_hour'])
                                    or np.isnan(right['peak_hour']))
                         else float(meteo.circular_difference(
                             right['peak_hour'] * 15.0,
                             left['peak_hour'] * 15.0) / 15.0))
            rows.append({
                'quantity': quantity,
                'reference': reference,
                'compared': source,
                'split': label,
                **scores,
                'amplitude_reference': left['amplitude'],
                'amplitude_compared': right['amplitude'],
                'amplitude_ratio': (right['amplitude'] / left['amplitude']
                                    if left['amplitude'] not in (0, np.nan)
                                    and pd.notna(left['amplitude'])
                                    else np.nan),
                'peak_hour_reference': left['peak_hour'],
                'peak_hour_compared': right['peak_hour'],
                'phase_lag_h': phase_lag,
            })
    return pd.DataFrame(rows)


def agreement_stability(df, quantity, reference='str', compared=None,
                        freq='MS', min_hours=100, circular=proxies.CIRCULAR):
    """
    The bias between two sources, month by month.

    Whether a source may stand in for the local measurement turns on whether its
    bias is *stable*, not on whether it is small. A constant offset can be
    subtracted and the substitution documented; a bias that drifts with the
    season cannot, and the two are indistinguishable in any statistic pooled
    over the whole record.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame.
    quantity : str
        Canonical quantity name.
    reference : str, optional
        Source the others are measured against. Default ``'str'``.
    compared : sequence of str or None, optional
        Sources to score. Default ``None``, every other source in
        :data:`shmlib.proxies.SOURCES`.
    freq : str, optional
        Window the bias is computed over. Default ``'MS'``, calendar months.
    min_hours : int, optional
        Paired hours a window must hold to be reported. Default ``100``.
    circular : sequence of str, optional
        Quantities differenced on the circle. Default
        :data:`shmlib.proxies.CIRCULAR`.

    Returns
    -------
    pd.DataFrame
        One row per window per pair: the window start, the paired count, the
        bias, and the interquartile range of the difference within the window.
    """
    compared = ([source for source in proxies.SOURCES if source != reference]
                if compared is None else compared)
    is_circular = quantity in circular

    rows = []
    for source in compared:
        paired = _paired(df, quantity, reference, source)
        if not len(paired):
            continue
        if is_circular:
            difference = meteo.circular_difference(paired['compared'],
                                                   paired['reference'])
        else:
            difference = paired['compared'] - paired['reference']
        grouped = difference.groupby(pd.Grouper(freq=freq))
        for window, values in grouped:
            values = values.dropna()
            if len(values) < min_hours:
                continue
            rows.append({
                'quantity': quantity,
                'reference': reference,
                'compared': source,
                'window': window,
                'n': int(len(values)),
                'bias': float(values.mean()),
                'iqr': float(values.quantile(0.75) - values.quantile(0.25)),
            })
    return pd.DataFrame(rows)


def calibrate_against_sensor(df, quantity, days, reference='str',
                             compared=None, season_months=None,
                             daylight_only=True, elevation=solar.NIGHT_ELEVATION,
                             latitude=solar.SITE_LATITUDE,
                             longitude=solar.SITE_LONGITUDE, min_hours=20):
    """
    Score the external sources against the reference over the certified window.

    The same measures as :func:`pairwise_agreement`, restricted to the days on
    which the reference is trustworthy (see
    :func:`shmlib.quality.certified_sr_window`), and with a linear fit added: a
    slope and an intercept are what a documented substitution would actually
    apply, and reporting them is how the verdict states its price.

    Radiation is scored over the daylight hours by default. Half of every day is
    a substituted zero in the reference and a near-zero in the sources, and a
    statistic computed over the whole day is dominated by hours in which all
    three sources agree that it is dark — which is agreement about the position
    of the sun, not about radiation.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame in UTC.
    quantity : str
        Canonical quantity name.
    days : pd.DatetimeIndex
        The certified days, from :func:`shmlib.quality.certified_sr_window`.
    reference : str, optional
        Source scored against. Default ``'str'``.
    compared : sequence of str or None, optional
        Sources to score. Default ``None``, every other source in
        :data:`shmlib.proxies.SOURCES`.
    season_months : dict or None, optional
        Months belonging to each season. When given, a row per season is
        appended to the overall row.
    daylight_only : bool, optional
        Restrict to hours the sun is above ``elevation``. Default ``True``.
    elevation : float, optional
        Solar elevation defining daylight. Default
        :data:`shmlib.solar.NIGHT_ELEVATION`.
    latitude, longitude : float, optional
        Site coordinates. Default :data:`shmlib.solar.SITE_LATITUDE` and
        :data:`shmlib.solar.SITE_LONGITUDE`.
    min_hours : int, optional
        Hours a day must carry for the amplitude and phase columns. Default
        ``20``.

    Returns
    -------
    pd.DataFrame
        One row per source per window, carrying the agreement scores, the
        fitted slope and intercept, and the coefficient of determination.
    """
    window = quality.restrict_to_days(df, days)
    if daylight_only and len(window):
        above = solar.solar_elevation(window.index, latitude=latitude,
                                      longitude=longitude) > elevation
        window = window[np.asarray(above)]

    scored = pairwise_agreement(window, quantity, reference=reference,
                                compared=compared, min_hours=min_hours)
    if season_months is not None and len(window):
        by_season = pairwise_agreement(window, quantity, reference=reference,
                                       compared=compared,
                                       season_months=season_months,
                                       by='season', min_hours=min_hours)
        scored = pd.concat([scored, by_season], ignore_index=True)

    slopes, intercepts, r2 = [], [], []
    for _, row in scored.iterrows():
        subset = window
        if row['split'] != 'all' and season_months is not None:
            labels = site.season_of(window.index, season_months)
            subset = window[labels.values == row['split']]
        paired = _paired(subset, quantity, reference, row['compared'])
        if len(paired) < 2:
            slopes.append(np.nan)
            intercepts.append(np.nan)
            r2.append(np.nan)
            continue
        slope, intercept = np.polyfit(paired['reference'], paired['compared'], 1)
        correlation = paired['compared'].corr(paired['reference'])
        slopes.append(float(slope))
        intercepts.append(float(intercept))
        r2.append(float(correlation ** 2))

    scored['slope'] = slopes
    scored['intercept'] = intercepts
    scored['r2'] = r2
    scored['window_days'] = int(len(days))
    scored['daylight_only'] = bool(daylight_only)
    return scored
