"""
Module: shmlib.quality

Verdicts on the record, and the windows they leave.

Where :mod:`shmlib.proxies` states what a source is and how to load it, this
module states what happened to one channel's samples and days: the flag
vocabulary a rejection or a correction is written in, the census of what became
of every calendar day of the on-structure radiation channel's life, the
narrower window of days that survives every verdict and is therefore fit to
calibrate against, and the two independent tests of whether a source keeps the
clock it claims to.

The flag vocabulary was moved here from study 1's ``de_lib.py``: a rejection
empties the corrected column because nothing is known in the value's place, a
correction substitutes a known value, and every study that reads a ``_flag``
column downstream of study 1 needs to agree on what a code in it means.
``de_lib.py`` keeps a thin alias at each original name, so every call site
there is unaffected.

``load_sr_record``, ``certified_sr_window``, ``restrict_to_days``,
``sr_day_census``, ``clock_check`` and ``verdict_table`` were moved here from
study 2's now-deleted private library, where they were written for that study's own
characterisation of the on-structure radiation channel. They state facts about
the record's verdicts rather than a choice study 2 is making, so a later study
that needs the same census or the same certified window finds it here instead
of reaching into study 2's library.

Nothing here re-decides a rejection study 1 already made, and nothing here
computes how two sources agree — that is :mod:`shmlib.compare`.
"""

import numpy as np
import pandas as pd

from . import coupling, proxies, site, solar


# ──────────────────────────────────────────────────────────────────────
# The flag vocabulary
# ──────────────────────────────────────────────────────────────────────
#
# Moved from study 1's ``de_lib.py``, lines 117-140, because study 2's
# characterisation of the on-structure radiation channel needs the identical
# codes and ``shmlib`` must never be reached into from one study by way of
# another.

#: Rejection codes written into the ``_flag`` columns. A rejected value is
#: always ``NaN`` in the matching ``_ok`` column, and the code says why.
FLAG_SENTINEL = 'sentinel'
FLAG_WRAP = 'wrap'
FLAG_UNPHYSICAL = 'unphysical'

#: Correction codes. Unlike a rejection, a correction knows what the value
#: should have been, so the ``_ok`` column carries the substituted value rather
#: than a gap. Only one exists: radiation recorded while the sun is below the
#: horizon, whose true value is zero.
FLAG_NIGHT = 'night'

#: Every code that empties the corrected column.
#:
#: The calibrated range of the inclinometer conditioner is deliberately not a
#: rejection criterion here. The certificate covers +/- 2 degrees about the
#: 2500 mV zero, but the recorded deflections leave that interval, so a test
#: built on it would discard readings the instrument demonstrably produced
#: rather than markers the acquisition system wrote. Impulsive excursions on the
#: inclinometer are handled where they belong, by the Hampel filter of study
#: 1's ``de_lib.hampel_filter``, which judges a sample against its own
#: neighbourhood instead of against a certificate the record has already
#: outgrown.
REJECTION_CODES = (FLAG_SENTINEL, FLAG_WRAP, FLAG_UNPHYSICAL)

#: Every code that substitutes a known value into the corrected column.
CORRECTION_CODES = (FLAG_NIGHT,)


# ──────────────────────────────────────────────────────────────────────
# The certified radiation window
# ──────────────────────────────────────────────────────────────────────

def certified_sr_window(sensor, column='sr_str', min_hours=1):
    """
    The days on which the on-structure radiation survives every study 1 verdict.

    Study 1 rejected radiation samples for three reasons, corrected a fourth,
    and then condemned 161 whole days on which the channel showed no diurnal
    cycle at all. What is left is short, but it is the only measurement in the
    project that sees the sky this wall sees, and every calibration in the study
    is conditioned on it. Its extent is therefore computed rather than assumed.

    The days are contiguous only by accident, so the gaps between them are
    reported alongside the count. A window advertised as five months that is in
    fact a dozen fragments supports a different claim from one that is not.

    Parameters
    ----------
    sensor : pd.DataFrame
        On-structure frame from :func:`shmlib.proxies.load_sensor_forcings`, on
        which the condemned days are already missing when ``honour_suspect``
        was ``True``.
    column : str, optional
        Radiation channel. Default ``'sr_str'``.
    min_hours : int, optional
        Hours a day must carry to count as certified. Default ``1``.

    Returns
    -------
    days : pd.DatetimeIndex
        The certified days, ascending.
    summary : dict
        Extent, day count, calendar span, the fraction of that span the days
        cover, the number of separate runs they fall into, and the longest run
        and longest gap in days.
    """
    if column not in sensor.columns:
        return pd.DatetimeIndex([]), {'n_days': 0}

    values = pd.to_numeric(sensor[column], errors='coerce')
    per_day = values.groupby(pd.DatetimeIndex(values.index).normalize()).count()
    days = pd.DatetimeIndex(per_day[per_day >= min_hours].index).sort_values()

    if not len(days):
        return days, {'n_days': 0}

    first, last = days.min(), days.max()
    span_days = int((last - first).days) + 1
    steps = pd.Series(days).diff().dt.days.fillna(1).astype(int)
    run_id = (steps != 1).cumsum()
    runs = pd.Series(days).groupby(run_id).size()
    gaps = steps[steps > 1] - 1

    summary = {
        'first': first,
        'last': last,
        'n_days': int(len(days)),
        'span_days': span_days,
        'coverage_pct': 100.0 * len(days) / span_days,
        'n_runs': int(len(runs)),
        'longest_run_days': int(runs.max()),
        'n_gaps': int(len(gaps)),
        'longest_gap_days': int(gaps.max()) if len(gaps) else 0,
        'n_hours': int(values.loc[np.isin(
            pd.DatetimeIndex(values.index).normalize(), days)].count()),
    }
    return days, summary


def restrict_to_days(df, days):
    """
    The rows of a frame that fall on a given set of days.

    Parameters
    ----------
    df : pd.DataFrame
        Frame on the analysis grid.
    days : pd.DatetimeIndex
        Days to keep, at midnight.

    Returns
    -------
    pd.DataFrame
        The rows falling on those days. The frame is not copied deeply; callers
        that modify the result should copy it first.
    """
    if not len(days):
        return df.iloc[:0]
    return df[np.isin(pd.DatetimeIndex(df.index).normalize(), days)]


def load_sr_record(path, freq=site.ANALYSIS_FREQ, tz=site.SITE_TZ,
                   min_count=site.MIN_SAMPLES_PER_HOUR, block=site.CURRENT_BLOCK,
                   suspect_column=proxies.SUSPECT_COLUMN,
                   era_start=site.CURRENT_START):
    """
    The radiation channel with its verdicts beside it, before the verdicts bite.

    :func:`shmlib.proxies.load_sensor_forcings` returns the channel study 1's
    decisions have already been applied to, which is what the comparison needs
    and is exactly what a census of those decisions cannot use — the condemned
    days are gone from it. This reads the same column again, unfiltered, and
    carries the per-sample flag and the day-level verdict alongside it.

    It puts all three on the same grid, by the same rule, as
    :func:`shmlib.proxies.load_sensor_forcings`. That is the whole point of the
    function: the certified window and the census of what became of every day
    are two statements about one thing, and computing them on two different
    grids is how a report ends up quoting two different day counts for it.

    The era boundary is converted to UTC before it is used to cut. The
    changeover is recorded as a civil-time date, and applying it to an index
    that has already been converted would move the first day of the current era
    by an hour and drop it.

    Parameters
    ----------
    path : str
        Path to ``gubbio_archive_20min.csv``.
    freq : str, optional
        Analysis grid. Default :data:`shmlib.site.ANALYSIS_FREQ`.
    tz : str, optional
        The logger's civil clock. Default :data:`shmlib.site.SITE_TZ`.
    min_count : int, optional
        Twenty-minute samples an hour must carry. Default
        :data:`shmlib.site.MIN_SAMPLES_PER_HOUR`.
    block : str, optional
        Acquisition block carrying the channel. Default
        :data:`shmlib.site.CURRENT_BLOCK`.
    suspect_column : str, optional
        Study 1's day-level verdict. Default
        :data:`shmlib.proxies.SUSPECT_COLUMN`.
    era_start : str, optional
        First day of the era, on the site's civil clock. Default
        :data:`shmlib.site.CURRENT_START`.

    Returns
    -------
    pd.DataFrame
        Indexed by the analysis grid in UTC, from the changeover onwards, with
        three columns: ``sr``, the corrected value with no verdict applied;
        ``suspect``, true on an hour of a day study 1 condemned; and ``flag``,
        carrying :data:`FLAG_NIGHT` where any sample of the hour was a
        substituted zero.
    """
    value_column = f'{block}_sr_ok'
    flag_column = f'{block}_sr_flag'

    raw = pd.read_csv(path, index_col=0, parse_dates=True,
                      usecols=['datetime', value_column, flag_column,
                               suspect_column], low_memory=False)
    raw = raw.sort_index()

    boundary = site.to_utc(pd.DatetimeIndex([pd.Timestamp(era_start)]),
                           tz=tz)[0]
    raw.index = site.to_utc(raw.index, tz=tz)
    raw = raw[raw.index.notna()].sort_index()
    raw = raw.loc[raw.index >= boundary]

    values = pd.to_numeric(raw[value_column], errors='coerce')
    resampler = values.resample(freq)
    hourly = resampler.mean().where(resampler.count() >= min_count)

    suspect = raw[suspect_column].fillna(False).astype(bool)
    suspect_hourly = suspect.resample(freq).max().fillna(False).astype(bool)

    is_night = (raw[flag_column] == FLAG_NIGHT)
    night_hourly = is_night.resample(freq).max().fillna(False).astype(bool)

    out = pd.DataFrame({
        'sr': hourly,
        'suspect': suspect_hourly.reindex(hourly.index).fillna(False),
        'flag': np.where(night_hourly.reindex(hourly.index).fillna(False),
                         FLAG_NIGHT, ''),
    }, index=hourly.index)
    out.attrs['era_start_utc'] = boundary
    return out


def sr_day_census(sensor_raw, suspect, flag, night_flag=FLAG_NIGHT):
    """
    Every day of the radiation channel's life, and what became of it.

    Partitions the current era into the days that carry no record, the days
    study 1 condemned, and the days that survived, so that the certified window
    can be read as a fraction of what existed rather than as a bare number.

    Parameters
    ----------
    sensor_raw : pd.Series
        The corrected radiation channel *before* the suspect verdict is applied,
        on the analysis grid.
    suspect : pd.Series
        Study 1's day-level verdict, aligned to ``sensor_raw``.
    flag : pd.Series
        Study 1's per-sample flag for the radiation channel, aligned to
        ``sensor_raw``.
    night_flag : str, optional
        Flag code marking a substituted zero. Default :data:`FLAG_NIGHT`.

    Returns
    -------
    pd.DataFrame
        One row per calendar day: ``n_hours`` recorded, ``n_surviving`` left
        after the verdict, ``n_condemned`` removed by it, ``n_night`` carrying
        the substituted zero, ``condemned`` where the verdict touched the day at
        all, and the resulting ``state``.

    Notes
    -----
    The state is decided on the **surviving** hours, not on whether the verdict
    touched the day. The distinction is not pedantic here. ``sr_suspect`` is a
    verdict on a *civil* calendar day, and this census is taken on a UTC index,
    so a condemned civil day lands across two UTC days and leaves each of them
    part condemned and part intact. Calling such a day condemned would throw
    away hours that survived every test, and would disagree with
    :func:`certified_sr_window`, which counts exactly those surviving hours. A
    day is therefore condemned only when nothing of it is left, and the partly
    condemned days are visible in ``n_condemned`` beside ``n_surviving``.
    """
    values = pd.to_numeric(sensor_raw, errors='coerce')
    day = pd.DatetimeIndex(values.index).normalize()
    frame = pd.DataFrame({
        'value': values,
        'suspect': suspect.reindex(values.index).fillna(False).astype(bool),
        'flag': flag.reindex(values.index),
    })
    frame['surviving'] = frame['value'].where(~frame['suspect'])
    frame['condemned_hour'] = frame['value'].where(frame['suspect'])
    grouped = frame.groupby(day)

    census = pd.DataFrame({
        'n_hours': grouped['value'].count(),
        'n_surviving': grouped['surviving'].count(),
        'n_condemned': grouped['condemned_hour'].count(),
        'n_night': grouped['flag'].apply(lambda s: int((s == night_flag).sum())),
        'condemned': grouped['suspect'].any(),
    })
    census.index.name = 'day'

    census['state'] = np.where(
        census['n_hours'] == 0, 'no record',
        np.where(census['n_surviving'] == 0, 'condemned', 'certified'))
    return census.reset_index()


# ──────────────────────────────────────────────────────────────────────
# The clock each source keeps
# ──────────────────────────────────────────────────────────────────────

#: Paired hours the relative clock test needs at a lag before it scores that
#: lag. One day: a shorter overlap says more about which hours happened to
#: survive than about the offset between two clocks.
_CLOCK_MIN_PAIRED = 24


def clock_check(df, quantity, sources=proxies.SOURCES, reference='era5',
                min_daily_peak=200.0, threshold=50.0, max_lag_hours=12,
                latitude=solar.SITE_LATITUDE, longitude=solar.SITE_LONGITUDE):
    """
    Two independent tests of the clock each source keeps, run on one quantity.

    A source that keeps a different clock cannot be compared on a shared index
    at all, and the offset between two clocks is not something a correlation
    will reveal — a series shifted by an hour still correlates well with itself.
    So two tests are run and both are reported.

    The **absolute** test locates the midpoint between the first and last
    crossing of a small threshold on each usable day and compares it with solar
    noon computed from the site longitude and the equation of time. Its answer
    is in hours from UTC and does not depend on any other source.

    The **relative** test cross-correlates the source against a reference source
    over integer-hour lags and reports the lag of maximum correlation. Its answer
    is in hours from that reference.

    Where the two disagree for a channel, that channel is defective, and the
    report says so rather than averaging the two into a number that describes
    neither. The on-structure radiation is the known case: study 1 recorded that
    it reads roughly sixty watts at midnight and places its own solar noon after
    its temperature maximum.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame in UTC.
    quantity : str
        Quantity to test. Only a quantity with a sharp daily cycle can carry
        this test, which in practice means solar radiation.
    sources : sequence of str, optional
        Sources to test. Default :data:`shmlib.proxies.SOURCES`.
    reference : str, optional
        Source the relative test is measured against. Default ``'era5'``, the
        only source whose timestamps are stated as UTC by its provider.
    min_daily_peak : float, optional
        A day must reach this value to enter the absolute test. Default
        ``200.0``, matching study 1.
    threshold : float, optional
        Crossing level defining the start and end of the day. Default ``50.0``,
        matching study 1.
    max_lag_hours : int, optional
        Largest lag searched by the relative test, in hours. Default ``12``.
    latitude, longitude : float, optional
        Site coordinates for the solar geometry. Default
        :data:`shmlib.solar.SITE_LATITUDE` and
        :data:`shmlib.solar.SITE_LONGITUDE`.

    Returns
    -------
    pd.DataFrame
        One row per source: the median solar-noon offset in hours, its spread,
        the days it rests on, the cross-correlation lag against ``reference``,
        the correlation at that lag, and whether the two tests agree to within
        an hour.
    """
    reference_series = df.get(f'{quantity}_{reference}')

    rows = []
    for source in sources:
        column = f'{quantity}_{source}'
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors='coerce').dropna()

        offsets = []
        for day, group in series.groupby(pd.DatetimeIndex(series.index).normalize()):
            if group.max() < min_daily_peak:
                continue
            lit = group[group > threshold]
            if len(lit) < 3:
                continue
            first, last = lit.index[0], lit.index[-1]
            midpoint = ((first.hour + first.minute / 60.0)
                        + (last.hour + last.minute / 60.0)) / 2.0
            noon = float(solar.solar_noon_utc([day], longitude=longitude).iloc[0])
            offsets.append(midpoint - noon)
        offsets = pd.Series(offsets, dtype=float)

        lag, correlation = np.nan, np.nan
        if reference_series is not None and source != reference:
            # The search itself lives in ``shmlib.coupling``, which the thermal
            # coupling of study 3 uses over a different range and with a
            # different objective. An alignment between two records of one
            # quantity is the signed case: the shift wanted is the one that
            # makes them agree, not the one that makes them move together most
            # strongly in either direction.
            curve = coupling.lag_scan(reference_series, df[column],
                                      range(-max_lag_hours, max_lag_hours + 1),
                                      min_paired=_CLOCK_MIN_PAIRED)
            winner = coupling.best_lag(curve, objective='signed')
            lag, correlation = winner['lag'], winner['r']
        elif source == reference:
            lag, correlation = 0.0, 1.0

        solar_offset = float(offsets.median()) if len(offsets) else np.nan
        agree = (np.nan if (np.isnan(solar_offset) or np.isnan(lag))
                 else bool(abs(solar_offset - lag) < 1.0))
        rows.append({
            'source': source,
            'channel': column,
            'n_days_solar': int(len(offsets)),
            'solar_offset_h': solar_offset,
            'solar_offset_iqr_h': (float(offsets.quantile(0.75)
                                         - offsets.quantile(0.25))
                                   if len(offsets) else np.nan),
            'xcorr_lag_h': lag,
            'xcorr_r': correlation,
            'tests_agree': agree,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# The verdict
# ──────────────────────────────────────────────────────────────────────

def verdict_table(rows):
    """
    The study's conclusion, one row per quantity and source.

    A table rather than a paragraph because the verdict has to be quotable by
    later studies without their author re-reading the report. Each row is
    written by the notebook from the numbers above it: this function only fixes
    the columns and their order, so that no verdict can be recorded without
    stating the window its evidence covers and the limitation it carries.

    Parameters
    ----------
    rows : sequence of dict
        One dict per verdict, carrying at least ``quantity``, ``source``,
        ``verdict``, ``transformation``, ``evidence_window`` and
        ``limitation``.

    Returns
    -------
    pd.DataFrame
        The verdicts, with the columns in reporting order.

    Raises
    ------
    ValueError
        If any row omits a required field. A verdict missing its evidence
        window is not a weaker verdict, it is an unsupported one.
    """
    required = ('quantity', 'source', 'verdict', 'transformation',
                'evidence_window', 'limitation')
    for index, row in enumerate(rows):
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(f'verdict row {index} is missing {missing}')
    return pd.DataFrame(list(rows), columns=list(required))
