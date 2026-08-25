"""
Module: shmlib.site

The deployment at Gubbio: which stations exist, which instrument era covers
which dates, what clock the logger keeps, and where the archive begins and ends.

These are facts about the installation rather than about any one analysis, so
they are stated once here and imported wherever they are needed. A study that
needs a different value has found a different deployment, not a different
opinion.

The two instrument eras are the single most consequential fact in this module.
The package installed on 2025-02-21 replaced the legacy network; it is different
hardware, its inclinometer shares no baseline with the legacy block at the same
station, and the two are never merged into one column. Their recorded levels are
nevertheless continuous across the changeover, so anything spanning both eras is
compensated and anchored **once**, as a single series
(``docs/raw-data-format.md`` Sections 3.3 and 7.3).
"""

import os

import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Extent and eras
# ──────────────────────────────────────────────────────────────────────

#: First day of the archive.
ARCHIVE_START = '2018-07-26'

#: Last day of the legacy 14-column era, inclusive.
LEGACY_END = '2025-02-20'

#: First day of the current 20-column era.
CURRENT_START = '2025-02-21'

#: Instrument labels. The inclinometer baseline is constant within a label and
#: meaningless across labels.
INSTRUMENT = {'legacy': 'legacy_block', 'current': 'current_package'}


# ──────────────────────────────────────────────────────────────────────
# Stations and channels
# ──────────────────────────────────────────────────────────────────────

#: Legacy stations, in block order: b1 = st01, b2 = st02, b3 = st03. The current
#: era instruments ``st02`` only.
STATIONS = ('st01', 'st02', 'st03')

#: The station this project analyses. Station 02, at the Porta di Sant'Ubaldo,
#: is the best covered and the only one spanning both instrument eras.
TARGET_STATION = 'st02'

#: Channels every era carries.
COMMON_CHANNELS = ('batt', 'tair', 'rh', 'inc')

#: Channels only the current-era package carries.
CURRENT_ONLY_CHANNELS = ('sr', 'twall')

#: The current-era package. Named for the block it occupies in the file rather
#: than for the station it stands at, because its inclinometer shares no
#: baseline with the legacy block at that station.
#:
#: Moved from study 1's ``de_lib.py``, where it was defined for that study's own
#: use, because study 2 needs the same fact and ``shmlib`` must never be reached
#: into from one study by way of another. ``de_lib.py`` keeps a thin alias at the
#: original name, so every call site there is unaffected.
CURRENT_BLOCK = 'n'


# ──────────────────────────────────────────────────────────────────────
# Clocks and grids
# ──────────────────────────────────────────────────────────────────────

#: The logger's civil clock. Measured on this archive rather than assumed: the
#: midpoint of the radiation curve on clean days sits +0.95 h from UTC in
#: February and March and +2.11 h from April onwards, stepping between the two
#: on 2025-03-30, the last Sunday of March and the European daylight-saving
#: changeover. The logger therefore records Italian civil time and does observe
#: daylight saving.
SITE_TZ = 'Europe/Rome'

#: Analysis grid. Every external proxy the pipeline aligns against is published
#: hourly, so an hourly grid is the finest resolution at which the record and
#: its drivers can be compared without inventing detail in the drivers.
ANALYSIS_FREQ = '1h'

#: Minimum number of valid raw samples required to accept an aggregated hour.
#: The raw rate is three samples per hour; requiring two keeps an hour that lost
#: a single record and rejects one built from an isolated reading.
MIN_SAMPLES_PER_HOUR = 2


# ──────────────────────────────────────────────────────────────────────
# Archive extent
# ──────────────────────────────────────────────────────────────────────

def last_archive_day(archive_dir):
    """
    Date of the most recent ``.adc`` file present in the archive.

    Lets a study discover the archive's extent instead of carrying a hard-coded
    end date that goes stale every time the acquisition system writes another
    file.

    Parameters
    ----------
    archive_dir : str
        Read-only archive directory.

    Returns
    -------
    str
        Date in ``YYYY-MM-DD`` form.

    Raises
    ------
    ValueError
        If the directory holds no ``.adc`` file.
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


# ──────────────────────────────────────────────────────────────────────
# Clock conversion, position within the day, and season labels
# ──────────────────────────────────────────────────────────────────────
#
# Moved here from study 2's now-deleted private library, where they were written for that
# study's own use: converting the archive's civil-time index to UTC before it
# is compared against the proxies' UTC timestamps, and labelling a season for
# the diurnal-cycle figures. Both state a fact about the site's clock and its
# calendar rather than a choice study 2 is making, so a later study that needs
# either finds it here instead of reaching into study 2's library.

def to_utc(index, tz=SITE_TZ, ambiguous='NaT', nonexistent='shift_forward'):
    """
    Convert a naive index kept on the site's civil clock to naive UTC.

    Study 1 measured the logger's clock against computed solar noon and
    recorded the result in the archive manifest: the acquisition system writes
    Italian civil time and does observe daylight saving. Its timestamps
    therefore mean different UTC instants in summer and winter, and the two
    hours a year the civil clock steps produce one repeated hour and one that
    never happens. Both are handled explicitly rather than allowed to raise or
    to silently pick a side: the repeated hour becomes missing, because there
    is no way to tell which of the two passes a record belongs to, and the
    hour that does not exist is shifted forward out of the gap.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Naive timestamps on the site's civil clock.
    tz : str, optional
        IANA timezone of those timestamps. Default :data:`SITE_TZ`.
    ambiguous : str, optional
        Passed to ``tz_localize`` for the repeated autumn hour. Default
        ``'NaT'``.
    nonexistent : str, optional
        Passed to ``tz_localize`` for the skipped spring hour. Default
        ``'shift_forward'``.

    Returns
    -------
    pd.DatetimeIndex
        Naive UTC timestamps, with ``NaT`` wherever the civil timestamp was
        ambiguous.
    """
    localised = pd.DatetimeIndex(index).tz_localize(
        tz, ambiguous=ambiguous, nonexistent=nonexistent)
    return localised.tz_convert('UTC').tz_localize(None)


def time_of_day(index):
    """
    Position within the day, in fractional hours.

    On a sub-hourly grid the hour alone no longer identifies a position in the
    day, since several samples share it. This returns the hour plus the
    fraction of it that has elapsed — on study 1's native twenty-minute grid,
    0.0, 0.333, 0.667, 1.0 and so on — which is unique per slot and still
    reads directly against an hour axis.

    Moved here from study 1's ``de_lib.py``, where it was written for that
    study's own twenty-minute grid, because study 2 needs the identical
    computation for its hourly diurnal profiles and ``shmlib`` must never be
    reached into from one study by way of another. ``de_lib.py`` keeps a thin
    alias at the original name, so every call site there is unaffected.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Timestamps to reduce.

    Returns
    -------
    np.ndarray of float
        Fractional hours in ``[0, 24)``.
    """
    return index.hour + index.minute / 60.0 + index.second / 3600.0


def season_of(index, season_months):
    """
    Season label of every timestamp, from a months-to-season mapping.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Timestamps to label.
    season_months : dict of str to sequence of int
        Months belonging to each season.

    Returns
    -------
    pd.Series
        Season name per timestamp, indexed by ``index``. Months named by no
        season become missing.
    """
    lookup = {month: season
              for season, months in season_months.items()
              for month in months}
    return pd.Series(pd.DatetimeIndex(index).month, index=index).map(lookup)
